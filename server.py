from __future__ import annotations

import logging
import os
import time
import hmac
import hashlib
import requests
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated

from src.config import load_settings
from src.messenger.conversation_flow import ConversationFlowManager
from src.pipeline.rag_pipeline import RAGPipeline
from src.whatsapp.dedupe import RecentMessageCache
import src.whatsapp.client as wa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG pipeline instance
rag: RAGPipeline | None = None
settings = None
startup_error: str | None = None

# Load environment variables from .env file immediately
load_dotenv()

# Basic in-memory rate limiter (per session_id)
MAX_REQUESTS_PER_MINUTE = 20
RATE_WINDOW_SECONDS = 60
_rate_limits: dict[str, list[float]] = {}
conversation_flow = ConversationFlowManager(state_dir="artifacts/flow_state")
recent_whatsapp_messages = RecentMessageCache(ttl_seconds=300)
recent_messenger_messages = RecentMessageCache(ttl_seconds=300)

# WhatsApp Cloud API env vars still needed by server.py for webhook validation
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip()
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "").strip()

# Messenger (Meta) Cloud API env vars
MESSENGER_ACCESS_TOKEN = os.getenv("MESSENGER_ACCESS_TOKEN", "").strip()
MESSENGER_VERIFY_TOKEN = os.getenv("MESSENGER_VERIFY_TOKEN", "").strip()
MESSENGER_GRAPH_VERSION = os.getenv("MESSENGER_GRAPH_VERSION", "v23.0").strip()

# Public HTTPS base URL used when sending images to WhatsApp / Messenger
# Example: https://wmcnc.yourdomain.com
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()


def _allow_request(session_id: str) -> bool:
    now = time.time()
    window_start = now - RATE_WINDOW_SECONDS
    entries = _rate_limits.setdefault(session_id, [])
    entries[:] = [ts for ts in entries if ts >= window_start]
    if len(entries) >= MAX_REQUESTS_PER_MINUTE:
        return False
    entries.append(now)
    return True


def _try_initialize_rag() -> bool:
    global rag, settings, startup_error
    if rag is not None:
        return True
    try:
        settings = load_settings()
        rag = RAGPipeline(settings)
        rag.initialize(force_rebuild=False)
        startup_error = None
        logger.info("RAG Pipeline initialized via lazy recovery.")
        return True
    except Exception as exc:
        startup_error = f"{exc.__class__.__name__}: {exc}"
        logger.exception("Lazy recovery initialization failed")
        return False


def _verify_whatsapp_signature(raw_body: bytes, signature_header: str | None) -> bool:
    if not WHATSAPP_APP_SECRET:
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(
        WHATSAPP_APP_SECRET.encode("utf-8"),
        msg=raw_body,
        digestmod=hashlib.sha256,
    ).hexdigest()
    received = signature_header.split("=", 1)[1]
    return hmac.compare_digest(expected, received)


def _extract_whatsapp_messages(payload: dict) -> list[dict]:
    items: list[dict] = []
    entries = payload.get("entry", []) if isinstance(payload, dict) else []
    for entry in entries:
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for message in value.get("messages", []):
                items.append(message)
    return items


def _is_duplicate_whatsapp_message(session_id: str, message: dict) -> bool:
    message_id = str(message.get("id") or "").strip()
    text = _extract_text_from_message(message)
    dedupe_key = message_id or f"{session_id}:{text.lower()}"
    return recent_whatsapp_messages.seen_recently(dedupe_key)


def _is_duplicate_messenger_message(session_id: str, event: dict, text: str) -> bool:
    message_id = str((event.get("message") or {}).get("mid") or "").strip()
    dedupe_key = message_id or f"{session_id}:{text.lower()}"
    return recent_messenger_messages.seen_recently(dedupe_key)


def _extract_text_from_message(message: dict) -> str:
    mtype = message.get("type")
    if mtype == "text":
        return (message.get("text", {}) or {}).get("body", "").strip()
    if mtype == "interactive":
        interactive = message.get("interactive", {}) or {}
        if "button_reply" in interactive:
            return (interactive["button_reply"].get("title") or "").strip()
        if "list_reply" in interactive:
            return (interactive["list_reply"].get("title") or "").strip()
    return ""


async def _send_public_images_whatsapp(to_number: str, image_paths: list[str]) -> None:
    if not PUBLIC_BASE_URL:
        return
    base_url = PUBLIC_BASE_URL.rstrip("/")
    for img_path in image_paths:
        public_url = f"{base_url}/{img_path}"
        await run_in_threadpool(wa.send_image, to_number, public_url)


async def _send_whatsapp_flow_response(to_number: str, reply: str, options: list[dict] | None, images: list[str] | None) -> None:
    if options:
        await run_in_threadpool(wa.send_interactive_buttons, to_number, reply, options)
    elif reply:
        await run_in_threadpool(wa.send_text, to_number, reply)

    if images:
        await _send_public_images_whatsapp(to_number, images)


def _send_messenger_text(psid: str, text: str) -> None:
    """Send a plain text message via Messenger Send API."""
    if not MESSENGER_ACCESS_TOKEN:
        logger.error("MESSENGER_ACCESS_TOKEN is missing. Cannot send Messenger message.")
        return
    url = f"https://graph.facebook.com/{MESSENGER_GRAPH_VERSION}/me/messages"
    resp = requests.post(
        url,
        json={
            "recipient": {"id": psid},
            "message": {"text": text[:2000]},
            "messaging_type": "RESPONSE",
        },
        params={"access_token": MESSENGER_ACCESS_TOKEN},
        headers={"Content-Type": "application/json"},
        timeout=20,
    )
    if resp.status_code >= 300:
        logger.error("Messenger text send failed: status=%s body=%s", resp.status_code, resp.text)


def _send_messenger_quick_replies(psid: str, text: str, options: list[dict]) -> None:
    """Send quick reply buttons — Messenger equivalent of WhatsApp interactive buttons."""
    if not MESSENGER_ACCESS_TOKEN:
        logger.error("MESSENGER_ACCESS_TOKEN is missing. Cannot send Messenger quick replies.")
        return
    quick_replies = [
        {"content_type": "text", "title": opt["label"], "payload": opt["value"]}
        for opt in options
    ]
    url = f"https://graph.facebook.com/{MESSENGER_GRAPH_VERSION}/me/messages"
    resp = requests.post(
        url,
        json={
            "recipient": {"id": psid},
            "message": {"text": text, "quick_replies": quick_replies},
            "messaging_type": "RESPONSE",
        },
        params={"access_token": MESSENGER_ACCESS_TOKEN},
        headers={"Content-Type": "application/json"},
        timeout=20,
    )
    if resp.status_code >= 300:
        logger.error("Messenger quick reply send failed: status=%s body=%s", resp.status_code, resp.text)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag, settings, startup_error
    load_dotenv()
    startup_error = None

    try:
        settings = load_settings()
        rag = RAGPipeline(settings)
        logger.info("Initializing RAG Pipeline...")
        rag.initialize(force_rebuild=False)
        logger.info("RAG Pipeline initialized.")
    except Exception as exc:
        rag = None
        startup_error = f"{exc.__class__.__name__}: {exc}"
        logger.exception("Startup initialization failed")

    yield
    logger.info("Shutting down...")

app = FastAPI(title="Woodmaster CNC Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve only images publicly. Keep all other data (for example sessions) private.
images_dir = Path("data") / "images"
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/data/images", StaticFiles(directory=str(images_dir)), name="images")

class WebhookRequest(BaseModel):
    session_id: Annotated[
        str,
        StringConstraints(pattern=r"^[A-Za-z0-9:_+\-]{3,64}$", strip_whitespace=True),
    ]
    message: Annotated[str, Field(min_length=1, max_length=2000)]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "woodmaster-cnc-assistant",
    }


@app.get("/ready")
async def ready():
    if rag:
        return {"status": "ready"}
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "detail": startup_error or "RAG pipeline not initialized"},
    )

@app.post("/webhook")
async def handle_webhook(req: WebhookRequest):
    if not _allow_request(req.session_id):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": "Too many requests for this session."},
        )

    if not rag:
        recovered = await run_in_threadpool(_try_initialize_rag)
        if not recovered:
            return JSONResponse(
                status_code=503,
                content={"error": "RAG pipeline not initialized", "detail": startup_error or "Initialization pending"},
            )

    if not rag:
        return JSONResponse(
            status_code=503,
            content={"error": "RAG pipeline not initialized", "detail": startup_error or "Initialization pending"},
        )

    flow_result = conversation_flow.handle_message(req.session_id, req.message)
    
    if flow_result.timeout_occurred and rag:
        rag.memory_manager.clear(req.session_id)

    if flow_result.handled:
        return {
            "session_id": req.session_id,
            "reply": flow_result.reply,
            "images": flow_result.images or [],
            "options": flow_result.options or [],
            "metadata": [],
        }
    
    try:
        response = await run_in_threadpool(
            rag.query,
            req.message,
            req.session_id,
            flow_result.preferred_language,
        )
        
        final_reply = response.get("answer", "")
        if flow_result.reply:
            final_reply = f"{flow_result.reply}\n\n{final_reply}"
            
        final_images = (flow_result.images or []) + response.get("images", [])

        # response should contain 'answer', 'sources', 'images'
        return {
            "session_id": req.session_id,
            "reply": final_reply.strip(),
            "images": final_images,
            "options": [],
            "metadata": response.get("retrieval", [])
        }
    except Exception as exc:
        logger.exception(
            "Webhook query failed",
            extra={"session_id": req.session_id, "message_length": len(req.message)},
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc.__class__.__name__)},
        )


@app.get("/whatsapp/webhook")
async def verify_whatsapp_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge", "")

    if mode == "subscribe" and token and token == WHATSAPP_VERIFY_TOKEN:
        return HTMLResponse(content=challenge, status_code=200)

    return JSONResponse(status_code=403, content={"error": "Webhook verification failed"})


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    if not rag:
        recovered = await run_in_threadpool(_try_initialize_rag)
        if not recovered:
            return JSONResponse(status_code=503, content={"error": "RAG pipeline unavailable"})

    raw_body = await request.body()
    signature = request.headers.get("x-hub-signature-256")
    if not _verify_whatsapp_signature(raw_body, signature):
        return JSONResponse(status_code=401, content={"error": "Invalid signature"})

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    messages = _extract_whatsapp_messages(payload)
    if not messages:
        return {"status": "ignored"}

    for message in messages:
        from_number = str(message.get("from", "")).strip()
        if not from_number:
            continue
        session_id = f"whatsapp:{from_number}"
        if _is_duplicate_whatsapp_message(session_id, message):
            continue

        user_text = _extract_text_from_message(message)
        if not user_text:
            continue

        if not _allow_request(session_id):
            await run_in_threadpool(
                wa.send_text, from_number,
                "Too many requests. Please wait a minute and try again."
            )
            continue

        # Step 1: Run conversation flow FIRST (greeting → language selection → chat)
        flow_result = conversation_flow.handle_message(session_id, user_text)

        if flow_result.timeout_occurred and rag:
            rag.memory_manager.clear(session_id)

        if flow_result.reply and not flow_result.handled:
            await run_in_threadpool(wa.send_text, from_number, flow_result.reply)
            await _send_public_images_whatsapp(from_number, flow_result.images or [])

        if flow_result.handled:
            await _send_whatsapp_flow_response(
                from_number,
                flow_result.reply,
                flow_result.options,
                flow_result.images,
            )
            continue

        # Step 2: RAG query with the user's stored language preference
        try:
            response = await run_in_threadpool(
                rag.query, user_text, session_id, flow_result.preferred_language
            )
            answer = (response.get("answer") or "I could not generate a response right now.").strip()
            await run_in_threadpool(wa.send_text, from_number, answer)
            await _send_public_images_whatsapp(from_number, response.get("images", []))
        except Exception:
            logger.exception("WhatsApp processing failed", extra={"session_id": session_id})
            await run_in_threadpool(
                wa.send_text, from_number,
                "Sorry, I hit an internal error. Please try again.",
            )

    return {"status": "ok"}


@app.get("/messenger/webhook")
async def verify_messenger_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge", "")
    if mode == "subscribe" and token and token == MESSENGER_VERIFY_TOKEN:
        return HTMLResponse(content=challenge, status_code=200)
    return JSONResponse(status_code=403, content={"error": "Messenger webhook verification failed"})


@app.post("/messenger/webhook")
async def messenger_webhook(request: Request):
    if not rag:
        recovered = await run_in_threadpool(_try_initialize_rag)
        if not recovered:
            return JSONResponse(status_code=503, content={"error": "RAG pipeline unavailable"})

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    for entry in payload.get("entry", []):
        for event in entry.get("messaging", []):
            psid = event.get("sender", {}).get("id", "").strip()
            msg = event.get("message", {})
            # Handle both plain text AND quick_reply button taps
            text = (
                (msg.get("quick_reply") or {}).get("payload")
                or msg.get("text")
                or ""
            ).strip()
            if not psid or not text:
                continue

            session_id = f"messenger:{psid}"
            if _is_duplicate_messenger_message(session_id, event, text):
                continue
            if not _allow_request(session_id):
                await run_in_threadpool(
                    _send_messenger_text, psid,
                    "Too many requests. Please wait a minute and try again."
                )
                continue

            # Same 3-stage flow as web frontend and WhatsApp
            flow_result = conversation_flow.handle_message(session_id, text)

            if flow_result.timeout_occurred and rag:
                rag.memory_manager.clear(session_id)

            if flow_result.reply and not flow_result.handled:
                # We have a greeting to send before the RAG generated answer
                await run_in_threadpool(_send_messenger_text, psid, flow_result.reply)
                # Note: Currently not sending images for Messenger early-greeting as it is basic,
                # but could be added similar to WhatsApp.

            if flow_result.handled:
                if flow_result.options:
                    # Use quick replies for language selection on Messenger
                    await run_in_threadpool(
                        _send_messenger_quick_replies,
                        psid, flow_result.reply, flow_result.options,
                    )
                else:
                    await run_in_threadpool(_send_messenger_text, psid, flow_result.reply)
                continue

            try:
                response = await run_in_threadpool(
                    rag.query, text, session_id, flow_result.preferred_language
                )
                answer = (response.get("answer") or "I could not generate a response.").strip()
                await run_in_threadpool(_send_messenger_text, psid, answer)
            except Exception:
                logger.exception("Messenger processing failed", extra={"session_id": session_id})
                await run_in_threadpool(
                    _send_messenger_text, psid,
                    "Sorry, I hit an internal error. Please try again."
                )

    return {"status": "ok"}


# Serve the simple static index.html on root
@app.get("/")
async def get_index():
    if os.path.exists("frontend.html"):
        with open("frontend.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
