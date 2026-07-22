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

from src.api.router import api_router
from src.config import load_settings
from src.db.client import DatabaseDisabledError, close_db_client, is_db_enabled, run_sql_file
from src.db import messages as db_messages
from src.messenger.conversation_flow import ConversationFlowManager
from src.lead.scorer import score_conversation
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


def _try_initialize_database() -> None:
    if not is_db_enabled():
        if os.getenv("REQUIRE_DATABASE", "true").strip().lower() in {"1", "true", "yes", "on"}:
            raise RuntimeError("Persistent conversation state requires DB_ENABLED=true and DATABASE_URL or DATABASE_POOLER_URL.")
        logger.warning("Database is disabled; conversation state will not survive process restarts.")
        return
    if os.getenv("REQUIRE_SALES_CONTACT", "true").strip().lower() in {"1", "true", "yes", "on"}:
        if not os.getenv("SALES_PHONE", "").strip() or not os.getenv("SALES_EMAIL", "").strip():
            raise RuntimeError("SALES_PHONE and SALES_EMAIL must be configured for the sales handoff.")
    run_sql_file()
    logger.info("Database migrations verified.")
    try:
        from src.db import qna as db_qna

        seeded = db_qna.seed_from_markdown_if_empty(Path(__file__).resolve().parent / "data" / "knowledge.md")
        if seeded:
            logger.info("Seeded %d Q&A entries from data/knowledge.md.", seeded)
    except Exception:
        logger.exception("Failed to seed Q&A entries from data/knowledge.md")


def _score_if_needed(conversation_id: str | None) -> None:
    if not conversation_id or not is_db_enabled():
        return
    every_n = int(os.getenv("LEAD_SCORE_EVERY_N_MESSAGES", "3"))
    try:
        bot_count = db_messages.get_message_count(conversation_id, sender="BOT")
        if bot_count == 0 or bot_count % max(1, every_n) != 0:
            return
        score_conversation(conversation_id)
    except DatabaseDisabledError:
        return
    except Exception:
        logger.exception("Lead scoring failed", extra={"conversation_id": conversation_id})


def _record_bot_reply(
    conversation_id: str | None,
    reply: str,
    language: str | None = None,
    tokens: int | None = None,
    model: str | None = None,
    latency_ms: int | None = None,
) -> None:
    if not conversation_id or not reply or not is_db_enabled():
        return
    try:
        db_messages.append_message(
            conversation_id,
            "BOT",
            reply,
            language=language,
            tokens=tokens,
            model=model,
            latency_ms=latency_ms,
        )
        _score_if_needed(conversation_id)
    except DatabaseDisabledError:
        return
    except Exception:
        logger.exception("Failed to record bot reply", extra={"conversation_id": conversation_id})


def _record_bot_images(conversation_id: str | None, image_paths: list[str]) -> None:
    """Log each image the AI actually sent so the dashboard thread reflects it."""
    if not conversation_id or not image_paths or not is_db_enabled():
        return
    base_url = PUBLIC_BASE_URL.rstrip("/") if PUBLIC_BASE_URL else ""
    try:
        for img_path in image_paths:
            image_url = f"{base_url}/{img_path}" if base_url else img_path
            db_messages.append_message(conversation_id, "BOT", image_url, message_type="image")
    except DatabaseDisabledError:
        return
    except Exception:
        logger.exception("Failed to record bot image", extra={"conversation_id": conversation_id})


def _add_sales_contact_prompt(reply: str, flow_result) -> str:
    """Append the one-time sales handoff after the configured chat-turn threshold."""
    if not flow_result.contact_forced:
        return reply

    phone = os.getenv("SALES_PHONE", "").strip()
    email = os.getenv("SALES_EMAIL", "").strip()
    language = flow_result.preferred_language
    if language == "Hindi":
        handoff = f"अधिक व्यक्तिगत सहायता या कोटेशन के लिए हमारी सेल्स टीम से संपर्क करें: {phone} | {email}"
    elif language == "Bengali":
        handoff = f"আরও ব্যক্তিগত সহায়তা বা কোটেশনের জন্য আমাদের সেলস টিমের সাথে যোগাযোগ করুন: {phone} | {email}"
    else:
        handoff = f"For personalised assistance or a quotation, please contact our sales team: {phone} | {email}"

    if flow_result.conversation_id and is_db_enabled():
        try:
            from src.db import conversations as db_conversations
            db_conversations.set_contact_shared(flow_result.conversation_id, True)
        except Exception:
            logger.exception("Failed to mark sales contact as shared", extra={"conversation_id": flow_result.conversation_id})
    return handoff if not reply.strip() else f"{reply.rstrip()}\n\n{handoff}"


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
        _try_initialize_database()
    except Exception as exc:
        logger.exception("Database initialization failed")
        startup_error = f"{exc.__class__.__name__}: {exc}"

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
    close_db_client()
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
app.include_router(api_router, prefix="/api")

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
        handled_reply = _add_sales_contact_prompt(flow_result.reply, flow_result)
        _record_bot_reply(
            flow_result.conversation_id,
            handled_reply,
            language=flow_result.preferred_language,
        )
        return {
            "session_id": req.session_id,
            "reply": handled_reply,
            "images": flow_result.images or [],
            "options": flow_result.options or [],
            "flow_stage": flow_result.stage,
            "metadata": [],
        }
    
    try:
        started = time.time()
        response = await run_in_threadpool(
            rag.query,
            req.message,
            req.session_id,
            flow_result.preferred_language,
        )
        latency_ms = int((time.time() - started) * 1000)
        
        final_reply = response.get("answer", "")
        if flow_result.reply:
            final_reply = f"{flow_result.reply}\n\n{final_reply}"
        final_reply = _add_sales_contact_prompt(final_reply, flow_result)
            
        final_images = (flow_result.images or []) + response.get("images", [])

        _record_bot_reply(
            flow_result.conversation_id,
            final_reply.strip(),
            language=flow_result.preferred_language,
            tokens=((response.get("usage") or {}).get("total_tokens") if isinstance(response.get("usage"), dict) else None),
            model=(settings.llm_model if settings else None),
            latency_ms=latency_ms,
        )
        _record_bot_images(flow_result.conversation_id, final_images)

        # response should contain 'answer', 'sources', 'images'
        return {
            "session_id": req.session_id,
            "reply": final_reply.strip(),
            "images": final_images,
            "options": [],
            "flow_stage": flow_result.stage,
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
            _record_bot_reply(
                flow_result.conversation_id,
                flow_result.reply,
                language=flow_result.preferred_language,
            )
            _record_bot_images(flow_result.conversation_id, flow_result.images or [])

        if flow_result.handled:
            handled_reply = _add_sales_contact_prompt(flow_result.reply, flow_result)
            await _send_whatsapp_flow_response(
                from_number,
                handled_reply,
                flow_result.options,
                flow_result.images,
            )
            _record_bot_reply(
                flow_result.conversation_id,
                handled_reply,
                language=flow_result.preferred_language,
            )
            continue

        # Step 2: RAG query with the user's stored language preference
        try:
            started = time.time()
            response = await run_in_threadpool(
                rag.query, user_text, session_id, flow_result.preferred_language
            )
            answer = (response.get("answer") or "I could not generate a response right now.").strip()
            answer = _add_sales_contact_prompt(answer, flow_result)
            await run_in_threadpool(wa.send_text, from_number, answer)
            await _send_public_images_whatsapp(from_number, response.get("images", []))
            _record_bot_images(flow_result.conversation_id, response.get("images", []))
            _record_bot_reply(
                flow_result.conversation_id,
                answer,
                language=flow_result.preferred_language,
                tokens=((response.get("usage") or {}).get("total_tokens") if isinstance(response.get("usage"), dict) else None),
                model=(settings.llm_model if settings else None),
                latency_ms=int((time.time() - started) * 1000),
            )
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
                _record_bot_reply(
                    flow_result.conversation_id,
                    flow_result.reply,
                    language=flow_result.preferred_language,
                )
                # Note: Currently not sending images for Messenger early-greeting as it is basic,
                # but could be added similar to WhatsApp.

            if flow_result.handled:
                handled_reply = _add_sales_contact_prompt(flow_result.reply, flow_result)
                if flow_result.options:
                    # Use quick replies for language selection on Messenger
                    await run_in_threadpool(
                        _send_messenger_quick_replies,
                        psid, handled_reply, flow_result.options,
                    )
                elif handled_reply:
                    await run_in_threadpool(_send_messenger_text, psid, handled_reply)
                _record_bot_reply(
                    flow_result.conversation_id,
                    handled_reply,
                    language=flow_result.preferred_language,
                )
                continue

            try:
                started = time.time()
                response = await run_in_threadpool(
                    rag.query, text, session_id, flow_result.preferred_language
                )
                answer = (response.get("answer") or "I could not generate a response.").strip()
                answer = _add_sales_contact_prompt(answer, flow_result)
                await run_in_threadpool(_send_messenger_text, psid, answer)
                _record_bot_reply(
                    flow_result.conversation_id,
                    answer,
                    language=flow_result.preferred_language,
                    tokens=((response.get("usage") or {}).get("total_tokens") if isinstance(response.get("usage"), dict) else None),
                    model=(settings.llm_model if settings else None),
                    latency_ms=int((time.time() - started) * 1000),
                )
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
