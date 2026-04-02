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
from src.pipeline.rag_pipepline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG pipeline instance
rag: RAGPipeline | None = None
settings = None
startup_error: str | None = None

# Basic in-memory rate limiter (per session_id)
MAX_REQUESTS_PER_MINUTE = 20
RATE_WINDOW_SECONDS = 60
_rate_limits: dict[str, list[float]] = {}
conversation_flow = ConversationFlowManager()

# WhatsApp Cloud API env vars
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip()
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
WHATSAPP_GRAPH_VERSION = os.getenv("WHATSAPP_GRAPH_VERSION", "v23.0").strip()
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "").strip()


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


def _is_whatsapp_configured() -> bool:
    return bool(WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_ACCESS_TOKEN)


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


def _send_whatsapp_text(to_number: str, text: str) -> None:
    if not _is_whatsapp_configured():
        logger.error("WhatsApp env vars are missing. Cannot send message.")
        return

    url = f"https://graph.facebook.com/{WHATSAPP_GRAPH_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": text[:4096]},
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    if resp.status_code >= 300:
        logger.error("WhatsApp send failed: status=%s body=%s", resp.status_code, resp.text)

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
    allow_credentials=True,
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
        
        # response should contain 'answer', 'sources', 'images'
        return {
            "session_id": req.session_id,
            "reply": response.get("answer", ""),
            "images": response.get("images", []),
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

        user_text = _extract_text_from_message(message)
        if not user_text:
            continue

        session_id = f"whatsapp:{from_number}"
        if not _allow_request(session_id):
            _send_whatsapp_text(from_number, "Too many requests. Please wait a minute and try again.")
            continue

        try:
            response = await run_in_threadpool(rag.query, user_text, session_id)
            answer = (response.get("answer") or "I could not generate a response right now.").strip()
            await run_in_threadpool(_send_whatsapp_text, from_number, answer)
        except Exception:
            logger.exception("WhatsApp processing failed", extra={"session_id": session_id})
            await run_in_threadpool(
                _send_whatsapp_text,
                from_number,
                "Sorry, I hit an internal error. Please try again.",
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
