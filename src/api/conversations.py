from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

import src.whatsapp.client as wa
from src.db import conversation_summary, conversations, lead_analysis, messages, users
from src.db.client import DatabaseDisabledError, get_db_client
from src.lead.scorer import score_conversation


router = APIRouter()

_UPLOAD_DIR = Path("data") / "images" / "uploads"
_ALLOWED_IMAGE_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
_MAX_IMAGE_BYTES = 5 * 1024 * 1024


class CloseConversationRequest(BaseModel):
    reason: str = "manual_close"


class AppendMessageRequest(BaseModel):
    sender: str
    text: str
    language: str | None = None


@router.get("")
def get_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    status: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    try:
        db = get_db_client()
    except DatabaseDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    filters = []
    params: list[Any] = []
    if status:
        filters.append("c.status = %s")
        params.append(status)
    if source:
        filters.append("c.source = %s")
        params.append(source)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    count_row = db.fetch_one(
        f"SELECT COUNT(*)::int AS count FROM conversations c {where_clause};",
        tuple(params),
    ) or {"count": 0}

    params.extend([page_size, (page - 1) * page_size])
    rows = db.fetch_all(
        f"""
        SELECT
            c.*,
            u.name,
            u.phone_number,
            u.language
        FROM conversations c
        JOIN users u ON u.id = c.user_id
        {where_clause}
        ORDER BY c.started_at DESC
        LIMIT %s OFFSET %s;
        """,
        tuple(params),
    )
    return {
        "conversations": rows,
        "pagination": {"page": page, "pageSize": page_size, "total": count_row["count"]},
    }


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str) -> dict[str, Any]:
    conversation_row = conversations.get_conversation_by_id(conversation_id)
    if conversation_row is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return {
        "conversation": conversation_row,
        "messages": messages.get_conversation_messages(conversation_id, limit=500),
        "leadAnalysis": lead_analysis.get_lead_analysis(conversation_id),
        "summary": conversation_summary.get_summary(conversation_id),
    }


@router.post("/{conversation_id}/close")
def close_conversation(conversation_id: str, body: CloseConversationRequest) -> dict[str, Any]:
    row = conversations.close_conversation(conversation_id, body.reason)
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    analysis = score_conversation(conversation_id)
    return {"conversation": row, "leadAnalysis": analysis}


@router.post("/{conversation_id}/messages")
def append_conversation_message(conversation_id: str, body: AppendMessageRequest) -> dict[str, Any]:
    conversation_row = conversations.get_conversation_by_id(conversation_id)
    if conversation_row is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    db_message_id = messages.append_message(
        conversation_id,
        body.sender.upper(),
        body.text,
        language=body.language,
    )
    if body.sender.upper() == "AGENT":
        conversations.mark_human_handled(conversation_id)
        if conversation_row.get("source") == "WHATSAPP":
            user_row = users.get_user_by_id(conversation_row["user_id"])
            if user_row and user_row.get("phone_number"):
                to_number = users.to_whatsapp_number(user_row["phone_number"])
                wa.send_text(to_number, body.text)
    users.update_user_last_seen(conversation_row["user_id"])
    analysis = score_conversation(conversation_id)
    updated_messages = messages.get_conversation_messages(conversation_id, limit=500)
    return {
        "messageId": db_message_id,
        "messages": updated_messages,
        "leadAnalysis": analysis,
    }


@router.post("/{conversation_id}/messages/image")
async def append_conversation_image(conversation_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    """Let a human agent attach and send a product/photo image manually, mirroring
    how the AI sends catalog images: save it publicly, send via WhatsApp, and log
    it as a message so the dashboard thread shows what was actually sent."""
    conversation_row = conversations.get_conversation_by_id(conversation_id)
    if conversation_row is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    extension = _ALLOWED_IMAGE_TYPES.get(file.content_type or "")
    if extension is None:
        raise HTTPException(status_code=400, detail="Only PNG, JPEG, or WEBP images are supported.")

    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(body) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image must be 5MB or smaller.")

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    relative_path = f"data/images/uploads/{uuid.uuid4().hex}{extension}"
    Path(relative_path).write_bytes(body)

    public_base_url = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    image_url = f"{public_base_url}/{relative_path}" if public_base_url else relative_path

    db_message_id = messages.append_message(
        conversation_id, "AGENT", image_url, message_type="image",
    )
    conversations.mark_human_handled(conversation_id)
    if conversation_row.get("source") == "WHATSAPP" and public_base_url:
        user_row = users.get_user_by_id(conversation_row["user_id"])
        if user_row and user_row.get("phone_number"):
            to_number = users.to_whatsapp_number(user_row["phone_number"])
            wa.send_image(to_number, image_url)
    users.update_user_last_seen(conversation_row["user_id"])
    analysis = score_conversation(conversation_id)
    updated_messages = messages.get_conversation_messages(conversation_id, limit=500)
    return {
        "messageId": db_message_id,
        "messages": updated_messages,
        "leadAnalysis": analysis,
    }


@router.post("/{conversation_id}/score")
def rescore_conversation(conversation_id: str) -> dict[str, Any]:
    if conversations.get_conversation_by_id(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"leadAnalysis": score_conversation(conversation_id)}


@router.post("/{conversation_id}/handle")
def handle_conversation(conversation_id: str) -> dict[str, Any]:
    """Move a conversation from the AI bucket into the manually-handled (Leads) bucket."""
    if conversations.get_conversation_by_id(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    conversations.mark_human_handled(conversation_id)
    return {"conversation": conversations.get_conversation_by_id(conversation_id)}


@router.post("/{conversation_id}/unhandle")
def unhandle_conversation(conversation_id: str) -> dict[str, Any]:
    """Move a conversation back into the AI bucket (undo of /handle)."""
    row = conversations.clear_human_handled(conversation_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"conversation": row}


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict[str, Any]:
    deleted = conversations.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"deleted": True}
