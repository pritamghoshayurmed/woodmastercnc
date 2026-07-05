from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.db import conversation_summary, conversations, lead_analysis, messages
from src.db.client import DatabaseDisabledError, get_db_client
from src.lead.scorer import score_conversation


router = APIRouter()


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
    analysis = score_conversation(conversation_id)
    updated_messages = messages.get_conversation_messages(conversation_id, limit=500)
    return {
        "messageId": db_message_id,
        "messages": updated_messages,
        "leadAnalysis": analysis,
    }
