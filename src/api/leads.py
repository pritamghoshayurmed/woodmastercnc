from __future__ import annotations

import csv
import io
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.db import ai_settings, conversation_summary, lead_analysis, users
from src.db.client import DatabaseDisabledError, get_db_client


router = APIRouter()


class LeadUpdateRequest(BaseModel):
    status: str | None = None
    assignedTo: str | None = None
    email: str | None = None
    name: str | None = None
    address: str | None = None
    language: str | None = None
    dashboardState: Literal["normal", "favourite", "archived"] | None = None
    note: str | None = None
    manualOrder: int | None = None


def _lead_thresholds() -> dict[str, int]:
    settings = ai_settings.get_settings()
    return {"hot": int(settings["hot_lead_threshold"]), "warm": int(settings["warm_lead_threshold"])}


def _purchase_probability(score: int, thresholds: dict[str, int] | None = None) -> str:
    thresholds = thresholds or _lead_thresholds()
    if score >= thresholds["hot"]:
        return "High"
    if score >= thresholds["warm"]:
        return "Medium"
    return "Low"


def _lead_row_to_payload(row: dict[str, Any], thresholds: dict[str, int] | None = None) -> dict[str, Any]:
    thresholds = thresholds or _lead_thresholds()
    score = int(
        row["analysis_score"]
        if row.get("analysis_score") is not None
        else (row.get("lead_score") or 0)
    )
    summary = row.get("summary") or ""
    potential = "Hot" if score >= thresholds["hot"] else "Warm" if score >= thresholds["warm"] else "Cold"
    dashboard_state = row.get("dashboard_state") or "normal"
    return {
        "id": row["user_id"],
        "name": row.get("name") or row["phone_number"],
        "phone": row["phone_number"],
        "email": row.get("email") or "",
        "location": row.get("address") or row.get("city") or "",
        "language": row.get("language") or "Unknown",
        "source": row.get("source") or "WEB",
        "score": score,  # Kept for existing API consumers.
        "leadScore": score,
        "potential": potential,
        "status": row.get("status") or "New",
        "lastMessage": row.get("last_message") or "",
        "time": row.get("last_seen_human") or "",
        "firstContact": row.get("first_seen_iso") or "",
        "lastActivity": row.get("last_seen_iso") or "",
        "interestedIn": row.get("product_interest") or "",
        "assignedTo": row.get("assigned_to") or "—",
        "aiSummary": summary,
        "purchaseProbability": _purchase_probability(score, thresholds),
        "currentConversationId": row.get("conversation_id"),
        "conversationId": row.get("conversation_id"),
        "dashboardState": dashboard_state,
        "note": row.get("manager_note") or "",
        "manualOrder": int(row.get("manual_order") or 0),
        "manualHandled": row.get("human_handled_at") is not None,
        "humanHandledAt": row.get("human_handled_at"),
        "unreadCount": 0,
    }


LEADS_BASE_QUERY = """
WITH latest_messages AS (
    SELECT DISTINCT ON (m.conversation_id)
        m.conversation_id,
        m.message,
        m.timestamp
    FROM messages m
    ORDER BY m.conversation_id, m.timestamp DESC
)
SELECT
    u.id AS user_id,
    u.phone_number,
    u.name,
    u.email,
    u.address,
    u.city,
    u.language,
    u.source,
    u.status,
    u.assigned_to,
    u.dashboard_state,
    u.manager_note,
    u.manual_order,
    u.first_seen,
    u.last_seen,
    to_char(u.first_seen, 'YYYY-MM-DD"T"HH24:MI:SS') AS first_seen_iso,
    to_char(u.last_seen, 'YYYY-MM-DD"T"HH24:MI:SS') AS last_seen_iso,
    c.id AS conversation_id,
    c.human_handled_at,
    c.lead_score,
    la.lead_score AS analysis_score,
    la.product_interest,
    cs.summary,
    lm.message AS last_message,
    CONCAT(EXTRACT(EPOCH FROM (now() - u.last_seen))::int, 's ago') AS last_seen_human
FROM users u
LEFT JOIN conversations c
    ON c.id = u.current_conversation_id
LEFT JOIN lead_analysis la
    ON la.conversation_id = c.id
LEFT JOIN conversation_summary cs
    ON cs.conversation_id = c.id
LEFT JOIN latest_messages lm
    ON lm.conversation_id = c.id
"""


@router.get("")
def get_leads(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=1000),
    source: str | None = None,
    min_score: int | None = Query(None, ge=0, le=100),
    max_score: int | None = Query(None, ge=0, le=100),
    status: str | None = None,
    language: str | None = None,
    potential: Literal["Hot", "Warm", "Cold"] | None = None,
    view: Literal["active", "archived", "all"] = "active",
    search: str | None = Query(None, max_length=200),
    sort: Literal["date", "manual", "potential"] = "date",
    manual_handled: bool = False,
) -> dict[str, Any]:
    try:
        db = get_db_client()
    except DatabaseDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    thresholds = _lead_thresholds()
    filters = []
    params: list[Any] = []
    if source:
        filters.append("u.source = %s")
        params.append(source)
    if status:
        filters.append("u.status = %s")
        params.append(status)
    if language:
        filters.append("u.language = %s")
        params.append(language)
    if view == "active":
        filters.append("COALESCE(u.dashboard_state, 'normal') <> 'archived'")
    elif view == "archived":
        filters.append("COALESCE(u.dashboard_state, 'normal') = 'archived'")
    if manual_handled:
        filters.append("c.human_handled_at IS NOT NULL")
    if potential == "Hot":
        filters.append("COALESCE(la.lead_score, c.lead_score, 0) >= %s")
        params.append(thresholds["hot"])
    elif potential == "Warm":
        filters.append("COALESCE(la.lead_score, c.lead_score, 0) BETWEEN %s AND %s")
        params.extend([thresholds["warm"], thresholds["hot"] - 1])
    elif potential == "Cold":
        filters.append("COALESCE(la.lead_score, c.lead_score, 0) < %s")
        params.append(thresholds["warm"])
    if search:
        filters.append("(COALESCE(u.name, '') ILIKE %s OR u.phone_number ILIKE %s OR COALESCE(u.email, '') ILIKE %s)")
        term = f"%{search.strip()}%"
        params.extend([term, term, term])
    if min_score is not None:
        filters.append("COALESCE(la.lead_score, c.lead_score, 0) >= %s")
        params.append(min_score)
    if max_score is not None:
        filters.append("COALESCE(la.lead_score, c.lead_score, 0) <= %s")
        params.append(max_score)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    count_row = db.fetch_one(
        f"SELECT COUNT(*)::int AS count FROM ({LEADS_BASE_QUERY} {where_clause}) base;",
        tuple(params),
    ) or {"count": 0}

    order_by = {
        "date": "u.last_seen DESC",
        "manual": "u.manual_order ASC, u.last_seen DESC",
        "potential": "COALESCE(la.lead_score, c.lead_score, 0) DESC, u.last_seen DESC",
    }[sort]
    params.extend([page_size, (page - 1) * page_size])
    rows = db.fetch_all(
        f"""
        {LEADS_BASE_QUERY}
        {where_clause}
        ORDER BY {order_by}
        LIMIT %s OFFSET %s;
        """,
        tuple(params),
    )
    return {
        "leads": [_lead_row_to_payload(row, thresholds) for row in rows],
        "pagination": {
            "page": page,
            "pageSize": page_size,
            "total": count_row["count"],
        },
    }


@router.get("/export")
def export_leads() -> StreamingResponse:
    payload = get_leads(page=1, page_size=1000)
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "id", "name", "phone", "email", "location", "language", "source",
            "score", "status", "interestedIn", "assignedTo", "aiSummary",
        ],
    )
    writer.writeheader()
    for lead in payload["leads"]:
        writer.writerow(lead)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="woodmaster_leads.csv"'},
    )


@router.get("/{user_id}")
def get_lead(user_id: str) -> dict[str, Any]:
    try:
        db = get_db_client()
    except DatabaseDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rows = db.fetch_all(
        f"""
        {LEADS_BASE_QUERY}
        WHERE u.id = %s
        ORDER BY u.last_seen DESC;
        """,
        (user_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Lead not found.")

    row = rows[0]
    user_payload = _lead_row_to_payload(row)
    conversation_rows = db.fetch_all(
        """
        SELECT id, started_at, ended_at, status, lead_score, source
        FROM conversations
        WHERE user_id = %s
        ORDER BY started_at DESC;
        """,
        (user_id,),
    )
    user_payload["conversations"] = conversation_rows
    return user_payload


@router.delete("/{user_id}")
def delete_lead(user_id: str) -> dict[str, Any]:
    """Permanently delete a lead and every conversation/message tied to it (cascading FKs)."""
    try:
        get_db_client()
    except DatabaseDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    deleted = users.delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Lead not found.")
    return {"deleted": True}


@router.put("/{user_id}")
def update_lead(user_id: str, body: LeadUpdateRequest) -> dict[str, Any]:
    user_row = users.get_user_by_id(user_id)
    if user_row is None:
        raise HTTPException(status_code=404, detail="Lead not found.")

    updated = users.update_user_state(
        user_id,
        stage=user_row.get("conversation_state") or "FAQ",
        language=body.language or user_row.get("language"),
        name=body.name or user_row.get("name"),
        address=body.address or user_row.get("address"),
        city=user_row.get("city"),
        state_name=user_row.get("state"),
        country=user_row.get("country"),
        current_conversation_id=user_row.get("current_conversation_id"),
        status=body.status or user_row.get("status"),
        email=body.email or user_row.get("email"),
        assigned_to=body.assignedTo or user_row.get("assigned_to"),
        touch_last_seen=False,
    )

    if body.dashboardState is not None or body.note is not None or body.manualOrder is not None:
        updated = get_db_client().execute_returning(
            """
            UPDATE users
            SET dashboard_state = COALESCE(%s, dashboard_state),
                manager_note = COALESCE(%s, manager_note),
                manual_order = COALESCE(%s, manual_order),
                updated_at = now()
            WHERE id = %s
            RETURNING *;
            """,
            (body.dashboardState, body.note, body.manualOrder, user_id),
        ) or updated

    current_conversation_id = updated.get("current_conversation_id")
    summary_row = conversation_summary.get_summary(current_conversation_id) if current_conversation_id else None
    analysis_row = lead_analysis.get_lead_analysis(current_conversation_id) if current_conversation_id else None
    return {
        "id": updated["id"],
        "status": updated["status"],
        "assignedTo": updated.get("assigned_to") or "—",
        "email": updated.get("email") or "",
        "name": updated.get("name") or updated["phone_number"],
        "address": updated.get("address") or "",
        "language": updated.get("language") or "Unknown",
        "aiSummary": (summary_row or {}).get("summary") or "",
        "score": (analysis_row or {}).get("lead_score") or 0,
        "dashboardState": updated.get("dashboard_state") or "normal",
        "note": updated.get("manager_note") or "",
        "manualOrder": int(updated.get("manual_order") or 0),
    }
