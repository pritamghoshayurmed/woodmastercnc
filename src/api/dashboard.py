from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.leads import LEADS_BASE_QUERY, _lead_row_to_payload, _lead_thresholds
from src.db.client import DatabaseDisabledError, get_db_client


router = APIRouter()


@router.get("/overview")
def dashboard_overview() -> dict[str, Any]:
    """Return the small, live data set needed by the operational dashboard."""
    try:
        db = get_db_client()
    except DatabaseDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    thresholds = _lead_thresholds()
    metrics = db.fetch_one(
        """
        SELECT
            COUNT(*) FILTER (WHERE COALESCE(dashboard_state, 'normal') <> 'archived')::int AS total_leads,
            COUNT(*) FILTER (
                WHERE COALESCE(dashboard_state, 'normal') <> 'archived'
                  AND COALESCE(la.lead_score, c.lead_score, 0) >= %s
            )::int AS hot_leads,
            COALESCE(ROUND(AVG(COALESCE(la.lead_score, c.lead_score, 0)) FILTER (
                WHERE COALESCE(dashboard_state, 'normal') <> 'archived'
            )), 0)::int AS average_score
        FROM users u
        LEFT JOIN conversations c ON c.id = u.current_conversation_id
        LEFT JOIN lead_analysis la ON la.conversation_id = c.id;
        """,
        (thresholds["hot"],),
    ) or {}
    conversations_today = db.fetch_one(
        """
        SELECT COUNT(*)::int AS count
        FROM conversations
        WHERE started_at >= date_trunc('day', now());
        """
    ) or {"count": 0}

    start = date.today() - timedelta(days=6)
    trend_rows = db.fetch_all(
        """
        SELECT to_char(day, 'Dy') AS label, to_char(day, 'YYYY-MM-DD') AS day,
               COALESCE(COUNT(u.id), 0)::int AS count
        FROM generate_series(current_date - interval '6 days', current_date, interval '1 day') AS series(day)
        LEFT JOIN users u ON u.first_seen >= day AND u.first_seen < day + interval '1 day'
        GROUP BY day
        ORDER BY day;
        """
    )
    top_rows = db.fetch_all(
        f"""
        {LEADS_BASE_QUERY}
        WHERE COALESCE(u.dashboard_state, 'normal') <> 'archived'
        ORDER BY COALESCE(la.lead_score, c.lead_score, 0) DESC, u.last_seen DESC
        LIMIT 5;
        """
    )
    return {
        "totalLeads": metrics.get("total_leads", 0),
        "hotLeads": metrics.get("hot_leads", 0),
        "avgLeadScore": metrics.get("average_score", 0),
        "conversationsToday": conversations_today["count"],
        "newLeadsThisWeek": [
            {"date": row["label"].strip(), "day": row["day"], "count": row["count"]}
            for row in trend_rows
        ],
        "topPotentialLeads": [_lead_row_to_payload(row, thresholds) for row in top_rows],
        "weekStart": start.isoformat(),
    }
