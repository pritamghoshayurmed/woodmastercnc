from __future__ import annotations

from fastapi import APIRouter, HTTPException
from src.db.client import DatabaseDisabledError, get_db_client


router = APIRouter()


@router.get("/overview")
def analytics_overview() -> dict:
    try:
        db = get_db_client()
    except DatabaseDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    total_leads = db.fetch_one("SELECT COUNT(*)::int AS count FROM users;")["count"]
    qualified = db.fetch_one("SELECT COUNT(*)::int AS count FROM users WHERE status = 'Qualified';")["count"]
    high_intent = db.fetch_one(
        """
        SELECT COUNT(*)::int AS count
        FROM lead_analysis
        WHERE interest_level = 'high' OR urgency = 'high';
        """
    )["count"]
    conversation_count = db.fetch_one("SELECT COUNT(*)::int AS count FROM conversations;")["count"]
    average_score = db.fetch_one("SELECT COALESCE(ROUND(AVG(lead_score)), 0)::int AS avg FROM conversations;")["avg"]
    return {
        "totalLeads": total_leads,
        "qualifiedLeads": qualified,
        "highIntentLeads": high_intent,
        "conversations": conversation_count,
        "averageScore": average_score,
    }


@router.get("/by-source")
def analytics_by_source() -> dict:
    rows = get_db_client().fetch_all(
        """
        SELECT source, COUNT(*)::int AS count
        FROM users
        GROUP BY source
        ORDER BY count DESC;
        """
    )
    total = sum(row["count"] for row in rows) or 1
    return {
        "sources": [
            {
                "name": row["source"],
                "count": row["count"],
                "percentage": round((row["count"] / total) * 100, 1),
            }
            for row in rows
        ]
    }


@router.get("/score-dist")
def analytics_score_distribution() -> dict:
    rows = get_db_client().fetch_one(
        """
        SELECT
            COUNT(*) FILTER (WHERE lead_score BETWEEN 0 AND 25)::int AS bucket_0_25,
            COUNT(*) FILTER (WHERE lead_score BETWEEN 26 AND 50)::int AS bucket_26_50,
            COUNT(*) FILTER (WHERE lead_score BETWEEN 51 AND 75)::int AS bucket_51_75,
            COUNT(*) FILTER (WHERE lead_score BETWEEN 76 AND 100)::int AS bucket_76_100
        FROM conversations;
        """
    ) or {}
    return {
        "distribution": [
            {"range": "0-25", "count": rows.get("bucket_0_25", 0)},
            {"range": "26-50", "count": rows.get("bucket_26_50", 0)},
            {"range": "51-75", "count": rows.get("bucket_51_75", 0)},
            {"range": "76-100", "count": rows.get("bucket_76_100", 0)},
        ]
    }


@router.get("/trend")
def analytics_trend() -> dict:
    rows = get_db_client().fetch_all(
        """
        SELECT to_char(date_trunc('day', first_seen), 'YYYY-MM-DD') AS day, COUNT(*)::int AS count
        FROM users
        GROUP BY date_trunc('day', first_seen)
        ORDER BY date_trunc('day', first_seen) ASC;
        """
    )
    return {"trend": rows}
