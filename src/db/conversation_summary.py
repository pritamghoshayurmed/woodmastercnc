from __future__ import annotations

from typing import Any

from psycopg2.extras import Json

from src.db.client import get_db_client


def upsert_summary(
    conversation_id: str,
    summary_text: str,
    key_points: list[str],
    requirements: dict[str, Any],
    follow_up_needed: bool,
) -> dict[str, Any]:
    row = get_db_client().execute_returning(
        """
        INSERT INTO conversation_summary (
            conversation_id, summary, key_points, customer_requirements, follow_up_needed, generated_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, now(), now())
        ON CONFLICT (conversation_id)
        DO UPDATE SET
            summary = EXCLUDED.summary,
            key_points = EXCLUDED.key_points,
            customer_requirements = EXCLUDED.customer_requirements,
            follow_up_needed = EXCLUDED.follow_up_needed,
            updated_at = now()
        RETURNING *;
        """,
        (conversation_id, summary_text, Json(key_points), Json(requirements), follow_up_needed),
    )
    if row is None:
        raise RuntimeError("Failed to upsert conversation summary.")
    return row


def get_summary(conversation_id: str) -> dict[str, Any] | None:
    return get_db_client().fetch_one(
        "SELECT * FROM conversation_summary WHERE conversation_id = %s;",
        (conversation_id,),
    )
