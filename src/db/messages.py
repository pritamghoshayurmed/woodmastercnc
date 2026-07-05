from __future__ import annotations

from typing import Any

from src.db.client import get_db_client


def append_message(
    conversation_id: str,
    sender: str,
    text: str,
    language: str | None = None,
    tokens: int | None = None,
    model: str | None = None,
    latency_ms: int | None = None,
    message_type: str = "text",
) -> int:
    row = get_db_client().execute_returning(
        """
        INSERT INTO messages (
            conversation_id, sender, message_type, message, language, tokens, ai_model, latency_ms
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (conversation_id, sender, message_type, text, language, tokens, model, latency_ms),
    )
    if row is None:
        raise RuntimeError("Failed to append message.")
    return int(row["id"])


def get_conversation_messages(conversation_id: str, limit: int = 200) -> list[dict[str, Any]]:
    return get_db_client().fetch_all(
        """
        SELECT *
        FROM messages
        WHERE conversation_id = %s
        ORDER BY timestamp ASC
        LIMIT %s;
        """,
        (conversation_id, limit),
    )


def get_message_count(conversation_id: str, sender: str | None = None) -> int:
    if sender:
        row = get_db_client().fetch_one(
            """
            SELECT COUNT(*)::int AS count
            FROM messages
            WHERE conversation_id = %s AND sender = %s;
            """,
            (conversation_id, sender),
        )
    else:
        row = get_db_client().fetch_one(
            """
            SELECT COUNT(*)::int AS count
            FROM messages
            WHERE conversation_id = %s;
            """,
            (conversation_id,),
        )
    return int((row or {}).get("count", 0))
