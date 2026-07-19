from __future__ import annotations

from typing import Any

from src.db.client import get_db_client


def create_conversation(user_id: str, source: str) -> dict[str, Any]:
    db = get_db_client()
    row = db.execute_returning(
        """
        INSERT INTO conversations (user_id, source)
        VALUES (%s, %s)
        RETURNING *;
        """,
        (user_id, source),
    )
    if row is None:
        raise RuntimeError("Failed to create conversation.")
    db.execute(
        """
        UPDATE users
        SET current_conversation_id = %s, last_seen = now(), updated_at = now()
        WHERE id = %s;
        """,
        (row["id"], user_id),
    )
    return row


def get_active_conversation(user_id: str) -> dict[str, Any] | None:
    return get_db_client().fetch_one(
        """
        SELECT * FROM conversations
        WHERE user_id = %s AND status = 'ACTIVE'
        ORDER BY started_at DESC
        LIMIT 1;
        """,
        (user_id,),
    )


def get_conversation_by_id(conversation_id: str) -> dict[str, Any] | None:
    return get_db_client().fetch_one(
        "SELECT * FROM conversations WHERE id = %s;",
        (conversation_id,),
    )


def close_conversation(conversation_id: str, reason: str) -> dict[str, Any] | None:
    return get_db_client().execute_returning(
        """
        UPDATE conversations
        SET status = 'CLOSED', ended_at = now(), closed_reason = %s, updated_at = now()
        WHERE id = %s
        RETURNING *;
        """,
        (reason, conversation_id),
    )


def update_conversation_lead_score(conversation_id: str, score: int) -> None:
    get_db_client().execute(
        """
        UPDATE conversations
        SET lead_score = %s, updated_at = now()
        WHERE id = %s;
        """,
        (score, conversation_id),
    )


def set_contact_shared(conversation_id: str, contact_shared: bool) -> None:
    get_db_client().execute(
        """
        UPDATE conversations
        SET contact_shared = %s, updated_at = now()
        WHERE id = %s;
        """,
        (contact_shared, conversation_id),
    )


def mark_human_handled(conversation_id: str) -> None:
    """Record the first manual owner/agent reply without changing the chat status."""
    get_db_client().execute(
        """
        UPDATE conversations
        SET human_handled_at = COALESCE(human_handled_at, now()), updated_at = now()
        WHERE id = %s;
        """,
        (conversation_id,),
    )
