from __future__ import annotations

from typing import Any

from psycopg2.extras import Json

from src.db.client import get_db_client


def log_event(conversation_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
    get_db_client().execute(
        """
        INSERT INTO events (conversation_id, event_type, event_data)
        VALUES (%s, %s, %s);
        """,
        (conversation_id, event_type, Json(data or {})),
    )
