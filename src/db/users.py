from __future__ import annotations

from typing import Any

from src.db.client import get_db_client


def upsert_user(phone_number: str, source: str) -> dict[str, Any]:
    db = get_db_client()
    row = db.execute_returning(
        """
        INSERT INTO users (phone_number, source, first_seen, last_seen)
        VALUES (%s, %s, now(), now())
        ON CONFLICT (phone_number)
        DO UPDATE SET
            source = EXCLUDED.source,
            last_seen = now(),
            updated_at = now()
        RETURNING *;
        """,
        (phone_number, source),
    )
    if row is None:
        raise RuntimeError("Failed to upsert user.")
    return row


def get_user_by_phone(phone_number: str) -> dict[str, Any] | None:
    return get_db_client().fetch_one(
        "SELECT * FROM users WHERE phone_number = %s;",
        (phone_number,),
    )


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    return get_db_client().fetch_one("SELECT * FROM users WHERE id = %s;", (user_id,))


def update_user_state(
    user_id: str,
    *,
    stage: str,
    language: str | None,
    name: str | None,
    address: str | None,
    city: str | None = None,
    state_name: str | None = None,
    country: str | None = None,
    current_conversation_id: str | None = None,
    status: str | None = None,
    email: str | None = None,
    assigned_to: str | None = None,
    touch_last_seen: bool = True,
) -> dict[str, Any]:
    row = get_db_client().execute_returning(
        """
        UPDATE users
        SET
            conversation_state = %s,
            language = COALESCE(%s, language),
            name = COALESCE(%s, name),
            address = COALESCE(%s, address),
            city = COALESCE(%s, city),
            state = COALESCE(%s, state),
            country = COALESCE(%s, country),
            current_conversation_id = COALESCE(%s, current_conversation_id),
            status = COALESCE(%s, status),
            email = COALESCE(%s, email),
            assigned_to = COALESCE(%s, assigned_to),
            last_seen = CASE WHEN %s THEN now() ELSE last_seen END,
            updated_at = now()
        WHERE id = %s
        RETURNING *;
        """,
        (
            stage,
            language,
            name,
            address,
            city,
            state_name,
            country,
            current_conversation_id,
            status,
            email,
            assigned_to,
            touch_last_seen,
            user_id,
        ),
    )
    if row is None:
        raise RuntimeError(f"User {user_id} not found.")
    return row


def update_user_last_seen(user_id: str) -> None:
    get_db_client().execute(
        "UPDATE users SET last_seen = now(), updated_at = now() WHERE id = %s;",
        (user_id,),
    )
