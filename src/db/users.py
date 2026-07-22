from __future__ import annotations

import re
from typing import Any

from src.db.client import get_db_client

_PHONE_CHANNEL_PREFIXES = ("whatsapp:", "messenger:")
_DEFAULT_COUNTRY_CODE = "91"


def normalize_phone_number(raw: str) -> str:
    """Derive the canonical local-format number stored for a phone-based channel.

    WhatsApp/Messenger session ids carry the full number with country code
    (e.g. "whatsapp:916295716352"); this strips the channel prefix and the
    leading country code so numbers are stored/looked-up in local format
    (e.g. "6295716352"). Non-phone channel ids (e.g. "web:...") pass through
    unchanged since they aren't real phone numbers.
    """
    for prefix in _PHONE_CHANNEL_PREFIXES:
        if raw.startswith(prefix):
            digits = re.sub(r"\D", "", raw[len(prefix):])
            if len(digits) > 10 and digits.startswith(_DEFAULT_COUNTRY_CODE):
                digits = digits[len(_DEFAULT_COUNTRY_CODE):]
            return digits
    return raw


def to_whatsapp_number(phone_number: str) -> str:
    """Reconstruct the full country-coded number the WhatsApp API expects for sending."""
    digits = re.sub(r"\D", "", phone_number)
    if len(digits) == 10:
        return f"{_DEFAULT_COUNTRY_CODE}{digits}"
    return digits


def upsert_user(phone_number: str, source: str) -> dict[str, Any]:
    phone_number = normalize_phone_number(phone_number)
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
    phone_number = normalize_phone_number(phone_number)
    return get_db_client().fetch_one(
        "SELECT * FROM users WHERE phone_number = %s;",
        (phone_number,),
    )


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    return get_db_client().fetch_one("SELECT * FROM users WHERE id = %s;", (user_id,))


def delete_user(user_id: str) -> bool:
    """Permanently delete a lead and every conversation/message/analysis tied to it (cascading FKs)."""
    row = get_db_client().execute_returning(
        "DELETE FROM users WHERE id = %s RETURNING id;",
        (user_id,),
    )
    return row is not None


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
