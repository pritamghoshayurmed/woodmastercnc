from __future__ import annotations

import time
from typing import Any

from src.db.client import DatabaseDisabledError, get_db_client, is_db_enabled

_CACHE_TTL_SECONDS = 15.0
_cache: dict[str, Any] = {"value": None, "ts": 0.0}

DEFAULTS: dict[str, Any] = {
    "system_prompt": None,
    "hot_lead_threshold": 75,
    "warm_lead_threshold": 40,
    "qualified_threshold": 50,
    "max_chat_turns": 6,
    "inactivity_timeout_seconds": 900,
}

FIELDS = tuple(DEFAULTS.keys())


def _row_to_settings(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return dict(DEFAULTS)
    return {field: row.get(field, DEFAULTS[field]) for field in FIELDS}


def get_settings(*, fresh: bool = False) -> dict[str, Any]:
    """Return the singleton AI settings row, cached briefly to avoid a DB hit per message."""
    if not is_db_enabled():
        return dict(DEFAULTS)

    now = time.monotonic()
    if not fresh and _cache["value"] is not None and now - _cache["ts"] < _CACHE_TTL_SECONDS:
        return _cache["value"]

    try:
        row = get_db_client().fetch_one("SELECT * FROM ai_settings WHERE id;")
    except DatabaseDisabledError:
        return dict(DEFAULTS)

    settings = _row_to_settings(row)
    _cache["value"] = settings
    _cache["ts"] = now
    return settings


def update_settings(fields: dict[str, Any]) -> dict[str, Any]:
    current = get_settings(fresh=True)
    merged = dict(current)
    for key, value in fields.items():
        if key not in FIELDS or value is None:
            continue
        merged[key] = value or None if key == "system_prompt" else value

    row = get_db_client().execute_returning(
        """
        UPDATE ai_settings
        SET
            system_prompt = %s,
            hot_lead_threshold = %s,
            warm_lead_threshold = %s,
            qualified_threshold = %s,
            max_chat_turns = %s,
            inactivity_timeout_seconds = %s,
            updated_at = now()
        WHERE id
        RETURNING *;
        """,
        (
            merged["system_prompt"],
            merged["hot_lead_threshold"],
            merged["warm_lead_threshold"],
            merged["qualified_threshold"],
            merged["max_chat_turns"],
            merged["inactivity_timeout_seconds"],
        ),
    )
    if row is None:
        raise RuntimeError("Failed to update AI settings.")

    settings = _row_to_settings(row)
    _cache["value"] = settings
    _cache["ts"] = time.monotonic()
    return settings
