from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.db.client import get_db_client

logger = logging.getLogger(__name__)


def get_all_entries() -> list[dict[str, Any]]:
    return get_db_client().fetch_all(
        "SELECT * FROM qna_entries ORDER BY priority ASC, created_at ASC;"
    )


def get_enabled_entries() -> list[dict[str, Any]]:
    return get_db_client().fetch_all(
        """
        SELECT * FROM qna_entries
        WHERE enabled = true
        ORDER BY priority ASC, created_at ASC;
        """
    )


def count_entries() -> int:
    row = get_db_client().fetch_one("SELECT COUNT(*)::int AS count FROM qna_entries;")
    return int((row or {}).get("count") or 0)


def create_entry(question: str, answer: str, priority: int, enabled: bool = True) -> dict[str, Any]:
    row = get_db_client().execute_returning(
        """
        INSERT INTO qna_entries (question, answer, priority, enabled)
        VALUES (%s, %s, %s, %s)
        RETURNING *;
        """,
        (question, answer, priority, enabled),
    )
    if row is None:
        raise RuntimeError("Failed to create QnA entry.")
    return row


def update_entry(entry_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
    current = get_db_client().fetch_one(
        "SELECT * FROM qna_entries WHERE id = %s;",
        (entry_id,),
    )
    if current is None:
        return None

    def pick(key: str) -> Any:
        # dict.get()'s default only applies when the key is absent, not when the
        # caller explicitly sent None for "field not changed" (e.g. a partial
        # PUT) -- an explicit is-not-None check is required to preserve it.
        value = fields.get(key)
        return value if value is not None else current[key]

    return get_db_client().execute_returning(
        """
        UPDATE qna_entries
        SET
            question = %s,
            answer = %s,
            enabled = %s,
            priority = %s,
            updated_at = now()
        WHERE id = %s
        RETURNING *;
        """,
        (
            pick("question"),
            pick("answer"),
            pick("enabled"),
            pick("priority"),
            entry_id,
        ),
    )


def delete_entry(entry_id: str) -> bool:
    row = get_db_client().execute_returning(
        "DELETE FROM qna_entries WHERE id = %s RETURNING id;",
        (entry_id,),
    )
    return row is not None


def reorder_entries(ordered_ids: list[str]) -> None:
    db = get_db_client()
    with db.connection() as conn:
        with conn.cursor() as cur:
            for priority, entry_id in enumerate(ordered_ids, start=1):
                cur.execute(
                    """
                    UPDATE qna_entries
                    SET priority = %s, updated_at = now()
                    WHERE id = %s;
                    """,
                    (priority, entry_id),
                )


def seed_from_markdown_if_empty(knowledge_path: Path) -> int:
    """Populate qna_entries from data/knowledge.md the first time the table is empty.

    Keeps the dashboard pre-populated with the existing FAQ content instead of
    starting blank, without ever overwriting rows an admin has since edited.
    """
    if count_entries() > 0 or not knowledge_path.exists():
        return 0

    from src.pipeline.rag_pipeline import RAGPipeline

    text = knowledge_path.read_text(encoding="utf-8")
    try:
        entries = RAGPipeline._parse_faq_entries(text)
    except Exception:
        logger.exception("Failed to parse %s while seeding qna_entries", knowledge_path)
        return 0

    seeded = 0
    for priority, entry in enumerate(entries, start=1):
        if entry.get("is_placeholder"):
            continue
        create_entry(entry["question"], entry["answer"], priority)
        seeded += 1
    return seeded
