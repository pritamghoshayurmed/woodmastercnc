from __future__ import annotations

from typing import Any

from src.db.client import get_db_client


def get_all_rules() -> list[dict[str, Any]]:
    return get_db_client().fetch_all(
        "SELECT * FROM lead_score_rules ORDER BY priority ASC, created_at ASC;"
    )


def get_enabled_rules() -> list[dict[str, Any]]:
    return get_db_client().fetch_all(
        """
        SELECT * FROM lead_score_rules
        WHERE enabled = true
        ORDER BY priority ASC, created_at ASC;
        """
    )


def create_rule(name: str, description: str, weight: int, priority: int, matching_type: str) -> dict[str, Any]:
    row = get_db_client().execute_returning(
        """
        INSERT INTO lead_score_rules (rule_name, description, weight, priority, matching_type)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING *;
        """,
        (name, description, weight, priority, matching_type),
    )
    if row is None:
        raise RuntimeError("Failed to create scoring rule.")
    return row


def update_rule(rule_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
    current = get_db_client().fetch_one(
        "SELECT * FROM lead_score_rules WHERE id = %s;",
        (rule_id,),
    )
    if current is None:
        return None

    def pick(key: str) -> Any:
        # dict.get()'s default only applies when the key is absent, not when the
        # caller explicitly sent None for "field not changed" (e.g. a partial
        # PUT) -- an explicit is-not-None check is required to preserve it.
        value = fields.get(key)
        return value if value is not None else current[key]

    row = get_db_client().execute_returning(
        """
        UPDATE lead_score_rules
        SET
            rule_name = %s,
            description = %s,
            weight = %s,
            enabled = %s,
            priority = %s,
            matching_type = %s,
            updated_at = now()
        WHERE id = %s
        RETURNING *;
        """,
        (
            pick("rule_name"),
            pick("description"),
            pick("weight"),
            pick("enabled"),
            pick("priority"),
            pick("matching_type"),
            rule_id,
        ),
    )
    return row


def delete_rule(rule_id: str) -> bool:
    row = get_db_client().execute_returning(
        "DELETE FROM lead_score_rules WHERE id = %s RETURNING id;",
        (rule_id,),
    )
    return row is not None


def reorder_rules(ordered_ids: list[str]) -> None:
    db = get_db_client()
    with db.connection() as conn:
        with conn.cursor() as cur:
            for priority, rule_id in enumerate(ordered_ids, start=1):
                cur.execute(
                    """
                    UPDATE lead_score_rules
                    SET priority = %s, updated_at = now()
                    WHERE id = %s;
                    """,
                    (priority, rule_id),
                )
