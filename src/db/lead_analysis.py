from __future__ import annotations

from typing import Any

from src.db.client import get_db_client


def upsert_lead_analysis(conversation_id: str, analysis: dict[str, Any]) -> dict[str, Any]:
    row = get_db_client().execute_returning(
        """
        INSERT INTO lead_analysis (
            conversation_id, intent, urgency, interest_level, product_interest, budget,
            timeline, sentiment, language, lead_score, qualified, confidence,
            recommended_action, matched_rule_ids, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (conversation_id)
        DO UPDATE SET
            intent = EXCLUDED.intent,
            urgency = EXCLUDED.urgency,
            interest_level = EXCLUDED.interest_level,
            product_interest = EXCLUDED.product_interest,
            budget = EXCLUDED.budget,
            timeline = EXCLUDED.timeline,
            sentiment = EXCLUDED.sentiment,
            language = EXCLUDED.language,
            lead_score = EXCLUDED.lead_score,
            qualified = EXCLUDED.qualified,
            confidence = EXCLUDED.confidence,
            recommended_action = EXCLUDED.recommended_action,
            matched_rule_ids = EXCLUDED.matched_rule_ids,
            updated_at = now()
        RETURNING *;
        """,
        (
            conversation_id,
            analysis.get("intent"),
            analysis.get("urgency"),
            analysis.get("interest_level"),
            analysis.get("product_interest"),
            analysis.get("budget"),
            analysis.get("timeline"),
            analysis.get("sentiment"),
            analysis.get("language"),
            analysis.get("lead_score", 0),
            analysis.get("qualified", False),
            analysis.get("confidence"),
            analysis.get("recommended_action"),
            analysis.get("matched_rule_ids", []),
        ),
    )
    if row is None:
        raise RuntimeError("Failed to upsert lead analysis.")
    return row


def get_lead_analysis(conversation_id: str) -> dict[str, Any] | None:
    return get_db_client().fetch_one(
        "SELECT * FROM lead_analysis WHERE conversation_id = %s;",
        (conversation_id,),
    )
