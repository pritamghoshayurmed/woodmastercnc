from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from typing import Any

from src.db import conversation_summary, conversations, events, lead_analysis, lead_score_rules, messages, users
from src.db.client import is_db_enabled


logger = logging.getLogger(__name__)

PRICE_PATTERN = re.compile(r"\b(price|cost|quotation|quote|budget|dam|rate)\b", re.IGNORECASE)
FEATURE_PATTERN = re.compile(r"\b(feature|spec|specification|power|spindle|table|axis|motor)\b", re.IGNORECASE)
TIMELINE_PATTERN = re.compile(r"\b(today|tomorrow|week|month|urgent|asap|immediately|timeline)\b", re.IGNORECASE)
BROCHURE_PATTERN = re.compile(r"\b(brochure|catalog|catalogue|pdf|details sheet)\b", re.IGNORECASE)
LOCATION_PATTERN = re.compile(r"\b(delhi|mumbai|kolkata|bangalore|hyderabad|chennai|india|city|location)\b", re.IGNORECASE)
PRODUCT_PATTERN = re.compile(r"\b(1325|1530|2040|6090|router|engraver|laser|plasma|machine)\b", re.IGNORECASE)
PURCHASE_PATTERN = re.compile(r"\b(buy|purchase|order|dealer|distributor|price|quotation)\b", re.IGNORECASE)
HIGH_INTENT_PATTERN = re.compile(r"\b(payment|advance|visit|demo|quotation|invoice|delivery)\b", re.IGNORECASE)
NEGATIVE_PATTERN = re.compile(r"\b(not interested|too expensive|expensive|later|not now)\b", re.IGNORECASE)


def _lead_thresholds() -> dict[str, int]:
    """Dashboard-configurable hot/warm/qualified score thresholds, with env fallbacks."""
    try:
        if is_db_enabled():
            from src.db import ai_settings

            settings = ai_settings.get_settings()
            return {
                "hot": int(settings["hot_lead_threshold"]),
                "warm": int(settings["warm_lead_threshold"]),
                "qualified": int(settings["qualified_threshold"]),
            }
    except Exception:
        logger.exception("Failed to load lead score thresholds; using env fallback")
    return {
        "hot": int(os.getenv("LEAD_SCORE_HOT_THRESHOLD", "75")),
        "warm": int(os.getenv("LEAD_SCORE_WARM_THRESHOLD", "40")),
        "qualified": int(os.getenv("LEAD_SCORE_THRESHOLD", "50")),
    }


def _qualification_threshold() -> int:
    return _lead_thresholds()["qualified"]


def _clean_text(parts: Iterable[str | None]) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _detect_rule_match(rule_name: str, description: str, text: str, conversation_row: dict[str, Any], user_row: dict[str, Any]) -> bool:
    combined = f"{rule_name} {description}".lower()
    if "price" in combined:
        return bool(PRICE_PATTERN.search(text))
    if "product interest" in combined:
        return bool(PRODUCT_PATTERN.search(text))
    if "feature" in combined:
        return bool(FEATURE_PATTERN.search(text))
    if "timeline" in combined:
        return bool(TIMELINE_PATTERN.search(text))
    if "brochure" in combined or "catalog" in combined:
        return bool(BROCHURE_PATTERN.search(text))
    if "location" in combined:
        return bool(user_row.get("address") or user_row.get("city") or LOCATION_PATTERN.search(text))
    if "multiple interactions" in combined:
        return _has_multiple_interactions(user_row["id"])
    if "qualified by ai" in combined:
        return bool(HIGH_INTENT_PATTERN.search(text))
    return combined in text.lower()


def _has_multiple_interactions(user_id: str) -> bool:
    active = conversations.get_active_conversation(user_id)
    if not active:
        return False
    all_messages = messages.get_message_count(active["id"])
    return all_messages >= 4


def _extract_product_interest(text: str) -> str | None:
    found = []
    for match in PRODUCT_PATTERN.finditer(text):
        token = match.group(0)
        if token not in found:
            found.append(token)
    return ", ".join(found[:5]) or None


def _derive_intent(text: str) -> str:
    lowered = text.lower()
    if PURCHASE_PATTERN.search(lowered):
        return "purchase"
    if "support" in lowered or "service" in lowered:
        return "support"
    return "inquiry"


def _derive_sentiment(text: str) -> str:
    if NEGATIVE_PATTERN.search(text):
        return "negative"
    if HIGH_INTENT_PATTERN.search(text):
        return "positive"
    return "neutral"


def _derive_urgency(text: str) -> str:
    if re.search(r"\b(today|now|urgent|asap|immediately)\b", text, re.IGNORECASE):
        return "high"
    if re.search(r"\b(this week|soon|next week)\b", text, re.IGNORECASE):
        return "medium"
    return "low"


def _derive_interest_level(score: int) -> str:
    thresholds = _lead_thresholds()
    if score >= thresholds["hot"]:
        return "high"
    if score >= thresholds["warm"]:
        return "medium"
    return "low"


def _derive_timeline(text: str) -> str | None:
    match = re.search(r"\b(today|tomorrow|next week|this week|next month|month|quarter)\b", text, re.IGNORECASE)
    return match.group(0) if match else None


def _derive_recommended_action(score: int) -> str:
    thresholds = _lead_thresholds()
    if score >= thresholds["hot"]:
        return "Schedule a sales call and share quotation immediately."
    if score >= thresholds["warm"]:
        return "Follow up with brochure, product specs, and pricing clarification."
    return "Keep nurturing with FAQ answers and check buying timeline."


def _build_summary(user_row: dict[str, Any], analysis_row: dict[str, Any], message_rows: list[dict[str, Any]]) -> tuple[str, list[str], dict[str, Any], bool]:
    recent_user_messages = [row["message"] for row in message_rows if row["sender"] == "USER"][-3:]
    recent_text = "; ".join(recent_user_messages) if recent_user_messages else "No customer questions captured yet."
    summary = (
        f"{user_row.get('name') or user_row['phone_number']} is a {analysis_row['interest_level']} interest lead "
        f"with {analysis_row['intent']} intent. Recent discussion: {recent_text}"
    )
    key_points = [
        f"Lead score: {analysis_row['lead_score']}",
        f"Urgency: {analysis_row['urgency']}",
        f"Product interest: {analysis_row.get('product_interest') or 'Not identified'}",
    ]
    requirements = {
        "location": user_row.get("address") or user_row.get("city"),
        "timeline": analysis_row.get("timeline"),
        "budget": analysis_row.get("budget"),
        "product_interest": analysis_row.get("product_interest"),
    }
    follow_up_needed = analysis_row["qualified"] or analysis_row["urgency"] != "low"
    return summary, key_points, requirements, follow_up_needed


def score_conversation(conversation_id: str) -> dict[str, Any] | None:
    conversation_row = conversations.get_conversation_by_id(conversation_id)
    if conversation_row is None:
        return None

    user_row = users.get_user_by_id(conversation_row["user_id"])
    if user_row is None:
        return None

    message_rows = messages.get_conversation_messages(conversation_id, limit=500)
    combined_text = _clean_text(row["message"] for row in message_rows)
    rules = lead_score_rules.get_enabled_rules()

    score = 0
    matched_rule_ids: list[str] = []
    for rule in rules:
        if _detect_rule_match(rule["rule_name"], rule.get("description") or "", combined_text, conversation_row, user_row):
            score += int(rule["weight"])
            matched_rule_ids.append(rule["id"])

    score = min(score, 100)
    qualified = score >= _qualification_threshold()
    analysis_payload = {
        "intent": _derive_intent(combined_text),
        "urgency": _derive_urgency(combined_text),
        "interest_level": _derive_interest_level(score),
        "product_interest": _extract_product_interest(combined_text),
        "budget": None,
        "timeline": _derive_timeline(combined_text),
        "sentiment": _derive_sentiment(combined_text),
        "language": user_row.get("language"),
        "lead_score": score,
        "qualified": qualified,
        "confidence": 0.82 if matched_rule_ids else 0.35,
        "recommended_action": _derive_recommended_action(score),
        "matched_rule_ids": matched_rule_ids,
    }
    analysis_row = lead_analysis.upsert_lead_analysis(conversation_id, analysis_payload)
    conversations.update_conversation_lead_score(conversation_id, score)

    desired_status = "Qualified" if qualified else user_row.get("status") or "New"
    users.update_user_state(
        user_row["id"],
        stage=user_row.get("conversation_state") or "FAQ",
        language=user_row.get("language"),
        name=user_row.get("name"),
        address=user_row.get("address"),
        city=user_row.get("city"),
        state_name=user_row.get("state"),
        country=user_row.get("country"),
        current_conversation_id=conversation_id,
        status=desired_status,
        email=user_row.get("email"),
        assigned_to=user_row.get("assigned_to"),
    )

    summary_text, key_points, requirements, follow_up_needed = _build_summary(user_row, analysis_row, message_rows)
    conversation_summary.upsert_summary(
        conversation_id,
        summary_text,
        key_points,
        requirements,
        follow_up_needed,
    )
    events.log_event(
        conversation_id,
        "lead_scored",
        {"lead_score": score, "qualified": qualified, "matched_rule_ids": matched_rule_ids},
    )
    return analysis_row
