from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.db import lead_score_rules


router = APIRouter()


class RuleCreateRequest(BaseModel):
    name: str
    description: str = ""
    points: int = Field(10, ge=0, le=100)
    matchingType: str = "AI Detected (Recommended)"
    status: bool = True


class RuleUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    points: int | None = Field(None, ge=0, le=100)
    matchingType: str | None = None
    status: bool | None = None
    priority: int | None = None


class RuleReorderRequest(BaseModel):
    orderedIds: list[str]


def _to_payload(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["rule_name"],
        "description": row.get("description") or "",
        "points": row["weight"],
        "status": row["enabled"],
        "matchingType": row["matching_type"],
        "priority": row["priority"],
        "icon": "Lightbulb",
    }


@router.get("")
def get_rules() -> dict:
    return {"rules": [_to_payload(row) for row in lead_score_rules.get_all_rules()]}


@router.post("")
def create_rule(body: RuleCreateRequest) -> dict:
    priority = len(lead_score_rules.get_all_rules()) + 1
    row = lead_score_rules.create_rule(body.name, body.description, body.points, priority, body.matchingType)
    if body.status is False:
        row = lead_score_rules.update_rule(row["id"], {"enabled": False}) or row
    return {"rule": _to_payload(row)}


@router.put("/{rule_id}")
def update_rule(rule_id: str, body: RuleUpdateRequest) -> dict:
    row = lead_score_rules.update_rule(
        rule_id,
        {
            "rule_name": body.name,
            "description": body.description,
            "weight": body.points,
            "enabled": body.status,
            "matching_type": body.matchingType,
            "priority": body.priority,
        },
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Rule not found.")
    return {"rule": _to_payload(row)}


@router.delete("/{rule_id}")
def delete_rule(rule_id: str) -> dict:
    deleted = lead_score_rules.delete_rule(rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Rule not found.")
    return {"deleted": True}


@router.post("/reorder")
def reorder_rules(body: RuleReorderRequest) -> dict:
    lead_score_rules.reorder_rules(body.orderedIds)
    return {"rules": [_to_payload(row) for row in lead_score_rules.get_all_rules()]}
