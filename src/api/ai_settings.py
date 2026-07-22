from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.db import ai_settings as db_ai_settings

router = APIRouter()


class AiSettingsUpdateRequest(BaseModel):
    systemPrompt: str | None = None
    hotLeadThreshold: int | None = Field(None, ge=0, le=100)
    warmLeadThreshold: int | None = Field(None, ge=0, le=100)
    qualifiedThreshold: int | None = Field(None, ge=0, le=100)
    maxChatTurns: int | None = Field(None, ge=1, le=100)
    inactivityTimeoutSeconds: int | None = Field(None, ge=60, le=86400)


def _to_payload(settings: dict) -> dict:
    return {
        "systemPrompt": settings.get("system_prompt"),
        "hotLeadThreshold": settings["hot_lead_threshold"],
        "warmLeadThreshold": settings["warm_lead_threshold"],
        "qualifiedThreshold": settings["qualified_threshold"],
        "maxChatTurns": settings["max_chat_turns"],
        "inactivityTimeoutSeconds": settings["inactivity_timeout_seconds"],
    }


@router.get("")
def get_settings() -> dict:
    return {"settings": _to_payload(db_ai_settings.get_settings(fresh=True))}


@router.put("")
def update_settings(body: AiSettingsUpdateRequest) -> dict:
    fields = {
        "system_prompt": body.systemPrompt,
        "hot_lead_threshold": body.hotLeadThreshold,
        "warm_lead_threshold": body.warmLeadThreshold,
        "qualified_threshold": body.qualifiedThreshold,
        "max_chat_turns": body.maxChatTurns,
        "inactivity_timeout_seconds": body.inactivityTimeoutSeconds,
    }
    settings = db_ai_settings.update_settings(fields)
    return {"settings": _to_payload(settings)}
