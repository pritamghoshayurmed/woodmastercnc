from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class ConversationState:
    user_id: str
    phone_number: str
    source: str
    conversation_id: str
    stage: str
    language: str | None
    user_name: str | None
    user_address: str | None
    city: str | None
    state_name: str | None
    country: str | None
    turn_count: int
    last_seen: datetime | None
