from __future__ import annotations

"""Persistent onboarding flow for every chat channel."""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from litellm import completion as litellm_completion

from src.db import conversations, events, messages, users
from src.db.client import is_db_enabled

logger = logging.getLogger(__name__)

LANGUAGE_OPTIONS = [
    {"label": "English", "value": "english"},
    {"label": "Hindi", "value": "hindi"},
    {"label": "Bengali", "value": "bengali"},
]
MAX_CHAT_TURNS = max(1, int(os.getenv("MAX_CHAT_TURNS", "6")))
INACTIVITY_SECONDS = max(60, int(os.getenv("CONVERSATION_INACTIVITY_SECONDS", "900")))
_NIM_API_BASE = "https://integrate.api.nvidia.com/v1"

DB_TO_STAGE = {
    "LANGUAGE_SELECTION": "language", "ASK_NAME": "name", "ASK_LOCATION": "address",
    "FAQ": "chatting", "CONTACT_SHARED": "chatting", "COMPLETED": "chatting",
}
STAGE_TO_DB = {"language": "LANGUAGE_SELECTION", "name": "ASK_NAME", "address": "ASK_LOCATION", "chatting": "FAQ"}


@dataclass
class FlowResponse:
    handled: bool
    reply: str = ""
    options: list[dict[str, str]] | None = None
    images: list[str] | None = None
    preferred_language: str | None = None
    timeout_occurred: bool = False
    contact_forced: bool = False
    user_name: str | None = None
    conversation_id: str | None = None
    db_enabled: bool = False
    stage: str | None = None


class ConversationFlowManager:
    def __init__(
        self,
        state_dir: str | None = None,
        name_extractor: Callable[[str, str | None], str | None] | None = None,
    ) -> None:
        # state_dir is retained for backwards-compatible construction only.
        self._local: dict[str, dict[str, Any]] = {}
        self._name_extractor = name_extractor or self._extract_name_with_llm

    @staticmethod
    def _source(session_id: str) -> str:
        return "WHATSAPP" if session_id.startswith("whatsapp:") else "MESSENGER" if session_id.startswith("messenger:") else "WEB"

    @staticmethod
    def _initial() -> dict[str, Any]:
        return {"stage": "language", "language": None, "user_name": None, "user_address": None, "turn_count": 0, "last_active": time.time()}

    @staticmethod
    def _greeting(text: str) -> bool:
        return text.strip().lower() in {"hi", "hii", "hello", "hey", "helo", "namaste", "namaskar", "start", "begin", "menu"}

    @staticmethod
    def _language(text: str) -> str | None:
        return {"english": "English", "eng": "English", "en": "English", "hindi": "Hindi", "hin": "Hindi", "bengali": "Bengali", "bangla": "Bengali", "bn": "Bengali"}.get(text.strip().lower())

    @staticmethod
    def _normalise_extracted_name(value: object) -> str | None:
        """Validate LLM output; extraction itself is deliberately not regex based."""
        if not isinstance(value, str):
            return None
        name = " ".join(value.strip().strip(" .,!:-").split())
        if not 2 <= len(name) <= 100 or "\n" in name or not any(char.isalpha() for char in name):
            return None
        return name

    @classmethod
    def _name_from_plain_reply(cls, message: str) -> str | None:
        """Accept an unambiguous name-only reply without calling the LLM.

        The name prompt invites exactly this response (for example, ``Pritam
        Ghosh``).  Keeping this local avoids an onboarding failure when the
        external extractor is slow or unavailable, while deliberately
        rejecting sentences and numbers.
        """
        name = cls._normalise_extracted_name(message)
        if not name:
            return None
        words = name.split()
        if not 1 <= len(words) <= 4:
            return None
        return name if all(re.fullmatch(r"[^\W\d_][^\W\d_'’.-]*", word, re.UNICODE) for word in words) else None

    @staticmethod
    def _response_text(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        return str(getattr(message, "content", "") or "").strip()

    @classmethod
    def _parse_name_response(cls, response_text: str) -> str | None:
        try:
            payload = json.loads(response_text)
        except (TypeError, ValueError):
            return None
        return cls._normalise_extracted_name(payload.get("name") if isinstance(payload, dict) else None)

    @classmethod
    def _extract_name_with_llm(cls, message: str, language: str | None) -> str | None:
        """Extract only an explicitly given self-name, never infer or fabricate one."""
        api_key = os.getenv("NVIDIA_NIM_API_KEY", "").strip()
        if not api_key:
            logger.error("Name extraction is unavailable because NVIDIA_NIM_API_KEY is not configured.")
            return None
        model = os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b").strip()
        system_prompt = (
            "Extract the user's own name from their latest message for an onboarding form. "
            "They may state it in any language or phrasing. Do not infer, translate, expand, "
            "or use a name mentioned as somebody else. If the user did not clearly provide their "
            "own name, return null. Reply with exactly one JSON object matching "
            '{"name": string|null} and no Markdown or explanation.'
        )
        try:
            response = litellm_completion(
                model=f"openai/{model}", api_base=_NIM_API_BASE, api_key=api_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Preferred language: {language or 'unknown'}\nUser message: {message}"},
                ],
                temperature=0, top_p=1, max_tokens=80,
                response_format={"type": "json_object"},
            )
        except Exception:
            logger.exception("LLM name extraction failed")
            return None
        return cls._parse_name_response(cls._response_text(response))

    @staticmethod
    def _ask_name(language: str | None) -> str:
        return {"Hindi": "कृपया अपना नाम बताइए।", "Bengali": "অনুগ্রহ করে আপনার নাম বলুন।"}.get(language, "Please tell me your name.")

    @staticmethod
    def _ask_address(name: str | None, language: str | None) -> str:
        first = (name or "").split()[0]
        return {
            "Hindi": f"नमस्ते {first}! कृपया अपना शहर या पता बताइए।",
            "Bengali": f"হ্যালো {first}! অনুগ্রহ করে আপনার শহর বা ঠিকানা বলুন।",
        }.get(language, f"Hello {first}! Please tell me your city or address.")

    @staticmethod
    def _welcome(name: str, language: str | None) -> str:
        first = name.split()[0]
        return {
            "Hindi": f"नमस्ते {first}! मैं आपकी कैसे मदद कर सकता हूँ? आप हमारी मशीनों के बारे में कोई भी प्रश्न पूछ सकते हैं।",
            "Bengali": f"হ্যালো {first}! আমি কীভাবে সাহায্য করতে পারি? আমাদের মেশিন সম্পর্কে যেকোনো প্রশ্ন করুন।",
        }.get(language, f"Hello {first}! How can I help you with our machines?")

    def _hydrate(self, session_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        user = users.upsert_user(session_id, self._source(session_id))
        conversation = conversations.get_active_conversation(user["id"])
        if conversation is None:
            conversation = conversations.create_conversation(user["id"], self._source(session_id))
            events.log_event(conversation["id"], "conversation_started", {"source": self._source(session_id)})
        return ({"stage": DB_TO_STAGE.get(user.get("conversation_state"), "language"), "language": user.get("language"), "user_name": user.get("name"), "user_address": user.get("address"), "turn_count": messages.get_message_count(conversation["id"], "USER"), "last_active": user.get("last_seen").timestamp() if user.get("last_seen") else time.time()}, user, conversation)

    def _persist(self, user: dict[str, Any], conversation: dict[str, Any], state: dict[str, Any]) -> None:
        users.update_user_state(user["id"], stage=STAGE_TO_DB[state["stage"]], language=state.get("language"), name=state.get("user_name"), address=state.get("user_address"), city=user.get("city"), state_name=user.get("state"), country=user.get("country"), current_conversation_id=conversation["id"], status=user.get("status"), email=user.get("email"), assigned_to=user.get("assigned_to"))

    def handle_message(self, session_id: str, message: str) -> FlowResponse:
        user = conversation = None
        db_enabled = is_db_enabled()
        try:
            state, user, conversation = self._hydrate(session_id) if db_enabled else (self._local.setdefault(session_id, self._initial()), None, None)
        except Exception:
            if db_enabled:
                logger.exception("Unable to load database conversation state", extra={"session_id": session_id})
                raise
            state = self._local.setdefault(session_id, self._initial())

        now = time.time()
        timeout = now - float(state.get("last_active", now)) > INACTIVITY_SECONDS
        if timeout:
            state = self._initial()
            if db_enabled and user and conversation:
                conversations.close_conversation(conversation["id"], "inactive_timeout")
                conversation = conversations.create_conversation(user["id"], self._source(session_id))
        if self._greeting(message) and state["stage"] != "chatting":
            state = self._initial()
        state["last_active"] = now
        if db_enabled and user and conversation:
            users.update_user_last_seen(user["id"])
            messages.append_message(conversation["id"], "USER", message, language=state.get("language"))

        def result(**kwargs: Any) -> FlowResponse:
            if db_enabled and user and conversation:
                self._persist(user, conversation, state)
            else:
                self._local[session_id] = state
            return FlowResponse(
                timeout_occurred=timeout,
                conversation_id=conversation["id"] if conversation else None,
                db_enabled=db_enabled,
                stage=state["stage"],
                **kwargs,
            )

        if state["stage"] == "language":
            selected = self._language(message)
            if selected:
                state.update(stage="name", language=selected)
                if db_enabled and conversation:
                    events.log_event(conversation["id"], "language_selected", {"language": selected})
                return result(handled=True, reply=self._ask_name(selected), preferred_language=selected)
            return result(handled=True, reply="Welcome to Woodmaster CNC. Please choose your preferred language.", options=LANGUAGE_OPTIONS, images=["data/images/welcome_message_session_start.png"])

        if state["stage"] == "name":
            name = self._name_from_plain_reply(message) or self._name_extractor(message, state.get("language"))
            if not name:
                return result(handled=True, reply=self._ask_name(state.get("language")), preferred_language=state.get("language"))
            state.update(stage="address", user_name=name)
            return result(handled=True, reply=self._ask_address(name, state.get("language")), preferred_language=state.get("language"), user_name=name)

        if state["stage"] == "address":
            address = message.strip()
            if len(address) < 2:
                return result(handled=True, reply=self._ask_address(state.get("user_name"), state.get("language")), preferred_language=state.get("language"))
            state.update(stage="chatting", user_address=address, turn_count=0)
            if db_enabled and conversation:
                events.log_event(conversation["id"], "user_info_collected", {"name": state["user_name"], "address": address})
            return result(handled=True, reply=self._welcome(state["user_name"], state.get("language")), preferred_language=state.get("language"), user_name=state["user_name"])

        state["turn_count"] = int(state.get("turn_count", 0)) + 1
        contact_due = state["turn_count"] >= MAX_CHAT_TURNS and not bool((conversation or {}).get("contact_shared"))
        return result(handled=False, preferred_language=state.get("language"), user_name=state.get("user_name"), contact_forced=contact_due)
