from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm
from litellm import completion as litellm_completion

from src.db.client import DatabaseDisabledError, is_db_enabled
from src.db import conversations, events, messages, users


logger = logging.getLogger(__name__)

LANGUAGE_OPTIONS: list[dict[str, str]] = [
    {"label": "English", "value": "english"},
    {"label": "Hindi", "value": "hindi"},
    {"label": "Bengali", "value": "bengali"},
]

COMPANY_PHONE = "+919434XXXXXX"
COMPANY_EMAIL = "woodmastercnc@gmail.com"

_NIM_API_KEY: str = os.getenv("NVIDIA_NIM_API_KEY", "").strip()
_NIM_API_BASE: str = "https://integrate.api.nvidia.com/v1"
_NIM_MODEL: str = os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b").strip()

litellm.set_verbose = False

_MAX_CHAT_TURNS: int = int(os.getenv("MAX_CHAT_TURNS", "12"))

STAGE_TO_DB_STATE = {
    "awaiting_greeting": "LANGUAGE_SELECTION",
    "awaiting_language": "LANGUAGE_SELECTION",
    "awaiting_user_info": "ASK_NAME",
    "chatting": "FAQ",
    "contact_forced": "CONTACT_SHARED",
}

DB_STATE_TO_STAGE = {
    "LANGUAGE_SELECTION": "awaiting_language",
    "ASK_NAME": "awaiting_user_info",
    "ASK_LOCATION": "awaiting_user_info",
    "FAQ": "chatting",
    "CONTACT_SHARED": "contact_forced",
    "COMPLETED": "chatting",
}


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


class ConversationFlowManager:
    def __init__(self, state_dir: str | Path = "artifacts/flow_state") -> None:
        self._session_state: dict[str, dict[str, Any]] = {}
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def _db_available(self) -> bool:
        return is_db_enabled()

    @staticmethod
    def _session_source(session_id: str) -> str:
        lowered = session_id.lower()
        if lowered.startswith("whatsapp:"):
            return "WHATSAPP"
        if lowered.startswith("messenger:"):
            return "MESSENGER"
        return "WEB"

    def _state_file(self, session_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "-", session_id)[:60]
        return self._state_dir / f"{safe}.json"

    def _default_state(self) -> dict[str, Any]:
        return {
            "stage": "awaiting_greeting",
            "language": None,
            "user_name": None,
            "user_address": None,
            "turn_count": 0,
        }

    def _get_file_state(self, session_id: str) -> dict[str, Any]:
        if session_id in self._session_state:
            return self._session_state[session_id]

        state_file = self._state_file(session_id)
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                self._session_state[session_id] = data
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load flow state for %s: %s - resetting.", session_id, exc)

        default = self._default_state()
        self._session_state[session_id] = default
        return default

    def _save_file_state(self, session_id: str) -> None:
        state_file = self._state_file(session_id)
        try:
            state_file.write_text(
                json.dumps(self._session_state[session_id], ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to persist flow state for %s: %s", session_id, exc)

    def _hydrate_db_state(self, session_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        source = self._session_source(session_id)
        user_row = users.upsert_user(session_id, source)
        conversation_row = conversations.get_active_conversation(user_row["id"])
        if conversation_row is None:
            conversation_row = conversations.create_conversation(user_row["id"], source)
            events.log_event(conversation_row["id"], "conversation_started", {"source": source})

        stage = DB_STATE_TO_STAGE.get(user_row.get("conversation_state") or "LANGUAGE_SELECTION", "awaiting_language")
        state = {
            "stage": stage,
            "language": user_row.get("language"),
            "user_name": user_row.get("name"),
            "user_address": user_row.get("address"),
            "turn_count": messages.get_message_count(conversation_row["id"], sender="USER"),
            "last_active": time.time(),
        }
        return state, user_row, conversation_row

    def _persist_db_state(
        self,
        user_row: dict[str, Any],
        conversation_row: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        users.update_user_state(
            user_row["id"],
            stage=STAGE_TO_DB_STATE.get(state["stage"], "FAQ"),
            language=state.get("language"),
            name=state.get("user_name"),
            address=state.get("user_address"),
            city=user_row.get("city"),
            state_name=user_row.get("state"),
            country=user_row.get("country"),
            current_conversation_id=conversation_row["id"],
            status=user_row.get("status"),
            email=user_row.get("email"),
            assigned_to=user_row.get("assigned_to"),
        )

    @staticmethod
    def _is_greeting(message: str) -> bool:
        normalized = message.strip().lower()
        greetings = {
            "hi", "hello", "hey", "hii", "helo",
            "namaste", "namaskar", "hola",
            "start", "begin", "menu",
        }
        if normalized in greetings:
            return True
        return normalized.startswith("hi ") or normalized.startswith("hello ")

    @staticmethod
    def _parse_language(message: str) -> str | None:
        normalized = message.strip().lower()
        if normalized in {"english", "eng", "en"}:
            return "English"
        if normalized in {"hindi", "hin", "hi"}:
            return "Hindi"
        if normalized in {"bengali", "bangla", "bn"}:
            return "Bengali"
        return None

    def _parse_name_address(self, message: str) -> tuple[str | None, str | None]:
        if not _NIM_API_KEY:
            parts = [p.strip() for p in message.split(",") if p.strip()]
            if len(parts) >= 2:
                return parts[0], ", ".join(parts[1:])
            return None, None

        system_msg = (
            "You are a precise data-extraction assistant. "
            "Identify the person's name and address from the user's message. "
            'Output exactly one JSON object: {"name": "...", "address": "..."}'
        )
        user_msg = f"Message from user: {message.strip()}\nExtract the name and address. Output only JSON."

        try:
            response = litellm_completion(
                model=f"openai/{_NIM_MODEL}",
                api_base=_NIM_API_BASE,
                api_key=_NIM_API_KEY,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                top_p=1,
                max_tokens=1024,
            )
            raw = ""
            choices = getattr(response, "choices", None) or []
            if choices:
                msg_obj = getattr(choices[0], "message", None)
                raw = (getattr(msg_obj, "content", "") or "").strip()
            raw = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", raw, flags=re.DOTALL | re.IGNORECASE)
            raw = re.sub(r"</?think[^>]*>", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"```[a-z]*\n?", "", raw, flags=re.IGNORECASE).strip()
            json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if not json_match:
                return None, None
            data = json.loads(json_match.group())
            name = (data.get("name") or "").strip() or None
            address = (data.get("address") or "").strip() or None
            if name and address and len(name) >= 2 and len(address) >= 3:
                return name, address
            return None, None
        except Exception as exc:
            logger.warning("LLM name/address extraction failed: %s", exc)
            return None, None

    def _ask_user_info(self, language: str | None) -> str:
        if language == "Bengali":
            return "অনুগ্রহ করে আপনার নাম এবং ঠিকানা জানান।\n(যেমন: প্রীতম ঘোষ, আরামবাগ, হুগলি)"
        if language == "Hindi":
            return "कृपया अपना नाम और पता बताएं।\n(जैसे: राम शर्मा, नई दिल्ली)"
        return "Please tell us your name and your address.\n(e.g. John Smith, New Delhi)"

    def _re_ask_user_info(self, language: str | None) -> str:
        if language == "Bengali":
            return "আমি আপনার নাম বা ঠিকানা বুঝতে পারিনি। অনুগ্রহ করে আবার লিখুন।\n(যেমন: প্রীতম ঘোষ, আরামবাগ, হুগলি)"
        if language == "Hindi":
            return "मुझे आपका नाम या पता समझ नहीं आया। कृपया फिर से लिखें।\n(जैसे: राम शर्मा, नई दिल्ली)"
        return "I could not catch your name or address. Please try again.\n(e.g. John Smith, New Delhi)"

    def _welcome_after_info(self, name: str, language: str | None) -> str:
        first = name.split()[0]
        if language == "Bengali":
            return f"স্বাগতম {first}! আপনি আমাদের মেশিন সম্পর্কে যে কোনো প্রশ্ন করতে পারেন।"
        if language == "Hindi":
            return f"स्वागत है {first}! आप हमारी मशीनों के बारे में कोई भी प्रश्न पूछ सकते हैं।"
        return f"Welcome {first}, how can we help you regarding your queries related to our machines?"

    def handle_message(self, session_id: str, message: str) -> FlowResponse:
        db_enabled = False
        user_row: dict[str, Any] | None = None
        conversation_row: dict[str, Any] | None = None

        try:
            if self._db_available():
                state, user_row, conversation_row = self._hydrate_db_state(session_id)
                db_enabled = True
            else:
                state = self._get_file_state(session_id)
        except DatabaseDisabledError:
            state = self._get_file_state(session_id)
        except Exception:
            logger.exception("Falling back to file-backed flow state for %s", session_id)
            state = self._get_file_state(session_id)

        now = time.time()
        timeout_occurred = False
        last_active = float(state.get("last_active", now))
        if now - last_active > 900:
            timeout_occurred = True
            if db_enabled and user_row and conversation_row:
                conversations.close_conversation(conversation_row["id"], "inactive_timeout")
                events.log_event(conversation_row["id"], "conversation_closed", {"reason": "inactive_timeout"})
                conversation_row = conversations.create_conversation(user_row["id"], self._session_source(session_id))
                state = self._default_state()
            else:
                state = self._default_state()

        state["last_active"] = now
        if db_enabled and user_row and conversation_row:
            users.update_user_last_seen(user_row["id"])
            messages.append_message(
                conversation_row["id"],
                "USER",
                message,
                language=state.get("language"),
            )
        else:
            self._session_state[session_id] = state
            self._save_file_state(session_id)

        stage = state["stage"]
        language = state.get("language")
        user_name = state.get("user_name")

        if stage == "awaiting_greeting":
            if self._is_greeting(message):
                state["stage"] = "awaiting_language"
                reply = "Welcome to Woodmaster CNC. Please choose your preferred language."
                if db_enabled and user_row and conversation_row:
                    self._persist_db_state(user_row, conversation_row, state)
                else:
                    self._session_state[session_id] = state
                    self._save_file_state(session_id)
                return FlowResponse(
                    handled=True,
                    reply=reply,
                    options=LANGUAGE_OPTIONS,
                    images=["data/images/welcome_message_session_start.png"],
                    timeout_occurred=timeout_occurred,
                    conversation_id=conversation_row["id"] if conversation_row else None,
                    db_enabled=db_enabled,
                )

            state["stage"] = "chatting"
            state["language"] = None
            state["turn_count"] = 0
            reply = "Welcome to Woodmaster CNC Assistant. I can help with your machine-related query."
            if db_enabled and user_row and conversation_row:
                self._persist_db_state(user_row, conversation_row, state)
            else:
                self._session_state[session_id] = state
                self._save_file_state(session_id)
            return FlowResponse(
                handled=False,
                reply=reply,
                images=["data/images/welcome_message_session_start.png"],
                timeout_occurred=timeout_occurred,
                preferred_language=None,
                conversation_id=conversation_row["id"] if conversation_row else None,
                db_enabled=db_enabled,
            )

        if stage == "awaiting_language":
            selected = self._parse_language(message)
            if not selected:
                return FlowResponse(
                    handled=True,
                    reply="Please choose a language from the options below.",
                    options=LANGUAGE_OPTIONS,
                    timeout_occurred=timeout_occurred,
                    conversation_id=conversation_row["id"] if conversation_row else None,
                    db_enabled=db_enabled,
                )

            state["language"] = selected
            state["stage"] = "awaiting_user_info"
            if db_enabled and user_row and conversation_row:
                self._persist_db_state(user_row, conversation_row, state)
                events.log_event(conversation_row["id"], "language_selected", {"language": selected})
            else:
                self._session_state[session_id] = state
                self._save_file_state(session_id)
            return FlowResponse(
                handled=True,
                reply=self._ask_user_info(selected),
                preferred_language=selected,
                timeout_occurred=timeout_occurred,
                conversation_id=conversation_row["id"] if conversation_row else None,
                db_enabled=db_enabled,
            )

        if stage == "awaiting_user_info":
            name, address = self._parse_name_address(message)
            if not name or not address:
                return FlowResponse(
                    handled=True,
                    reply=self._re_ask_user_info(language),
                    preferred_language=language,
                    timeout_occurred=timeout_occurred,
                    conversation_id=conversation_row["id"] if conversation_row else None,
                    db_enabled=db_enabled,
                )

            state["user_name"] = name
            state["user_address"] = address
            state["stage"] = "chatting"
            state["turn_count"] = 0
            if db_enabled and user_row and conversation_row:
                self._persist_db_state(user_row, conversation_row, state)
                events.log_event(conversation_row["id"], "user_info_collected", {"name": name, "address": address})
            else:
                self._session_state[session_id] = state
                self._save_file_state(session_id)
            return FlowResponse(
                handled=True,
                reply=self._welcome_after_info(name, language),
                preferred_language=language,
                user_name=name,
                timeout_occurred=timeout_occurred,
                conversation_id=conversation_row["id"] if conversation_row else None,
                db_enabled=db_enabled,
            )

        if stage == "contact_forced":
            state["stage"] = "chatting"

        turn_count = int(state.get("turn_count", 0)) + 1
        state["turn_count"] = turn_count
        if db_enabled and user_row and conversation_row:
            self._persist_db_state(user_row, conversation_row, state)
        else:
            self._session_state[session_id] = state
            self._save_file_state(session_id)

        return FlowResponse(
            handled=False,
            preferred_language=language,
            user_name=user_name,
            contact_forced=turn_count > _MAX_CHAT_TURNS,
            timeout_occurred=timeout_occurred,
            conversation_id=conversation_row["id"] if conversation_row else None,
            db_enabled=db_enabled,
        )
