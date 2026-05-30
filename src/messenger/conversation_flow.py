from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

LANGUAGE_OPTIONS: list[dict[str, str]] = [
    {"label": "English", "value": "english"},
    {"label": "Hindi", "value": "hindi"},
    {"label": "Bengali", "value": "bengali"},
]


@dataclass
class FlowResponse:
    handled: bool
    reply: str = ""
    options: list[dict[str, str]] | None = None
    images: list[str] | None = None
    preferred_language: str | None = None
    timeout_occurred: bool = False


class ConversationFlowManager:
    def __init__(self, state_dir: str | Path = "artifacts/flow_state") -> None:
        self._session_state: dict[str, dict[str, Any]] = {}
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def _state_file(self, session_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "-", session_id)[:60]
        return self._state_dir / f"{safe}.json"

    def _get_state(self, session_id: str) -> dict[str, Any]:
        if session_id in self._session_state:
            return self._session_state[session_id]

        state_file = self._state_file(session_id)
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                self._session_state[session_id] = data
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Failed to load flow state for %s: %s - resetting to fresh state.",
                    session_id,
                    exc,
                )

        default: dict[str, Any] = {"stage": "awaiting_greeting", "language": None}
        self._session_state[session_id] = default
        return default

    def _save_state(self, session_id: str) -> None:
        state_file = self._state_file(session_id)
        try:
            state_file.write_text(
                json.dumps(self._session_state[session_id], ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to persist flow state for %s: %s", session_id, exc)

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
<<<<<<< HEAD
        if normalized in {"hindi", "hin", "hi"}:
=======
        if normalized in {"hindi", "hin", "हिन्दी", "हिंदी"}:
>>>>>>> 0459f549af78c8063fbe9d00dd04b416dad88c04
            return "Hindi"
        if normalized in {"bengali", "bangla", "bn"}:
            return "Bengali"
        return None

    def handle_message(self, session_id: str, message: str) -> FlowResponse:
        state = self._get_state(session_id)

        now = time.time()
        last_active = state.get("last_active", now)

        timeout_occurred = False
        if now - last_active > 900:
            timeout_occurred = True
            state = {"stage": "awaiting_greeting", "language": None, "last_active": now}
            self._session_state[session_id] = state

        state["last_active"] = now
        self._save_state(session_id)

        stage = state["stage"]

        if stage == "awaiting_greeting":
            if self._is_greeting(message):
                state["stage"] = "awaiting_language"
                self._save_state(session_id)
                return FlowResponse(
                    handled=True,
                    reply="Welcome to Woodmaster CNC. Please choose your preferred language.",
                    options=LANGUAGE_OPTIONS,
                    images=["data/images/welcome_message_session_start.png"],
                    timeout_occurred=timeout_occurred,
                )

            state["stage"] = "chatting"
            state["language"] = None
            self._save_state(session_id)
            return FlowResponse(
                handled=False,
                reply="Welcome to Woodmaster CNC Assistant. I can help with your machine-related query.",
                images=["data/images/welcome_message_session_start.png"],
                timeout_occurred=timeout_occurred,
                preferred_language=None,
            )

        if stage == "awaiting_language":
            selected_language = self._parse_language(message)
            if not selected_language:
                state["stage"] = "chatting"
                state["language"] = None
                self._save_state(session_id)
                return FlowResponse(
                    handled=False,
                    reply="Okay, I will help you directly.",
                    timeout_occurred=timeout_occurred,
                    preferred_language=None,
                )

            state["stage"] = "chatting"
            state["language"] = selected_language
            self._save_state(session_id)

            welcome = f"Great. We will continue in {selected_language}. What type of CNC machine are you looking for?"
            return FlowResponse(
                handled=True,
                reply=welcome,
                preferred_language=selected_language,
            )

        language = state.get("language")
        return FlowResponse(handled=False, preferred_language=language)
