from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

LANGUAGE_OPTIONS: list[dict[str, str]] = [
    {"label": "English", "value": "english"},
    {"label": "हिंदी", "value": "hindi"},
    {"label": "বাংলা", "value": "bengali"},
]


@dataclass
class FlowResponse:
    handled: bool
    reply: str = ""
    options: list[dict[str, str]] | None = None
    images: list[str] | None = None
    preferred_language: str | None = None


class ConversationFlowManager:
    """
    Manages the multi-stage conversation flow:
      1. awaiting_greeting  → user must say hi
      2. awaiting_language  → user picks language
      3. chatting           → RAG pipeline takes over

    State is persisted to JSON files so it survives server restarts.
    Works identically for web, WhatsApp, and Messenger sessions.
    """

    def __init__(self, state_dir: str | Path = "artifacts/flow_state") -> None:
        self._session_state: dict[str, dict[str, Any]] = {}
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # State persistence helpers
    # ------------------------------------------------------------------

    def _state_file(self, session_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "-", session_id)[:60]
        return self._state_dir / f"{safe}.json"

    def _get_state(self, session_id: str) -> dict[str, Any]:
        """Load state from memory-cache or disk, creating a fresh state if missing."""
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
                    "Failed to load flow state for %s: %s — resetting to fresh state.",
                    session_id, exc,
                )

        default: dict[str, Any] = {"stage": "awaiting_greeting", "language": None}
        self._session_state[session_id] = default
        return default

    def _save_state(self, session_id: str) -> None:
        """Persist the current state of this session to disk."""
        state_file = self._state_file(session_id)
        try:
            state_file.write_text(
                json.dumps(self._session_state[session_id], ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to persist flow state for %s: %s", session_id, exc)

    # ------------------------------------------------------------------
    # Message parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_greeting(message: str) -> bool:
        normalized = message.strip().lower()
        greetings = {
            "hi", "hello", "hey", "hii", "helo",
            "namaste", "namaskar", "নমস্কার", "হাই", "hola",
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
        if normalized in {"hindi", "hin", "हिन्दी", "हिंदी"}:
            return "Hindi"
        if normalized in {"bengali", "bangla", "bn", "বাংলা", "বাঙালি"}:
            return "Bengali"
        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def handle_message(self, session_id: str, message: str) -> FlowResponse:
        state = self._get_state(session_id)
        stage = state["stage"]

        # ── Stage 1: awaiting greeting ─────────────────────────────────
        if stage == "awaiting_greeting":
            if not self._is_greeting(message):
                return FlowResponse(
                    handled=True,
                    reply="Please send Hi to start the Woodmaster CNC assistant.",
                )

            state["stage"] = "awaiting_language"
            self._save_state(session_id)
            return FlowResponse(
                handled=True,
                reply="Dear Customer,\n\nTo begin, please first select your language.",
                options=LANGUAGE_OPTIONS,
                images=["data/images/welcome_message_session_start.png"],
            )

        # ── Stage 2: awaiting language selection ───────────────────────
        if stage == "awaiting_language":
            selected_language = self._parse_language(message)
            if not selected_language:
                return FlowResponse(
                    handled=True,
                    reply="Please choose a language: English, Hindi, or Bengali.",
                    options=LANGUAGE_OPTIONS,
                )

            state["stage"] = "chatting"
            state["language"] = selected_language
            self._save_state(session_id)

            if selected_language == "Hindi":
                welcome = (
                    "बहुत बढ़िया। अब हम हिंदी में बात करेंगे। "
                    "आप किस तरह की CNC मशीन देख रहे हैं?"
                )
            elif selected_language == "Bengali":
                welcome = (
                    "দারুণ। এখন আমরা বাংলায় কথা বলব। "
                    "আপনি কী ধরনের CNC মেশিন খুঁজছেন?"
                )
            else:
                welcome = (
                    "Great. We will continue in English. "
                    "What type of CNC machine are you looking for?"
                )

            return FlowResponse(
                handled=True,
                reply=welcome,
                preferred_language=selected_language,
            )

        # ── Stage 3: chatting — hand off to RAG pipeline ───────────────
        language = state.get("language")
        return FlowResponse(handled=False, preferred_language=language)
