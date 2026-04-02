from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    def __init__(self) -> None:
        self._session_state: dict[str, dict[str, Any]] = {}

    def _get_state(self, session_id: str) -> dict[str, Any]:
        return self._session_state.setdefault(
            session_id,
            {
                "stage": "awaiting_greeting",
                "language": None,
            },
        )

    @staticmethod
    def _is_greeting(message: str) -> bool:
        normalized = message.strip().lower()
        greetings = {
            "hi",
            "hello",
            "hey",
            "hii",
            "helo",
            "namaste",
            "namaskar",
            "নমস্কার",
            "হাই",
            "hola",
        }
        if normalized in greetings:
            return True
        return normalized.startswith("hi ") or normalized.startswith("hello ")

    @staticmethod
    def _parse_language(message: str) -> str | None:
        normalized = message.strip().lower()
        if normalized in {"english", "eng", "en"}:
            return "English"
        if normalized in {"hindi", "hin", "hi", "हिन्दी", "हिंदी"}:
            return "Hindi"
        if normalized in {"bengali", "bangla", "bn", "বাংলা", "বাঙালি"}:
            return "Bengali"
        return None

    def handle_message(self, session_id: str, message: str) -> FlowResponse:
        state = self._get_state(session_id)
        stage = state["stage"]

        if stage == "awaiting_greeting":
            if not self._is_greeting(message):
                return FlowResponse(
                    handled=True,
                    reply="Please send Hi to start the Woodmaster CNC assistant.",
                )

            state["stage"] = "awaiting_language"
            return FlowResponse(
                handled=True,
                reply=(
                    "Dear Customer,\n\nTo begin, please first select your language."
                ),
                options=LANGUAGE_OPTIONS,
                images=["data/images/welcome_message_session_start.png"],
            )

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

        language = state.get("language")
        return FlowResponse(handled=False, preferred_language=language)
