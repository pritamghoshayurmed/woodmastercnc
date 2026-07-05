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


logger = logging.getLogger(__name__)

LANGUAGE_OPTIONS: list[dict[str, str]] = [
    {"label": "English", "value": "english"},
    {"label": "Hindi", "value": "hindi"},
    {"label": "Bengali", "value": "bengali"},
]

# ── Contact details (update COMPANY_PHONE with your real number) ───────────────
COMPANY_PHONE = "+919434XXXXXX"   # e.g. +919876543210
COMPANY_EMAIL = "woodmastercnc@gmail.com"

# ── LLM credentials (same keys as RAG pipeline) ───────────────────────────────
_NIM_API_KEY: str = os.getenv("NVIDIA_NIM_API_KEY", "").strip()
_NIM_API_BASE: str = "https://integrate.api.nvidia.com/v1"
_NIM_MODEL: str = os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b").strip()

# Silence LiteLLM's per-request debug noise
litellm.set_verbose = False

# ── Turn limit (read from .env → MAX_CHAT_TURNS, default 12) ──────────────────
_MAX_CHAT_TURNS: int = int(os.getenv("MAX_CHAT_TURNS", "12"))


@dataclass
class FlowResponse:
    handled: bool
    reply: str = ""
    options: list[dict[str, str]] | None = None
    images: list[str] | None = None
    preferred_language: str | None = None
    timeout_occurred: bool = False
    contact_forced: bool = False      # True when turn limit is reached
    user_name: str | None = None      # Populated once name is collected


class ConversationFlowManager:
    """
    Flow stages
    -----------
    awaiting_greeting -> awaiting_language -> awaiting_user_info
                      -> chatting -> contact_forced
    """

    def __init__(self, state_dir: str | Path = "artifacts/flow_state") -> None:
        self._session_state: dict[str, dict[str, Any]] = {}
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)

    # ── State persistence ──────────────────────────────────────────────────────

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
                    "Failed to load flow state for %s: %s - resetting.",
                    session_id, exc,
                )

        default: dict[str, Any] = {
            "stage": "awaiting_greeting",
            "language": None,
            "user_name": None,
            "user_address": None,
            "turn_count": 0,
        }
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

    # ── Static helpers ─────────────────────────────────────────────────────────

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
        """
        Uses the LLM to extract a person's name and address from a free-form
        message in any language and any format.

        Handles models that emit <think>…</think> reasoning blocks (e.g. Nemotron
        Ultra) by stripping those tags before JSON extraction, and searches for
        the JSON object anywhere in the response text rather than assuming the
        entire output is JSON.

        Returns (name, address) on success, or (None, None) when the model
        cannot confidently extract both fields (triggers re-ask loop).
        """
        if not _NIM_API_KEY:
            # No API key: basic comma-split fallback so the flow still works
            parts = [p.strip() for p in message.split(",") if p.strip()]
            if len(parts) >= 2:
                return parts[0], ", ".join(parts[1:])
            return None, None

        system_msg = (
            "You are a precise data-extraction assistant. "
            "Your ONLY task is to identify the person's name and address "
            "from the user's message. "
            "After any reasoning, output exactly one JSON object and nothing else:\n"
            '{"name": "<full name or null>", "address": "<full address or null>"}\n'
            "Use null (not a string) when a field is absent."
        )

        user_msg = (
            f"Message from user: {message.strip()}\n\n"
            "Extract the name and address. "
            "Output ONLY the JSON object."
        )

        try:
            response = litellm_completion(
                model=f"openai/{_NIM_MODEL}",
                api_base=_NIM_API_BASE,
                api_key=_NIM_API_KEY,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                top_p=1,
                # Nemotron Ultra needs ~500-800 tokens for its <think> block
                # before it emits the actual JSON answer — give it room.
                max_tokens=1024,
            )

            raw = ""
            choices = getattr(response, "choices", None) or []
            if choices:
                msg_obj = getattr(choices[0], "message", None)
                raw = (getattr(msg_obj, "content", "") or "").strip()

            logger.debug("LLM raw extraction response: %r", raw[:300])

            # ── 1. Strip <think>…</think> reasoning blocks ─────────────────────
            raw = re.sub(
                r"<think\b[^>]*>.*?</think\s*>",
                "",
                raw,
                flags=re.DOTALL | re.IGNORECASE,
            )
            # Remove any remaining lone think tags
            raw = re.sub(r"</?think[^>]*>", "", raw, flags=re.IGNORECASE)

            # ── 2. Strip markdown code fences ─────────────────────────────────
            raw = re.sub(r"```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
            raw = raw.strip()

            # ── 3. Find the first {...} JSON object anywhere in the text ───────
            json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if not json_match:
                logger.warning(
                    "LLM name/address extraction: no JSON object found in response. raw=%r",
                    raw[:200],
                )
                return None, None

            data = json.loads(json_match.group())
            name    = (data.get("name")    or "").strip() or None
            address = (data.get("address") or "").strip() or None

            # Reject trivially short / null values
            if name and address and len(name) >= 2 and len(address) >= 3:
                logger.info(
                    "LLM extracted name=%r address=%r from %r", name, address, message[:60]
                )
                return name, address

            logger.warning(
                "LLM returned insufficient name/address: name=%r address=%r", name, address
            )
            return None, None

        except Exception as exc:
            logger.warning("LLM name/address extraction failed: %s", exc)
            return None, None


    # ── Localised message builders ─────────────────────────────────────────────

    def _ask_user_info(self, language: str | None) -> str:
        if language == "Bengali":
            return (
                "অনুগ্রহ করে আপনার নাম এবং ঠিকানা জানান।\n"
                "(যেমন: প্রিতম ঘোষ, আরামবাগ, হুগলি)"
            )
        if language == "Hindi":
            return (
                "कृपया अपना नाम और पता बताएं।\n"
                "(जैसे: राम शर्मा, नई दिल्ली)"
            )
        return (
            "Please tell us your name and your address.\n"
            "(e.g. John Smith, New Delhi)"
        )

    def _re_ask_user_info(self, language: str | None) -> str:
        if language == "Bengali":
            return (
                "আমি আপনার নাম বা ঠিকানা বুঝতে পারিনি। "
                "অনুগ্রহ করে আবার লিখুন।\n"
                "(যেমন: প্রিতম ঘোষ, আরামবাগ, হুগলি)"
            )
        if language == "Hindi":
            return (
                "मुझे आपका नाम या पता समझ नहीं आया। "
                "कृपया फिर से लिखें।\n"
                "(जैसे: राम शर्मा, नई दिल्ली)"
            )
        return (
            "I could not catch your name or address. "
            "Please try again.\n"
            "(e.g. John Smith, New Delhi)"
        )

    def _welcome_after_info(self, name: str, language: str | None) -> str:
        first = name.split()[0]
        if language == "Bengali":
            return (
                f"স্বাগতম {first}! আপনি আমাদের মেশিন সম্পর্কে "
                "যে কোনো প্রশ্ন করতে পারেন।"
            )
        if language == "Hindi":
            return (
                f"स्वागत है {first}! आप हमारी मशीनों के बारे में "
                "कोई भी प्रश्न पूछ सकते हैं।"
            )
        return (
            f"Welcome {first}, how can we help you "
            "regarding your queries related to our machines?"
        )

    def _contact_message(self, user_name: str | None, language: str | None) -> str:
        greeting = f"{user_name.split()[0]}, " if user_name else ""
        if language == "Bengali":
            return (
                f"{greeting}আপনার সাথে কথা বলে খুব ভালো লাগলো! "
                "আরও বিস্তারিত সহায়তার জন্য অনুগ্রহ করে আমাদের সাথে সরাসরি যোগাযোগ করুন।"
            )
        if language == "Hindi":
            return (
                f"{greeting}आपसे बात करके बहुत अच्छा लगा! "
                "अधिक सहायता के लिए कृपया हमसे सीधे संपर्क करें।"
            )
        return (
            f"{greeting}it was great talking with you! "
            "For further assistance, please contact us directly."
        )

    # ── Main message handler ───────────────────────────────────────────────────

    def handle_message(self, session_id: str, message: str) -> FlowResponse:
        state = self._get_state(session_id)

        now = time.time()
        last_active = state.get("last_active", now)

        timeout_occurred = False
        if now - last_active > 900:   # 15-minute inactivity reset
            timeout_occurred = True
            state = {
                "stage": "awaiting_greeting",
                "language": None,
                "user_name": None,
                "user_address": None,
                "turn_count": 0,
                "last_active": now,
            }
            self._session_state[session_id] = state

        state["last_active"] = now
        self._save_state(session_id)

        stage = state["stage"]
        language = state.get("language")
        user_name = state.get("user_name")

        # ── awaiting_greeting ──────────────────────────────────────────────────
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

            # Direct non-greeting first message → skip setup, go straight to chat
            state["stage"] = "chatting"
            state["language"] = None
            state["turn_count"] = 0
            self._save_state(session_id)
            return FlowResponse(
                handled=False,
                reply="Welcome to Woodmaster CNC Assistant. I can help with your machine-related query.",
                images=["data/images/welcome_message_session_start.png"],
                timeout_occurred=timeout_occurred,
                preferred_language=None,
            )

        # ── awaiting_language ──────────────────────────────────────────────────
        if stage == "awaiting_language":
            selected = self._parse_language(message)
            if not selected:
                return FlowResponse(
                    handled=True,
                    reply="Please choose a language from the options below.",
                    options=LANGUAGE_OPTIONS,
                    timeout_occurred=timeout_occurred,
                )

            state["language"] = selected
            state["stage"] = "awaiting_user_info"
            self._save_state(session_id)

            return FlowResponse(
                handled=True,
                reply=self._ask_user_info(selected),
                preferred_language=selected,
                timeout_occurred=timeout_occurred,
            )

        # ── awaiting_user_info ─────────────────────────────────────────────────
        if stage == "awaiting_user_info":
            name, address = self._parse_name_address(message)

            if not name or not address:
                return FlowResponse(
                    handled=True,
                    reply=self._re_ask_user_info(language),
                    preferred_language=language,
                    timeout_occurred=timeout_occurred,
                )

            state["user_name"] = name
            state["user_address"] = address
            state["stage"] = "chatting"
            state["turn_count"] = 0
            self._save_state(session_id)

            return FlowResponse(
                handled=True,
                reply=self._welcome_after_info(name, language),
                preferred_language=language,
                user_name=name,
                timeout_occurred=timeout_occurred,
            )

        # ── contact_forced (legacy stage — redirect back to chatting) ─────────
        # This stage existed in an older version; if a session file still has
        # it, treat it as chatting so the user gets RAG answers again.
        if stage == "contact_forced":
            state["stage"] = "chatting"
            self._save_state(session_id)

        # ── chatting ───────────────────────────────────────────────────────────
        turn_count = state.get("turn_count", 0) + 1
        state["turn_count"] = turn_count
        self._save_state(session_id)

        # Past the turn limit → LLM still answers, but frontend also shows
        # the interactive contact card so the user can reach the company.
        over_limit = turn_count > _MAX_CHAT_TURNS

        return FlowResponse(
            handled=False,           # always let RAG answer the question
            preferred_language=language,
            user_name=user_name,
            contact_forced=over_limit,
        )
