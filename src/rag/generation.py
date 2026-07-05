"""
LiteLLM-backed answer generator.

Routes all LLM calls through LiteLLM, which forwards them to NVIDIA NIM
(or any other configured provider) using the OpenAI-compatible completion
interface. This means:

  • A single `litellm.completion()` call works for any model/provider.
  • Errors are normalised to OpenAI exception types.
  • Retry / fallback logic below is provider-agnostic.

NVIDIA NIM model strings take the form:
    nvidia_nim/<namespace>/<model-id>
e.g. nvidia_nim/nvidia/llama-3.1-nemotron-ultra-253b-v1
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

import litellm
from litellm import completion as litellm_completion

from src.rag.prompt import build_answer_prompt, build_reformulation_prompt, build_system_prompt

logger = logging.getLogger(__name__)

# Maximum number of prior messages (user+assistant pairs) to include in
# context.  Each pair = 2 messages; 3 pairs = 6 messages total.
_MAX_HISTORY_MESSAGES = 6

# LiteLLM: silence per-request verbose logging in production.
litellm.set_verbose = False


# ─── Custom Exceptions ─────────────────────────────────────────────────────────

class GeneratorRateLimitError(RuntimeError):
    """Raised when the upstream LLM returns HTTP 429 (rate limit)."""


class LLMSettingsError(RuntimeError):
    """Raised when required LLM configuration is missing or invalid."""


# ─── Generator ─────────────────────────────────────────────────────────────────

class LiteLLMGenerator:
    """
    Answer generator backed by LiteLLM → NVIDIA NIM (or any LiteLLM provider).

    All public methods maintain the same API contract as the original generator
    RAGPipeline requires no logic changes.
    """

    # NVIDIA NIM OpenAI-compatible base URL.
    _NIM_API_BASE = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "nvidia/nemotron-3-ultra-550b-a55b",
        fallback_model: str = "meta/llama-3.3-70b-instruct",
        temperature: float = 0.3,
        top_p: float = 0.85,
        max_tokens: int = 512,
    ) -> None:
        if not api_key:
            raise LLMSettingsError(
                "NVIDIA NIM API key is required. "
                "Set NVIDIA_NIM_API_KEY in your environment."
            )

        # Store the key so _call_llm can pass it directly to LiteLLM.
        # This avoids relying on LiteLLM's nvidia_nim provider env-var routing,
        # which has inconsistent model-name handling. Instead we use the openai/
        # provider with an explicit api_base — identical to what the OpenAI SDK
        # does but unified through LiteLLM.
        self._api_key = api_key
        self.model = model
        self.fallback_models: list[str] = [fallback_model] if fallback_model != model else []
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    # ─── Prompt helpers ────────────────────────────────────────────────────────

    def _compose_system_prompt(
        self,
        preferred_language: Optional[str] = None,
        suppress_thinking: bool = False,
    ) -> str:
        return build_system_prompt(
            preferred_language=preferred_language,
            suppress_thinking=suppress_thinking,
        )

    # ─── Text utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _contains_thinking_tag(text: str | None) -> bool:
        if not text:
            return False
        return bool(re.search(r"</?\\s*think\\b[^>]*>", text, flags=re.IGNORECASE))

    @staticmethod
    def _strip_thinking_tags(text: str | None) -> str:
        if not text:
            return ""
        cleaned = re.sub(
            r"<think\b[^>]*>.*?</think\s*>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = re.sub(r"</?\\s*think\\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    @staticmethod
    def _clean_response_text(text: str | None) -> str:
        if not text:
            return ""
        cleaned = LiteLLMGenerator._strip_thinking_tags(text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _looks_incomplete(text: str) -> bool:
        if not text:
            return True
        stripped = text.rstrip()
        if len(stripped) < 40:
            return False
        return stripped[-1] not in {".", "!", "?", '"', "'", ")", "]"}

    # ─── Message building ──────────────────────────────────────────────────────

    def _build_completion_messages(
        self,
        question: str,
        context: str,
        history: List[Dict[str, str]],
        preferred_language: Optional[str] = None,
        suppress_thinking: bool = False,
    ) -> List[Any]:
        messages: List[Any] = [
            {
                "role": "system",
                "content": self._compose_system_prompt(
                    preferred_language=preferred_language,
                    suppress_thinking=suppress_thinking,
                ),
            }
        ]
        for turn in history:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": turn.get("content", "")})
        messages.append(
            {"role": "user", "content": build_answer_prompt(question=question, context=context)}
        )
        return messages

    # ─── LiteLLM call wrapper ──────────────────────────────────────────────────

    def _call_llm(
        self,
        messages: List[Any],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Any:
        """
        Single point of contact with LiteLLM.

        Uses the ``openai/`` provider with NVIDIA NIM's base URL so that
        LiteLLM treats it as an OpenAI-compatible endpoint. This is the
        most reliable routing strategy — identical in behaviour to using
        the OpenAI SDK directly but going through the LiteLLM gateway.

        Model strings are plain NVIDIA NIM model IDs (e.g.
        ``nvidia/nemotron-3-ultra-550b-a55b``) — the ``openai/`` prefix is
        added here automatically.

        Raises:
            GeneratorRateLimitError: on HTTP 429.
            LLMSettingsError: on authentication failure.
            litellm.NotFoundError / RuntimeError: on other failures.
        """
        try:
            return litellm_completion(
                model=f"openai/{model}",
                api_base=self._NIM_API_BASE,
                api_key=self._api_key,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except litellm.RateLimitError as exc:
            raise GeneratorRateLimitError("Rate limit reached.") from exc
        except litellm.AuthenticationError as exc:
            raise LLMSettingsError(
                "Authentication failed — check NVIDIA_NIM_API_KEY."
            ) from exc

    # ─── Response extraction ───────────────────────────────────────────────────

    @classmethod
    def _extract_raw_message_text(cls, response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None
        return str(content or "")

    @classmethod
    def _extract_message_text(cls, response: Any) -> str:
        return cls._clean_response_text(cls._extract_raw_message_text(response))

    @staticmethod
    def _finish_reason(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        return str(getattr(choices[0], "finish_reason", "") or "").strip().lower()

    # ─── Repair helpers ────────────────────────────────────────────────────────

    def _repair_thinking_answer(
        self,
        draft: str,
        preferred_language: Optional[str],
        model_name: str,
    ) -> str:
        repair_prompt = (
            "Rewrite the following draft as the final customer-facing answer only. "
            "Remove any <think> tags, internal reasoning, or analysis. "
            "Keep the answer short and finish with one short follow-up question.\n\n"
            f"Draft:\n{draft}"
        )
        response = self._call_llm(
            messages=[
                {
                    "role": "system",
                    "content": self._compose_system_prompt(
                        preferred_language=preferred_language,
                        suppress_thinking=True,
                    ),
                },
                {"role": "user", "content": repair_prompt},
            ],
            model=model_name,
            temperature=0.2,
            top_p=self.top_p,
            max_tokens=min(self.max_tokens + 120, 512),
        )
        repaired = self._extract_message_text(response)
        return repaired or draft

    def _repair_incomplete_answer(
        self,
        draft: str,
        preferred_language: Optional[str],
        model_name: str,
    ) -> str:
        repair_prompt = (
            "Rewrite the following incomplete draft into a complete, short customer-facing answer. "
            "Keep only supported information already present in the draft. "
            "Do not add new facts. "
            "Finish with one short follow-up question.\n\n"
            f"Draft:\n{draft}"
        )
        response = self._call_llm(
            messages=[
                {
                    "role": "system",
                    "content": self._compose_system_prompt(
                        preferred_language=preferred_language,
                        suppress_thinking=True,
                    ),
                },
                {"role": "user", "content": repair_prompt},
            ],
            model=model_name,
            temperature=0.2,
            top_p=self.top_p,
            max_tokens=min(self.max_tokens + 120, 512),
        )
        repaired = self._extract_message_text(response)
        return repaired or draft

    # ─── Public API ────────────────────────────────────────────────────────────

    def reformulate_query(self, question: str, history: List[Dict[str, str]]) -> str:
        """Rewrite the latest user question into a standalone question."""
        if not history:
            return question

        system_prompt = (
            "You are an AI assistant tasked with reformulating user queries into standalone questions. "
            "Rewrite the latest user question so it can be understood without the history. "
            "Respond ONLY with the rewritten question."
        )
        messages: List[Any] = [{"role": "system", "content": system_prompt}]
        for turn in history:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": turn.get("content", "")})
        messages.append({"role": "user", "content": build_reformulation_prompt(question)})

        try:
            response = self._call_llm(
                messages=messages,
                model=self.model,
                temperature=0.1,
                top_p=self.top_p,
                max_tokens=100,
            )
            content = self._extract_message_text(response)
            return content or question
        except Exception as exc:
            logger.warning(
                "Query reformulation failed; using original question",
                extra={"error_type": exc.__class__.__name__},
            )
            return question

    def generate_stream(
        self,
        question: str,
        context: str,
        history: List[Dict[str, str]],
        preferred_language: Optional[str] = None,
    ):
        """Yield the complete answer as a single chunk (non-streaming fallback)."""
        full_response = self.generate(question, context, history, preferred_language)
        yield full_response

    def generate(
        self,
        question: str,
        context: str,
        history: List[Dict[str, str]],
        preferred_language: Optional[str] = None,
    ) -> str:
        """
        Generate a customer-facing answer via LiteLLM → NVIDIA NIM.

        Tries the primary model first, then each fallback model.
        Applies up to 2 internal repair passes per model for truncated or
        thinking-tagged responses.

        Raises:
            GeneratorRateLimitError: if the API returns HTTP 429.
            RuntimeError: if all models and repair passes are exhausted.
        """
        models_to_try = [self.model] + [
            m for m in self.fallback_models if m != self.model
        ]
        last_error: Exception | None = None

        for model_name in models_to_try:
            try:
                max_tokens = self.max_tokens
                suppress_thinking = False

                for attempt in range(2):
                    response = self._call_llm(
                        messages=self._build_completion_messages(
                            question=question,
                            context=context,
                            history=history,
                            preferred_language=preferred_language,
                            suppress_thinking=suppress_thinking,
                        ),
                        model=model_name,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=max_tokens,
                    )
                    raw_content = self._extract_raw_message_text(response)
                    content = self._clean_response_text(raw_content)
                    finish_reason = self._finish_reason(response)
                    has_thinking_tag = self._contains_thinking_tag(raw_content)

                    # Happy path: clean, complete answer.
                    if (
                        content
                        and finish_reason != "length"
                        and not has_thinking_tag
                        and not self._looks_incomplete(content)
                    ):
                        return content

                    # First pass hit thinking tags → retry with suppression.
                    if has_thinking_tag and not suppress_thinking:
                        suppress_thinking = True
                        if finish_reason == "length":
                            max_tokens = min(max_tokens * 2, 1024)
                        continue

                    # Still has thinking tags after suppression → repair call.
                    if has_thinking_tag:
                        repaired = self._repair_thinking_answer(
                            raw_content or content, preferred_language, model_name
                        )
                        repaired = self._clean_response_text(repaired)
                        if repaired:
                            return repaired
                        if content:
                            return content
                        continue

                    # Incomplete answer on first attempt → expand budget & retry.
                    if attempt == 0:
                        max_tokens = min(max_tokens * 2, 1024)
                        continue

                    # Still incomplete after retry → repair call.
                    if content:
                        return self._repair_incomplete_answer(
                            content, preferred_language, model_name
                        )

                raise ValueError(
                    f"LLM returned no final answer text for model {model_name}."
                )

            except GeneratorRateLimitError:
                # Do not swallow rate-limit errors — propagate immediately.
                raise
            except Exception as exc:
                last_error = exc
                logger.exception(
                    "LLM generation failed",
                    extra={"model": model_name, "error_type": exc.__class__.__name__},
                )
                # Continue to the next fallback model.

        if last_error is None:
            raise RuntimeError("LLM generation failed with an unknown error.")
        raise RuntimeError(f"LLM generation failed: {last_error}") from last_error
