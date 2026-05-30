import logging
import re
from typing import Any, Dict, List, Optional

from sarvamai import SarvamAI
from src.rag.prompt import build_answer_prompt, build_reformulation_prompt, build_system_prompt

logger = logging.getLogger(__name__)

# Maximum number of prior messages (user+assistant pairs) to send to Sarvam.
# Each pair = 2 messages. 3 pairs = 6 messages. Keeps context window manageable.
_MAX_HISTORY_MESSAGES = 6

class GeneratorRateLimitError(RuntimeError):
    pass


class SarvamSettingsError(RuntimeError):
    pass


class SarvamGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "sarvam-m",
        temperature: float = 0.5,
        top_p: float = 1.0,
        max_tokens: int = 2000,
    ) -> None:
        if not api_key:
            raise SarvamSettingsError("Sarvam API key is required.")
        self.client = SarvamAI(api_subscription_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.fallback_models = ["sarvam-m"]

    def _compose_system_prompt(
        self,
        preferred_language: Optional[str] = None,
        suppress_thinking: bool = False,
    ) -> str:
        return build_system_prompt(
            preferred_language=preferred_language,
            suppress_thinking=suppress_thinking,
        )

    @staticmethod
    def _contains_thinking_tag(text: str | None) -> bool:
        if not text:
            return False

        return bool(re.search(r"</?\s*think\b[^>]*>", text, flags=re.IGNORECASE))

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
        cleaned = re.sub(r"</?\s*think\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    @staticmethod
    def _clean_response_text(text: str | None) -> str:
        if not text:
            return ""

        cleaned = SarvamGenerator._strip_thinking_tags(text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

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

        messages.append({"role": "user", "content": build_answer_prompt(question=question, context=context)})
        return messages

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
        response = self.client.chat.completions(
            model=model_name,
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
            temperature=0.2,
            top_p=self.top_p,
            max_tokens=min(self.max_tokens + 120, 360),
        )
        repaired = self._extract_message_text(response)
        return repaired or draft

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

    @staticmethod
    def _looks_incomplete(text: str) -> bool:
        if not text:
            return True

        stripped = text.rstrip()
        if len(stripped) < 40:
            return False

        return stripped[-1] not in {".", "!", "?", '"', "'", ")", "]"}

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
        response = self.client.chat.completions(
            model=model_name,
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
            temperature=0.2,
            top_p=self.top_p,
            max_tokens=min(self.max_tokens + 120, 360),
        )
        repaired = self._extract_message_text(response)
        return repaired or draft

    def reformulate_query(self, question: str, history: List[Dict[str, str]]) -> str:
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
            response = self.client.chat.completions(
                model=self.model,
                messages=messages,
                temperature=0.1,
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
        full_response = self.generate(question, context, history, preferred_language)
        yield full_response

    def generate(
        self,
        question: str,
        context: str,
        history: List[Dict[str, str]],
        preferred_language: Optional[str] = None,
    ) -> str:
        fallback_models = getattr(self, "fallback_models", ["sarvam-m"])
        models_to_try = [self.model] + [model for model in fallback_models if model != self.model]
        last_error: Exception | None = None

        for model_name in models_to_try:
            try:
                max_tokens = self.max_tokens
                suppress_thinking = False
                last_content = ""
                for attempt in range(2):
                    response = self.client.chat.completions(
                        model=model_name,
                        messages=self._build_completion_messages(
                            question=question,
                            context=context,
                            history=history,
                            preferred_language=preferred_language,
                            suppress_thinking=suppress_thinking,
                        ),
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=max_tokens,
                    )
                    raw_content = self._extract_raw_message_text(response)
                    content = self._clean_response_text(raw_content)
                    last_content = content
                    finish_reason = self._finish_reason(response)
                    has_thinking_tag = self._contains_thinking_tag(raw_content)

                    if content and finish_reason != "length" and not has_thinking_tag and not self._looks_incomplete(content):
                        return content

                    if has_thinking_tag and not suppress_thinking:
                        suppress_thinking = True
                        if finish_reason == "length":
                            max_tokens = min(max_tokens * 2, 480)
                        continue

                    if has_thinking_tag:
                        repaired = self._repair_thinking_answer(raw_content or content, preferred_language, model_name)
                        repaired = self._clean_response_text(repaired)
                        if repaired:
                            return repaired
                        if content:
                            return content
                        continue

                    if attempt == 0:
                        max_tokens = min(max_tokens * 2, 480)
                        continue

                    if content:
                        return self._repair_incomplete_answer(content, preferred_language, model_name)

                raise ValueError(f"Sarvam returned no final answer text for model {model_name}.")
            except Exception as exc:
                last_error = exc
                error_text = str(exc)
                logger.exception(
                    "Sarvam generation failed",
                    extra={"model": model_name, "error_type": exc.__class__.__name__},
                )
                if "429" in error_text:
                    raise GeneratorRateLimitError("Rate limit reached.") from exc

        if last_error is None:
            raise RuntimeError("Sarvam generation request failed with an unknown error.")
        raise RuntimeError(f"Sarvam generation request failed: {last_error}") from last_error
