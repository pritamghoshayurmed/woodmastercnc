import logging
<<<<<<< HEAD
import re
from typing import Any, Dict, List, Optional
=======
import time

from sarvamai import SarvamAI

class GenerationRateLimitError(RuntimeError):
	pass
>>>>>>> 0459f549af78c8063fbe9d00dd04b416dad88c04

from sarvamai import SarvamAI
from src.rag.prompt import build_answer_prompt, build_reformulation_prompt, build_system_prompt

logger = logging.getLogger(__name__)

# Maximum number of prior messages (user+assistant pairs) to send to Sarvam.
# Each pair = 2 messages. 3 pairs = 6 messages. Keeps context window manageable.
_MAX_HISTORY_MESSAGES = 6

<<<<<<< HEAD
class GeneratorRateLimitError(RuntimeError):
    pass


class SarvamSettingsError(RuntimeError):
    pass
=======

class SarvamGenerator:
	def __init__(
		self,
		api_key: str,
		model: str = "sarvam-30b",
		timeout: int = 25,
		max_retries: int = 1,
		backoff_seconds: float = 0.6,
		temperature: float = 0.2,
		top_p: float = 1.0,
		max_tokens: int = 1500,
	) -> None:
		self.client = SarvamAI(api_subscription_key=api_key)
		self.api_key = api_key
		self.model = model
		self.timeout = timeout
		self.max_retries = max_retries
		self.backoff_seconds = backoff_seconds
		self.temperature = temperature
		self.top_p = top_p
		self.max_tokens = max_tokens

	def _build_contents(
		self,
		history: list[dict[str, str]],
		prompt_text: str,
		prefix: str = "",
		max_history_messages: int = _MAX_HISTORY_MESSAGES,
	) -> list[dict[str, str]]:
		contents: list[dict[str, str]] = []
		# Cap history to prevent token-window overflow on multi-turn conversations.
		# Taking the tail ensures we always keep the most recent context.
		capped_history = history[-max_history_messages:] if history else []
		for turn in capped_history:
			role = "assistant" if turn.get("role") == "assistant" else "user"
			contents.append({"role": role, "content": turn.get("content", "")})

		contents.append(
			{
				"role": "user",
				"content": f"{prefix}{prompt_text}",
			}
		)
		return contents
>>>>>>> 0459f549af78c8063fbe9d00dd04b416dad88c04


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

<<<<<<< HEAD
    def _compose_system_prompt(self, preferred_language: Optional[str] = None) -> str:
        return build_system_prompt(preferred_language=preferred_language)

    @staticmethod
    def _clean_response_text(text: str | None) -> str:
        if not text:
            return ""

        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
=======
		contents = self._build_contents(history, question, "Reformulate this question to be standalone: ")
		messages = [{"role": "system", "content": system_prompt}] + contents

		try:
			response = self.client.chat.completions(
				model=self.model,
				messages=messages,
				temperature=0.1,
				top_p=1.0,
				max_tokens=500,
			)
			return (response.choices[0].message.content or question).strip()
		except Exception as exc:
			logger.warning("Query reformulation failed; using original question", extra={"error_type": exc.__class__.__name__})
			return question

	def _compose_system_prompt(self, preferred_language: str | None = None) -> str:
		base_prompt = (
			"You are an expert, highly persuasive sales and support assistant for Woodmaster CNC machines. "
			"Your primary goal is to provide exceptional customer service, handle typical customer inquiries naturally, convince them of our machine's superior quality (such as our core focus on high-precision and robust support), and qualify them as a strong lead. "
			"Never sound robotic. Sound like a friendly, knowledgeable human sales engineer speaking directly with a client. Be respectful, encouraging, and confident. "
			"Use the provided catalog context for exact specs, numbers, and facts. If a specific detail is missing from context, do not make it up; instead, assure them our sales experts will have the precise answer and seamlessly transition into a qualifying question to connect them. "
			"Users often ask about these 16 key topics. Here is how you should handle them conversationally and persuasively based on context: "
			"1. Models: Highlight the specific CNC models we have. "
			"2. Price: Provide the pricing if available, framing it as a great investment. "
			"3. Warranty: Mention our warranty period to build trust. "
			"4. Training: Explain our post-purchase training, its duration, and emphasize how easy it makes getting started. "
			"5. Service: Emphasize our dedicated after-sales service to assure them they won't face downtime. "
			"6. EMI: Confirm if EMI options are available so it's affordable for them. "
			"7. Functions: Outline exactly what materials and products our machines can cut/rout. "
			"8. Best Model: Help them choose by asking about their specific needs (size, volume). "
			"9. Accessories: List the items provided with the machine. "
			"10. Delivery: Clarify if delivery is free or chargeable, adding value where possible. "
			"11. Service Centers: Confirm our service center availability. "
			"12. Spare Parts: Reassure them that spare parts are easily available directly with us. "
			"13. Motto: Share the company's core motto to build brand prestige. "
			"15. Electric: Confirm it runs on electricity. "
			"16. 220V vs others: Clarify the phase/voltage requirements based on the specific model. "
			"IMPORTANT: Keep your answers very concise and punchy to respond quickly. "
			"CRITICAL LEAD QUALIFICATION: Every single response you give must end with exactly ONE natural, conversational question to learn more about their needs and move the sale forward. Good examples to weave in: "
			"- 'What specific material are you planning to cut with this machine?' "
			"- 'Where is your workshop or factory located?' "
			"- 'Are you starting a new business or upgrading an existing operation?' "
			"- 'What is your current production volume?' "
			"- 'Do you have an immediate timeline or budget in mind you want to share?' "
			"CRITICAL INSTRUCTION: You must provide a conversational response. DO NOT show your thinking process or use phrases like 'Let me think'. Just provide the response directly. "
			"CRITICAL LANGUAGE INSTRUCTION: If the user writes in Bengali or Hindi using the English script (transliterated, e.g., 'Bonglish' or 'Hinglish'), you MUST respond in that same style. For example, if asked 'apnader machine te ki EMI available ache?', answer in the same transliterated Bengali style like 'obossoi! amader machine te emi available ache'. Similarly, if asked in Hinglish, respond in Hinglish like 'kya aplog emi pe machine dete hai ? ' Answer in transliterated language like 'hmm , hamare paas EMI option available hai ' . Always match the user's script and language style."
		)
>>>>>>> 0459f549af78c8063fbe9d00dd04b416dad88c04

    @classmethod
    def _extract_message_text(cls, response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None
        return cls._clean_response_text(content)

<<<<<<< HEAD
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
        system_prompt = self._compose_system_prompt(preferred_language=preferred_language)
=======
		contents = self._build_contents(
			history, 
			f"User Question:\n{question}\n\nCatalog Knowledge Context:\n{context}\n\nGive a direct answer first, then 2-4 concise points if useful."
		)
		messages = [{"role": "system", "content": system_prompt}] + contents

		for attempt in range(self.max_retries + 1):
			try:
				response = self.client.chat.completions(
					model=self.model,
					messages=messages,
					temperature=self.temperature,
					top_p=self.top_p,
					max_tokens=self.max_tokens,
				)
				content = ""
				if response.choices and response.choices[0].message:
					content = (response.choices[0].message.content or "").strip()
				if content:
					yield content
					return
				# Empty content on stream — log and retry without history
				logger.error(
					"Sarvam returned EMPTY content on stream attempt %d/%d. "
					"messages_count=%d. Retrying without history.",
					attempt + 1, self.max_retries + 1, len(messages),
				)
				if attempt < self.max_retries:
					# Strip history, keep only system + current user message
					system_msgs = [m for m in messages if m.get("role") == "system"]
					user_msgs = [m for m in messages if m.get("role") == "user"][-1:]
					messages = system_msgs + user_msgs
					time.sleep(self.backoff_seconds)
					continue
				# Still empty after retry — raise so caller uses fallback
				raise RuntimeError("Sarvam stream returned empty content after retry.")
			except GenerationRateLimitError:
				raise
			except RuntimeError:
				raise
			except Exception as exc:
				error_text = str(exc)
				is_rate_limited = "429" in error_text

				if is_rate_limited and attempt >= self.max_retries:
					raise GenerationRateLimitError("Sarvam rate limit reached (HTTP 429).") from exc

				if is_rate_limited and attempt < self.max_retries:
					time.sleep(self.backoff_seconds * (2 ** attempt))
					continue

				logger.error(
					"Sarvam stream exception on attempt %d/%d: %s: %s",
					attempt + 1, self.max_retries + 1, exc.__class__.__name__, error_text,
				)
				raise RuntimeError(f"Sarvam generation request failed: {error_text}") from exc

		raise RuntimeError("Sarvam request failed after retries.")
>>>>>>> 0459f549af78c8063fbe9d00dd04b416dad88c04

        messages: List[Any] = [{"role": "system", "content": system_prompt}]
        for turn in history:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": turn.get("content", "")})

<<<<<<< HEAD
        prompt = build_answer_prompt(question=question, context=context)
        messages.append({"role": "user", "content": prompt})

        models_to_try = [self.model] + [model for model in self.fallback_models if model != self.model]
        last_error: Exception | None = None

        for model_name in models_to_try:
            try:
                response = self.client.chat.completions(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                content = self._extract_message_text(response)
                if content:
                    return content
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
=======
		contents = self._build_contents(
			history, 
			f"User Question:\n{question}\n\nCatalog Knowledge Context:\n{context}\n\nGive a direct answer first, then 2-4 concise points if useful."
		)
		messages = [{"role": "system", "content": system_prompt}] + contents

		for attempt in range(self.max_retries + 1):
			try:
				response = self.client.chat.completions(
					model=self.model,
					messages=messages,
					temperature=self.temperature,
					top_p=self.top_p,
					max_tokens=self.max_tokens,
				)
				content = ""
				if response.choices and response.choices[0].message:
					content = (response.choices[0].message.content or "").strip()
				if not content:
					# Empty response — log details and retry without history context
					logger.error(
						"Sarvam returned EMPTY content on attempt %d/%d. "
						"messages_count=%d. This causes the 'live generator' fallback.",
						attempt + 1, self.max_retries + 1, len(messages),
					)
					if attempt < self.max_retries:
						# Retry with minimal context: system prompt + current user message only
						system_msgs = [m for m in messages if m.get("role") == "system"]
						user_msgs = [m for m in messages if m.get("role") == "user"][-1:]
						messages = system_msgs + user_msgs
						logger.info(
							"Retrying without history. Reduced messages_count=%d",
							len(messages),
						)
						time.sleep(self.backoff_seconds)
						continue
					raise ValueError("Sarvam returned an empty response after retry.")
				logger.info(
					"Sarvam generate() OK on attempt %d. content_len=%d",
					attempt + 1, len(content),
				)
				return content
			except (ValueError, GenerationRateLimitError, RuntimeError):
				raise
			except Exception as exc:
				error_text = str(exc)
				is_rate_limited = "429" in error_text

				if is_rate_limited and attempt >= self.max_retries:
					raise GenerationRateLimitError("Sarvam rate limit reached (HTTP 429).") from exc

				if is_rate_limited and attempt < self.max_retries:
					time.sleep(self.backoff_seconds * (2 ** attempt))
					continue

				logger.error(
					"Sarvam generate() exception on attempt %d/%d: %s: %s",
					attempt + 1, self.max_retries + 1, exc.__class__.__name__, error_text,
				)
				raise RuntimeError(f"Sarvam generation request failed: {error_text}") from exc

		raise RuntimeError("Sarvam request failed after retries.")
>>>>>>> 0459f549af78c8063fbe9d00dd04b416dad88c04
