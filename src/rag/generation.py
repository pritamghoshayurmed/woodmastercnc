from __future__ import annotations

import logging
import time

from google import genai
from google.genai import types


class GeminiRateLimitError(RuntimeError):
	pass


logger = logging.getLogger(__name__)


class GeminiGenerator:
	def __init__(
		self,
		api_key: str,
		model: str,
		timeout: int = 25,
		max_retries: int = 1,
		backoff_seconds: float = 0.6,
		temperature: float = 0.35,
		top_p: float = 0.9,
		max_output_tokens: int = 280,
		thinking_budget: int = 0,
	) -> None:
		self.client = genai.Client(api_key=api_key)
		self.api_key = api_key
		self.model = model
		self.timeout = timeout
		self.max_retries = max_retries
		self.backoff_seconds = backoff_seconds
		self.temperature = temperature
		self.top_p = top_p
		self.max_output_tokens = max_output_tokens
		self.thinking_budget = thinking_budget

	def reformulate_query(self, question: str, history: list[dict[str, str]]) -> str:
		if not history:
			return question

		system_prompt = (
			"You are an AI assistant tasked with reformulating user queries into standalone questions. "
			"Given the chat history, rewrite the latest user question so it can be understood without the history. "
			"Respond ONLY with the rewritten question. If it is already standalone, return it as is."
		)

		contents: list[types.Content] = []
		for turn in history:
			role = "model" if turn.get("role") == "assistant" else "user"
			contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn.get("content", ""))]))

		contents.append(
			types.Content(
				role="user",
				parts=[types.Part.from_text(text=f"Reformulate this question to be standalone: {question}")],
			)
		)

		config = types.GenerateContentConfig(
			system_instruction=system_prompt,
			temperature=0.1,
			max_output_tokens=100,
		)
		try:
			response = self.client.models.generate_content(
				model=self.model,
				contents=contents,
				config=config,
			)
			return (response.text or question).strip()
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
			"14. Origin (China etc.): Confidently state the origin (e.g., proudly manufactured/assembled, not just standard Chinese imports) as per catalog facts. "
			"15. Electric: Confirm it runs on electricity. "
			"16. 220V vs others: Clarify the phase/voltage requirements based on the specific model. "
			"IMPORTANT: Keep your answers very concise and punchy to respond quickly. "
			"CRITICAL LEAD QUALIFICATION: Every single response you give must end with exactly ONE natural, conversational question to learn more about their needs and move the sale forward. Good examples to weave in: "
			"- 'What specific material are you planning to cut with this machine?' "
			"- 'Where is your workshop or factory located?' "
			"- 'Are you starting a new business or upgrading an existing operation?' "
			"- 'What is your current production volume?' "
			"- 'Do you have an immediate timeline or budget in mind you want to share?' "
		)

		if preferred_language:
			return (
				f"{base_prompt} "
				f"For this entire conversation, unconditionally respond in {preferred_language}. "
				f"Ensure the grammar, vocabulary, and tone in {preferred_language} are completely natural and idiomatic to a native speaker."
			)
		return base_prompt

	def generate_stream(
		self,
		question: str,
		context: str,
		history: list[dict[str, str]],
		preferred_language: str | None = None,
	):
		system_prompt = self._compose_system_prompt(preferred_language=preferred_language)

		contents: list[types.Content] = []
		for turn in history:
			role = "model" if turn.get("role") == "assistant" else "user"
			contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn.get("content", ""))]))

		contents.append(
			types.Content(
				role="user",
				parts=[
					types.Part.from_text(
						text=(
							f"User Question:\n{question}\n\n"
							f"Catalog Knowledge Context:\n{context}\n\n"
							"Give a direct answer first, then 2-4 concise points if useful."
						)
					)
				],
			)
		)

		config = types.GenerateContentConfig(
			system_instruction=system_prompt,
			temperature=self.temperature,
			top_p=self.top_p,
			max_output_tokens=self.max_output_tokens,
			thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
		)

		for attempt in range(self.max_retries + 1):
			try:
				response_stream = self.client.models.generate_content_stream(
					model=self.model,
					contents=contents,
					config=config,
				)
				for chunk in response_stream:
					if chunk.text:
						yield chunk.text
				return
			except Exception as exc:
				error_text = str(exc)
				is_rate_limited = "429" in error_text or "RESOURCE_EXHAUSTED" in error_text
				is_retriable_server = any(code in error_text for code in ["500", "502", "503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED"])

				if is_rate_limited and attempt >= self.max_retries:
					raise GeminiRateLimitError("Gemini rate limit reached (HTTP 429 / RESOURCE_EXHAUSTED).") from exc

				if (is_rate_limited or is_retriable_server) and attempt < self.max_retries:
					time.sleep(self.backoff_seconds * (2 ** attempt))
					continue

				raise RuntimeError(f"Gemini generation request failed: {error_text}") from exc

		raise RuntimeError("Gemini request failed after retries.")

	def generate(
		self,
		question: str,
		context: str,
		history: list[dict[str, str]],
		preferred_language: str | None = None,
	) -> str:
		system_prompt = self._compose_system_prompt(preferred_language=preferred_language)

		contents: list[types.Content] = []
		for turn in history:
			role = "model" if turn.get("role") == "assistant" else "user"
			contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn.get("content", ""))]))

		contents.append(
			types.Content(
				role="user",
				parts=[
					types.Part.from_text(
						text=(
							f"User Question:\n{question}\n\n"
							f"Catalog Knowledge Context:\n{context}\n\n"
							"Give a direct answer first, then 2-4 concise points if useful."
						)
					)
				],
			)
		)

		config = types.GenerateContentConfig(
			system_instruction=system_prompt,
			temperature=self.temperature,
			top_p=self.top_p,
			max_output_tokens=self.max_output_tokens,
			thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
		)

		for attempt in range(self.max_retries + 1):
			try:
				response = self.client.models.generate_content(
					model=self.model,
					contents=contents,
					config=config,
				)
				content = (response.text or "").strip()
				if not content:
					raise ValueError("Gemini returned an empty response.")
				return content
			except Exception as exc:
				error_text = str(exc)
				is_rate_limited = "429" in error_text or "RESOURCE_EXHAUSTED" in error_text
				is_retriable_server = any(code in error_text for code in ["500", "502", "503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED"])

				if is_rate_limited and attempt >= self.max_retries:
					raise GeminiRateLimitError("Gemini rate limit reached (HTTP 429 / RESOURCE_EXHAUSTED).") from exc

				if (is_rate_limited or is_retriable_server) and attempt < self.max_retries:
					time.sleep(self.backoff_seconds * (2 ** attempt))
					continue

				raise RuntimeError(f"Gemini generation request failed: {error_text}") from exc

		raise RuntimeError("Gemini request failed after retries.")

