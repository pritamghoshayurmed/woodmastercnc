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

	def generate_stream(self, question: str, context: str, history: list[dict[str, str]]):
		system_prompt = (
			"You are a helpful, conversational, and highly engaging CNC sales and support assistant for Woodmaster machines. "
			"Reply in the same language as the user, including English, Hindi, and Bengali. "
			"Keep your tone warm, friendly, and naturally curious—like a knowledgeable human sales engineer chatting with a client. "
			"Users may ask about model options, prices, warranty, training, EMI, delivery charges, after-sales service, spare parts, motor brand, power usage, and voltage compatibility. "
			"Use only the provided catalog context for factual details. If a detail is missing, say so honestly and suggest they speak with our human experts to find out. "
			"IMPORTANT: Keep your answers very concise and punchy to respond quickly. "
			"LEAD QUALIFICATION: Your ultimate goal is to gently guide the user towards qualifying themselves as a promising lead. "
			"In your responses, seamlessly integrate ONE conversational question to learn more about them. Examples of good questions to weave in: "
			"What specific material are you planning to cut? Where is your workshop located? What is your current production volume? Do you have an immediate timeline or budget in mind? "
			"Do not sound robotic or interrogate them. Make it feel like a natural part of helping them find the right machine."
		)

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

	def generate(self, question: str, context: str, history: list[dict[str, str]]) -> str:
		system_prompt = (
			"You are a helpful, conversational, and highly engaging CNC sales and support assistant for Woodmaster machines. "
			"Reply in the same language as the user, including English, Hindi, and Bengali. "
			"Keep your tone warm, friendly, and naturally curious—like a knowledgeable human sales engineer chatting with a client. "
			"Users may ask about model options, prices, warranty, training, EMI, delivery charges, after-sales service, spare parts, motor brand, power usage, and voltage compatibility. "
			"Use only the provided catalog context for factual details. If a detail is missing, say so honestly and suggest they speak with our human experts to find out. "
			"IMPORTANT: Keep your answers very concise and punchy to respond quickly. "
			"LEAD QUALIFICATION: Your ultimate goal is to gently guide the user towards qualifying themselves as a promising lead. "
			"In your responses, seamlessly integrate ONE conversational question to learn more about them. Examples of good questions to weave in: "
			"What specific material are you planning to cut? Where is your workshop located? What is your current production volume? Do you have an immediate timeline or budget in mind? "
			"Do not sound robotic or interrogate them. Make it feel like a natural part of helping them find the right machine."
		)

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

