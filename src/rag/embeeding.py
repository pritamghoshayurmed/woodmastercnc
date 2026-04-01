from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import requests


class GeminiEmbedder:
	def __init__(
		self,
		api_key: str,
		model: str = "models/gemini-embedding-001",
		timeout: int = 30,
		max_retries: int = 2,
		backoff_seconds: float = 0.6,
		document_workers: int = 4,
	) -> None:
		self.api_key = api_key
		self.model = model
		self.timeout = timeout
		self.max_retries = max_retries
		self.backoff_seconds = backoff_seconds
		self.document_workers = max(1, document_workers)

	def _embed_single(self, text: str, task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]) -> list[float]:
		url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:embedContent"
		payload = {
			"content": {"parts": [{"text": text}]},
			"taskType": task_type,
		}

		for attempt in range(self.max_retries + 1):
			try:
				response = requests.post(
					url,
					params={"key": self.api_key},
					json=payload,
					timeout=self.timeout,
				)
				response.raise_for_status()
				body = response.json()

				values = body.get("embedding", {}).get("values")
				if not values:
					raise ValueError(f"Empty embedding response from Gemini model {self.model}")
				return values
			except requests.RequestException as exc:
				error_text = str(exc)
				is_rate_limited = "429" in error_text or "RESOURCE_EXHAUSTED" in error_text
				is_retriable_server = any(code in error_text for code in ["500", "502", "503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED"])
				if (is_rate_limited or is_retriable_server) and attempt < self.max_retries:
					time.sleep(self.backoff_seconds * (2 ** attempt))
					continue
				raise RuntimeError(f"Embedding request failed: {error_text}") from exc

		raise RuntimeError("Embedding request failed after retries.")

	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		if not texts:
			return []
		with ThreadPoolExecutor(max_workers=self.document_workers) as executor:
			return list(executor.map(lambda text: self._embed_single(text, "RETRIEVAL_DOCUMENT"), texts))

	def embed_query(self, text: str) -> list[float]:
		return self._embed_single(text, "RETRIEVAL_QUERY")

