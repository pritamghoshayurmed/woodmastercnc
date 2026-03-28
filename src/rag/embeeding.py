from __future__ import annotations

from typing import Literal

import requests


class GeminiEmbedder:
	def __init__(self, api_key: str, model: str = "models/gemini-embedding-001", timeout: int = 30) -> None:
		self.api_key = api_key
		self.model = model
		self.timeout = timeout

	def _embed_single(self, text: str, task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]) -> list[float]:
		url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:embedContent"
		payload = {
			"content": {"parts": [{"text": text}]},
			"taskType": task_type,
		}
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

	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		return [self._embed_single(text, "RETRIEVAL_DOCUMENT") for text in texts]

	def embed_query(self, text: str) -> list[float]:
		return self._embed_single(text, "RETRIEVAL_QUERY")

