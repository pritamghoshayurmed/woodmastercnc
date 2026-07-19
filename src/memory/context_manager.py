from __future__ import annotations

import re

from src.rag.types import RetrievedChunk


class ContextManager:
	def __init__(self, max_context_chars: int = 5000) -> None:
		self.max_context_chars = max_context_chars

	def build_context(
		self,
		retrieved_chunks: list[RetrievedChunk],
		history: list[dict[str, str]] | None = None,
	) -> tuple[str, list[str]]:
		used_sources: list[str] = []
		parts: list[str] = []
		current_length = 0

		history_block = self._build_history_block(history or [])
		if history_block:
			parts.append(history_block)
			current_length += len(history_block) + 2

		for idx, item in enumerate(retrieved_chunks, start=1):
			if item.chunk.metadata.get("is_placeholder"):
				continue

			source = item.chunk.source
			if source not in used_sources:
				used_sources.append(source)

			question = str(item.chunk.metadata.get("question", "")).strip()
			answer = str(item.chunk.metadata.get("answer", "")).strip()
			is_product_description = item.chunk.metadata.get("content_type") == "product_description"
			score = f"{item.score:.3f}"
			if is_product_description:
				snippet = f"[Product description {idx}] Source: {source} | Score: {score}\n{item.chunk.text.strip()}\n"
			else:
				snippet = (
					f"[FAQ {idx}] Source: {source} | Score: {score}\n"
					f"Question: {question}\n"
					f"Answer: {answer or item.chunk.text.strip()}\n"
				)
			if current_length + len(snippet) > self.max_context_chars:
				break

			parts.append(snippet)
			current_length += len(snippet)

		return "\n".join(parts), used_sources

	@staticmethod
	def _build_history_block(history: list[dict[str, str]]) -> str:
		recent_turns = history[-4:]
		if not recent_turns:
			return ""

		lines: list[str] = ["Recent conversation:"]
		for turn in recent_turns:
			role = "Assistant" if turn.get("role") == "assistant" else "User"
			content = re.sub(r"\s+", " ", turn.get("content", "")).strip()
			if not content:
				continue
			lines.append(f"{role}: {content}")

		return "\n".join(lines).strip()

