from __future__ import annotations

from src.rag.types import RetrievedChunk


class ContextManager:
	def __init__(self, max_context_chars: int = 5000) -> None:
		self.max_context_chars = max_context_chars

	def build_context(self, retrieved_chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
		used_sources: list[str] = []
		parts: list[str] = []
		current_length = 0

		for idx, item in enumerate(retrieved_chunks, start=1):
			source = item.chunk.source
			if source not in used_sources:
				used_sources.append(source)

			snippet = f"[{idx}] Source: {source}\n{item.chunk.text.strip()}\n"
			if current_length + len(snippet) > self.max_context_chars:
				break

			parts.append(snippet)
			current_length += len(snippet)

		return "\n".join(parts), used_sources

