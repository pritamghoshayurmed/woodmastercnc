from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings
from src.memory.context_manager import ContextManager
from src.memory.memory_manager import ConversationMemoryManager
from src.rag.embedding import GeminiEmbedder
from src.rag.faiss_store import FaissVectorStore
from src.rag.generation import GeminiGenerator, GeminiRateLimitError
from src.rag.types import DocumentChunk, RAGResponse


class RAGPipeline:
	def __init__(self, settings: Settings) -> None:
		self.settings = settings
		self.embedder = GeminiEmbedder(
			api_key=settings.gemini_api_key,
			model=settings.gemini_embedding_model,
		)
		self.generator = GeminiGenerator(
			api_key=settings.gemini_api_key,
			model=settings.gemini_generation_model,
			timeout=settings.gemini_timeout_seconds,
			max_retries=settings.gemini_max_retries,
			backoff_seconds=settings.gemini_backoff_seconds,
			temperature=settings.gemini_temperature,
			top_p=settings.gemini_top_p,
			max_output_tokens=settings.gemini_max_output_tokens,
			thinking_budget=settings.gemini_thinking_budget,
		)
		self.vector_store = FaissVectorStore(
			index_path=settings.faiss_index_path,
			metadata_path=settings.faiss_metadata_path,
		)
		self.context_manager = ContextManager(max_context_chars=settings.max_context_chars)
		self.memory_manager = ConversationMemoryManager(
			max_turns=settings.memory_turns,
			session_dir=settings.session_store_dir,
			encryption_key=settings.session_encryption_key,
		)

	def initialize(self, force_rebuild: bool = False) -> None:
		self.settings.artifact_dir.mkdir(parents=True, exist_ok=True)

		if not force_rebuild and self.vector_store.exists():
			self.vector_store.load()
			return

		chunks = self._load_and_chunk_documents(self.settings.data_dir)
		if not chunks:
			raise ValueError("No chunks found in data directory.")

		embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
		self.vector_store.build(chunks=chunks, embeddings=embeddings)
		self.vector_store.save()

	def query(
		self,
		question: str,
		session_id: str = "default",
		preferred_language: str | None = None,
	) -> dict:
		self.memory_manager.add_user_message(session_id, question)
		
		# Context dilution fix: fetch smaller history for reformulate and generation
		history = self.memory_manager.get_recent_history(session_id)
		recent_history = history[-5:-1] if len(history) > 1 else []
		
		# Reformulate question based on previous turns
		standalone_question = self.generator.reformulate_query(question, history=recent_history)
		query_embedding = self.embedder.embed_query(standalone_question)

		retrieved = self.vector_store.search(query_embedding, top_k=self.settings.top_k)
		context, sources = self.context_manager.build_context(retrieved)

		try:
			answer = self.generator.generate(
				question=question,
				context=context,
				history=recent_history,
				preferred_language=preferred_language,
			)
		except GeminiRateLimitError:
			answer = self._build_fallback_answer(question=question, retrieved=retrieved, rate_limited=True)
		except Exception as exc:
			answer = self._build_fallback_answer(question=question, retrieved=retrieved, rate_limited=False, error_message=str(exc))
		self.memory_manager.add_assistant_message(session_id, answer)

		images = self._resolve_images(question, retrieved)
		response = RAGResponse(
			answer=answer,
			sources=sources,
			images=images,
			retrieved_chunks=retrieved,
		)
		return {
			"answer": response.answer,
			"sources": response.sources,
			"images": response.images,
			"retrieval": [
				{
					"score": round(item.score, 4),
					"chunk_id": item.chunk.chunk_id,
					"source": item.chunk.source,
					"metadata": item.chunk.metadata,
				}
				for item in response.retrieved_chunks
			],
		}

	def query_stream(
		self,
		question: str,
		session_id: str = "default",
		preferred_language: str | None = None,
	):
		self.memory_manager.add_user_message(session_id, question)
		
		# Context dilution fix: fetch smaller history for reformulate and generation
		history = self.memory_manager.get_recent_history(session_id)
		recent_history = history[-5:-1] if len(history) > 1 else []
		
		# Reformulate question based on previous turns
		standalone_question = self.generator.reformulate_query(question, history=recent_history)
		query_embedding = self.embedder.embed_query(standalone_question)

		retrieved = self.vector_store.search(query_embedding, top_k=self.settings.top_k)
		context, sources = self.context_manager.build_context(retrieved)

		images = self._resolve_images(question, retrieved)
		retrieval_data = [
			{
				"score": round(item.score, 4),
				"chunk_id": item.chunk.chunk_id,
				"source": item.chunk.source,
				"metadata": item.chunk.metadata,
			}
			for item in retrieved
		]

		# Yield metadata before streaming text
		yield {
			"type": "metadata",
			"sources": sources,
			"images": images,
			"retrieval": retrieval_data
		}

		full_answer = ""
		try:
			for text_chunk in self.generator.generate_stream(
				question=question,
				context=context,
				history=recent_history,
				preferred_language=preferred_language,
			):
				full_answer += text_chunk
				yield {"type": "chunk", "content": text_chunk}
		except GeminiRateLimitError:
			fallback = self._build_fallback_answer(question=question, retrieved=retrieved, rate_limited=True)
			full_answer = fallback
			yield {"type": "chunk", "content": fallback}
		except Exception as exc:
			fallback = self._build_fallback_answer(question=question, retrieved=retrieved, rate_limited=False, error_message=str(exc))
			full_answer = fallback
			yield {"type": "chunk", "content": fallback}

		self.memory_manager.add_assistant_message(session_id, full_answer)
		yield {"type": "done"}

	def _load_and_chunk_documents(self, data_dir: Path) -> list[DocumentChunk]:
		text_files = sorted(data_dir.glob("**/*.txt"))
		chunks: list[DocumentChunk] = []
		
		# Load image mapping
		image_map = {}
		mapping_file = data_dir / "image_mapping.json"
		if mapping_file.exists():
			with open(mapping_file, "r", encoding="utf-8") as f:
				image_map = json.load(f)

		for file_path in text_files:
			if "images" in file_path.parts:
				continue

			text = file_path.read_text(encoding="utf-8")
			source = str(file_path.relative_to(data_dir.parent)).replace("\\", "/")
			sections = self._split_sections(text)

			for section_idx, section in enumerate(sections):
				for chunk_idx, chunk_text in enumerate(self._chunk_text(section)):
					chunk_lower = chunk_text.lower()
					chunk_images = []
					for key, img_path in image_map.items():
						if key.lower() in chunk_lower and img_path not in chunk_images:
							chunk_images.append(img_path)

					metadata = {
						"section_index": section_idx,
						"chunk_index": chunk_idx,
						"product_tag": self._extract_product_tag(chunk_text),
						"images": chunk_images,
					}
					chunks.append(
						DocumentChunk(
							chunk_id=f"{file_path.stem}-{uuid.uuid4().hex[:10]}",
							text=chunk_text,
							source=source,
							metadata=metadata,
						)
					)
		return chunks

	def _split_sections(self, text: str) -> list[str]:
		cleaned = text.replace("\r\n", "\n")
		sections = re.split(r"\n={5,}.*?={5,}\n|\n\s*product\d+\s*\n", cleaned, flags=re.IGNORECASE)
		return [section.strip() for section in sections if section.strip()]

	def _chunk_text(self, text: str) -> list[str]:
		chunk_size = self.settings.chunk_size
		overlap = self.settings.chunk_overlap

		splitter = RecursiveCharacterTextSplitter(
			chunk_size=chunk_size,
			chunk_overlap=overlap,
		)
		return splitter.split_text(text)

	@staticmethod
	def _extract_product_tag(text: str) -> str | None:
		lowered = text.lower()
		match = re.search(r"product\s*(\d+)|product(\d+)", lowered)
		if match:
			group = match.group(1) or match.group(2)
			return f"product{group}"
		return None

	def _resolve_images(self, question: str, retrieved) -> list[str]:
		selected: list[str] = []
		product_tags: set[str] = set()
		
		for item in retrieved:
			chunk_images = item.chunk.metadata.get("images", [])
			product_tag = item.chunk.metadata.get("product_tag")
			if product_tag:
				product_tags.add(str(product_tag).lower())
			for img in chunk_images:
				if img not in selected:
					selected.append(img)

		# If no direct chunk image match, infer only from product IDs referenced by
		# retrieved chunks and explicit question product mentions.
		if not selected:
			image_map = {}
			mapping_file = self.settings.data_dir / "image_mapping.json"
			if mapping_file.exists():
				with open(mapping_file, "r", encoding="utf-8") as f:
					image_map = json.load(f)

			question_tags: set[str] = set()
			for match in re.finditer(r"product\s*(\d+)|product(\d+)", question.lower()):
				group = match.group(1) or match.group(2)
				if group:
					question_tags.add(f"product{group}")
			target_tags = product_tags.union(question_tags)

			for tag in target_tags:
				path = image_map.get(tag)
				if path and path not in selected:
					selected.append(path)

			q = question.lower()
			for stem, path in image_map.items():
				if stem.lower() in q and path not in selected:
					selected.append(path)

		return selected

	def _build_fallback_answer(
		self,
		question: str,
		retrieved,
		rate_limited: bool,
		error_message: str | None = None,
	) -> str:
		if not retrieved:
			if rate_limited:
				return (
					"I couldn't reach Gemini right now because of temporary traffic limits. "
					"Please try again in a few seconds."
				)
			return "I couldn't generate a full answer right now. Please try once more in a moment."

		snippets: list[str] = []
		for item in retrieved[:2]:
			text = item.chunk.text.strip().replace("\n", " ")
			snippets.append(text[:220])

		intro = (
			"The live model is busy, but here's what I found in your catalog:"
			if rate_limited
			else "I couldn't reach the live generator, but here's what I found in your catalog:"
		)

		combined = "\n".join(f"- {snippet}" for snippet in snippets)
		return f"{intro}\n{combined}"

