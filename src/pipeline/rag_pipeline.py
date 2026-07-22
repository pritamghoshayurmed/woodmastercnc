from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher

from src.config import Settings
from src.memory.context_manager import ContextManager
from src.memory.memory_manager import ConversationMemoryManager
from src.rag.generation import GeneratorRateLimitError, LiteLLMGenerator
from src.rag.types import DocumentChunk, RAGResponse, RetrievedChunk
from src.product_catalog import build_catalog, find_product, format_catalog, load_catalog

logger = logging.getLogger(__name__)

_active_pipeline: "RAGPipeline | None" = None


def set_active_pipeline(pipeline: "RAGPipeline | None") -> None:
	"""Register the live pipeline instance so dashboard edits (e.g. Q&A changes) can reload it."""
	global _active_pipeline
	_active_pipeline = pipeline


def get_active_pipeline() -> "RAGPipeline | None":
	return _active_pipeline


class RAGPipeline:
	def __init__(self, settings: Settings) -> None:
		self.settings = settings
		self.generator = LiteLLMGenerator(
			api_key=settings.nvidia_nim_api_key,
			model=settings.llm_model,
			fallback_model=settings.llm_fallback_model,
			temperature=settings.llm_temperature,
			top_p=settings.llm_top_p,
			max_tokens=settings.llm_max_output_tokens,
		)
		self.context_manager = ContextManager(max_context_chars=settings.max_context_chars)
		self.memory_manager = ConversationMemoryManager(
			max_turns=settings.memory_turns,
			session_dir=settings.session_store_dir,
			encryption_key=settings.session_encryption_key,
		)
		self.knowledge_chunks: list[DocumentChunk] = []
		self.product_catalog: list[dict[str, Any]] = []

	def initialize(self, force_rebuild: bool = False) -> None:
		self.settings.artifact_dir.mkdir(parents=True, exist_ok=True)
		catalog_path = self.settings.data_dir / "product_catalog.json"
		self.product_catalog = build_catalog(self.settings.data_dir / "productspecification.md", catalog_path) if force_rebuild or not catalog_path.exists() else load_catalog(catalog_path)
		self.knowledge_chunks = self._build_knowledge_chunks()
		if not self.knowledge_chunks:
			raise ValueError(f"No FAQ entries found in {self.settings.knowledge_base_path}.")
		set_active_pipeline(self)

	def _build_knowledge_chunks(self) -> list[DocumentChunk]:
		chunks = self._load_knowledge_base(self.settings.knowledge_base_path)
		chunks.extend(self._load_product_descriptions(self.settings.data_dir / "productdescription.md"))
		chunks.extend(self._load_product_catalog_chunks())
		return chunks

	def _load_product_catalog_chunks(self) -> list[DocumentChunk]:
		"""Expose product_catalog.json (full specs/toolbox/unique features, plus the
		product image) to general Q&A retrieval, not just the exact-name shortcut
		in `query()`/`query_stream()`."""
		chunks: list[DocumentChunk] = []
		for index, product in enumerate(self.product_catalog, start=1):
			chunks.append(
				DocumentChunk(
					chunk_id=f"catalog-{index}",
					text=format_catalog(product),
					source="data/product_catalog.json",
					metadata={
						"content_type": "product_catalog",
						"product_name": product.get("name", ""),
						"answer": product.get("description", ""),
						"images": [product["image"]] if product.get("image") else [],
						"is_placeholder": False,
					},
				)
			)
		return chunks

	def reload_knowledge_base(self) -> None:
		"""Re-read Q&A content (database if enabled, else knowledge.md) without a process restart."""
		self.knowledge_chunks = self._build_knowledge_chunks()

	def query(
		self,
		question: str,
		session_id: str = "default",
		preferred_language: str | None = None,
	) -> dict:
		self.memory_manager.add_user_message(session_id, question)
		product = find_product(question, self.product_catalog)
		if product:
			answer = format_catalog(product)
			self.memory_manager.add_assistant_message(session_id, answer)
			return {"answer": answer, "sources": ["data/product_catalog.json"], "images": [product["image"]] if product.get("image") else [], "catalog": product, "retrieval": []}

		recent_history = self.memory_manager.get_context_window(
			session_id=session_id,
			question=question,
			max_messages=min(6, max(2, self.settings.memory_turns)),
		)
		standalone_question = self.generator.reformulate_query(question, history=recent_history)
		retrieval_query = self._build_retrieval_query(question, standalone_question, recent_history)
		generation_question = self._build_generation_question(question, standalone_question, recent_history)
		retrieved = self._retrieve_relevant_chunks(retrieval_query)
		context, sources = self.context_manager.build_context(retrieved, history=recent_history)

		if self._should_return_unavailable_answer(retrieved, context):
			answer = self._build_unavailable_answer(question=question)
		else:
			try:
				answer = self.generator.generate(
					question=generation_question,
					context=context,
					history=recent_history,
					preferred_language=preferred_language,
				)
			except GeneratorRateLimitError:
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
		product = find_product(question, self.product_catalog)
		if product:
			answer = format_catalog(product)
			yield {
				"type": "metadata",
				"sources": ["data/product_catalog.json"],
				"images": [product["image"]] if product.get("image") else [],
				"catalog": product,
				"retrieval": [],
			}
			yield {"type": "chunk", "content": answer}
			self.memory_manager.add_assistant_message(session_id, answer)
			yield {"type": "done"}
			return

		recent_history = self.memory_manager.get_context_window(
			session_id=session_id,
			question=question,
			max_messages=min(6, max(2, self.settings.memory_turns)),
		)
		standalone_question = self.generator.reformulate_query(question, history=recent_history)
		retrieval_query = self._build_retrieval_query(question, standalone_question, recent_history)
		generation_question = self._build_generation_question(question, standalone_question, recent_history)
		retrieved = self._retrieve_relevant_chunks(retrieval_query)
		context, sources = self.context_manager.build_context(retrieved, history=recent_history)

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
		if self._should_return_unavailable_answer(retrieved, context):
			full_answer = self._build_unavailable_answer(question=question)
			yield {"type": "chunk", "content": full_answer}
		else:
			try:
				for text_chunk in self.generator.generate_stream(
					question=generation_question,
					context=context,
					history=recent_history,
					preferred_language=preferred_language,
				):
					full_answer += text_chunk
					yield {"type": "chunk", "content": text_chunk}
			except GeneratorRateLimitError:
				fallback = self._build_fallback_answer(question=question, retrieved=retrieved, rate_limited=True)
				full_answer = fallback
				yield {"type": "chunk", "content": fallback}
			except Exception as exc:
				fallback = self._build_fallback_answer(question=question, retrieved=retrieved, rate_limited=False, error_message=str(exc))
				full_answer = fallback
				yield {"type": "chunk", "content": fallback}

		self.memory_manager.add_assistant_message(session_id, full_answer)
		yield {"type": "done"}

	def _load_faq_entries_from_db(self) -> list[dict[str, Any]] | None:
		"""Return dashboard-managed Q&A entries, or None to fall back to knowledge.md."""
		from src.db.client import is_db_enabled

		if not is_db_enabled():
			return None
		try:
			from src.db import qna as db_qna

			rows = db_qna.get_enabled_entries()
		except Exception:
			logger.exception("Failed to load Q&A entries from the database; falling back to knowledge.md")
			return None
		if not rows:
			return None
		return [
			{
				"question": row["question"],
				"answer": self._normalize_markdown_text(row["answer"]),
				"is_placeholder": False,
			}
			for row in rows
		]

	def _load_knowledge_base(self, knowledge_path: Path) -> list[DocumentChunk]:
		entries = self._load_faq_entries_from_db()
		if entries is not None:
			source = "database:qna_entries"
		else:
			if not knowledge_path.exists():
				raise FileNotFoundError(f"Knowledge base file not found: {knowledge_path}")
			text = knowledge_path.read_text(encoding="utf-8")
			entries = self._parse_faq_entries(text)
			source = str(knowledge_path.relative_to(self.settings.data_dir.parent)).replace("\\", "/")

		chunks: list[DocumentChunk] = []

		image_map = {}
		mapping_file = self.settings.data_dir / "image_mapping.json"
		if mapping_file.exists():
			image_map = json.loads(mapping_file.read_text(encoding="utf-8"))

		for index, entry in enumerate(entries, start=1):
			chunk_text = f"Question: {entry['question']}\nAnswer: {entry['answer']}"
			chunk_lower = chunk_text.lower()
			chunk_images = [
				img_path
				for key, img_path in image_map.items()
				if key.lower() in chunk_lower
			]
			metadata = {
				"faq_index": index,
				"question": entry["question"],
				"answer": entry["answer"],
				"product_tag": self._extract_product_tag(chunk_text),
				"images": chunk_images,
				"is_placeholder": entry["is_placeholder"],
			}
			chunks.append(
				DocumentChunk(
					chunk_id=f"faq-{index}",
					text=chunk_text,
					source=source,
					metadata=metadata,
				)
			)
		return chunks

	def _load_product_descriptions(self, description_path: Path) -> list[DocumentChunk]:
		if not description_path.exists():
			raise FileNotFoundError(f"Product description file not found: {description_path}")
		text = description_path.read_text(encoding="utf-8").replace("\r\n", "\n")
		matches = re.finditer(r"product name\s*:\s*(.+?)\s*,?\s*\n?\s*description\s*:\s*(.*?)(?=\n\s*product name\s*:|\Z)", text, re.IGNORECASE | re.DOTALL)
		source = str(description_path.relative_to(self.settings.data_dir.parent)).replace("\\", "/")
		chunks: list[DocumentChunk] = []
		for index, match in enumerate(matches, start=1):
			name = re.sub(r"\s+", " ", match.group(1)).strip(" ,")
			description = re.sub(r"\s+", " ", match.group(2)).strip()
			chunks.append(DocumentChunk(chunk_id=f"product-description-{index}", text=f"Product: {name}\nDescription: {description}", source=source, metadata={"content_type": "product_description", "product_name": name, "answer": description, "is_placeholder": False}))
		return chunks

	@staticmethod
	def _parse_faq_entries(text: str) -> list[dict[str, Any]]:
		cleaned = text.replace("\r\n", "\n")
		blocks = re.split(r"^##\s+", cleaned, flags=re.MULTILINE)
		entries: list[dict[str, Any]] = []

		for block in blocks:
			block = block.strip()
			if not block:
				continue

			lines = [line.rstrip() for line in block.splitlines()]
			heading = lines[0].strip()
			body = "\n".join(lines[1:]).strip()
			question = re.sub(r"^Q\d+\.\s*", "", heading).strip()
			answer = re.sub(r"^\*\*Answer:\*\*\s*", "", body, flags=re.IGNORECASE).strip()
			answer = re.sub(r"\n---\s*$", "", answer).strip()
			answer = RAGPipeline._normalize_markdown_text(answer)
			if not question or not answer:
				continue

			entries.append(
				{
					"question": question,
					"answer": answer,
					"is_placeholder": "answer will be provided by the user" in answer.lower(),
				}
			)

		return entries

	def _retrieve_relevant_chunks(self, question: str) -> list[RetrievedChunk]:
		informative: list[RetrievedChunk] = []
		placeholder_matches: list[RetrievedChunk] = []
		for chunk in self.knowledge_chunks:
			score = self._score_chunk(question, chunk)
			if score <= 0:
				continue
			item = RetrievedChunk(chunk=chunk, score=score)
			if chunk.metadata.get("is_placeholder"):
				placeholder_matches.append(item)
			else:
				informative.append(item)

		informative.sort(key=lambda item: item.score, reverse=True)
		if informative:
			top_score = informative[0].score
			min_score = max(0.12, top_score * 0.45)
			filtered = [item for item in informative if item.score >= min_score]
			return (filtered or informative[:1])[: self.settings.top_k]

		placeholder_matches.sort(key=lambda item: item.score, reverse=True)
		if placeholder_matches:
			return placeholder_matches[:1]

		fallback = [chunk for chunk in self.knowledge_chunks if not chunk.metadata.get("is_placeholder")]
		return [RetrievedChunk(chunk=chunk, score=0.0) for chunk in fallback[: self.settings.top_k]]

	def _build_retrieval_query(
		self,
		question: str,
		standalone_question: str,
		history: list[dict[str, str]],
	) -> str:
		base_question = standalone_question.strip() or question.strip()
		if not self._is_context_dependent(question):
			return base_question

		last_assistant = ""
		last_user = ""
		for turn in reversed(history):
			role = turn.get("role")
			content = str(turn.get("content", "")).strip()
			if not content:
				continue
			if not last_assistant and role == "assistant":
				last_assistant = self._extract_last_question(content) or content
			elif not last_user and role == "user":
				last_user = content
			if last_assistant and last_user:
				break

		parts = [self._expand_query_aliases(base_question)]
		if last_assistant:
			parts.append(f"Previous assistant question: {self._expand_query_aliases(last_assistant)}")
		if last_user:
			parts.append(f"Previous customer context: {self._expand_query_aliases(last_user)}")
		return "\n".join(parts)

	def _build_generation_question(
		self,
		question: str,
		standalone_question: str,
		history: list[dict[str, str]],
	) -> str:
		base_question = standalone_question.strip() or question.strip()
		if not self._is_context_dependent(question):
			return base_question

		last_assistant = ""
		for turn in reversed(history):
			if turn.get("role") == "assistant":
				content = str(turn.get("content", "")).strip()
				if content:
					last_assistant = self._extract_last_question(content) or content
					break

		if not last_assistant:
			return base_question

		if re.fullmatch(r"\d+\s*(ta|pcs|pieces|per day|daily)?", question.strip().lower()):
			return (
				f"Customer shared expected production quantity: {question.strip()}.\n"
				f"This is a reply to the previous assistant question: {last_assistant}"
			)

		return (
			f"Customer follow-up: {question.strip()}\n"
			f"This is a reply to the previous assistant question: {last_assistant}"
		)

	@staticmethod
	def _is_context_dependent(question: str) -> bool:
		normalized = re.sub(r"\s+", " ", question.strip().lower())
		if not normalized:
			return False

		tokens = re.findall(r"[a-z0-9]+", normalized)
		if len(tokens) <= 4:
			return True

		dependent_patterns = [
			r"^\d+\s*(ta|pcs|pieces|per day|daily)?$",
			r"^(ha|hmm|ok|okay|accha|acha|achha|yes|no|ji|haan|na)$",
			r"^(mostly|mainly|muloto|primarily)\b",
			r"^(amra|amar|eta|oi|seita|seta|etar)\b",
		]
		return any(re.search(pattern, normalized) for pattern in dependent_patterns)

	@staticmethod
	def _extract_last_question(text: str) -> str:
		sentences = re.split(r"(?<=[?.!])\s+|\n+", text.strip())
		questions = [sentence.strip() for sentence in sentences if "?" in sentence]
		return questions[-1] if questions else ""

	@staticmethod
	def _expand_query_aliases(text: str) -> str:
		normalized = text.strip()
		lowered = normalized.lower()
		extra_terms: list[str] = []

		alias_groups = [
			(
				{
					"kon machine", "which machine", "kon model", "best model", "recommend",
					"newa jai", "nibo", "bhabchilam", "macine", "macines", "mishin", "mishines",
					"machine", "machines", "model", "models", "ki ki", "kon kon", "মেশিন", "মডেল",
					"টাইপ", "আছে", "কি কি", "কোন কোন", "নেব", "কিনব", "list", "available"
				},
				"recommended model machine choice woodworking standard models"
			),
			(
				{
					"dam", "price", "koto", "cost", "quotation", "budget", "taka", "rupee",
					"দাম", "কত", "টাকা", "মূল্য"
				},
				"price quotation machine cost budget"
			),
			(
				{
					"emi", "finance", "loan", "bank", "kisti", "কিস্তি", "লোন", "ব্যাংক"
				},
				"emi finance bank funding"
			),
			(
				{
					"training", "shikhiye", "install", "installation", "support", "shikha",
					"ট্রেনিং", "শেখানো", "ইন্সটল", "সাপোর্ট"
				},
				"training installation support"
			),
			(
				{
					"chair", "furniture", "wood", "kath", "carving", "design", "material",
					"board", "sheet", "plywood", "mdf", "acrylic", "কাঠ", "ডিজাইন", "বোর্ড", "শীট"
				},
				"woodworking chair furniture carving materials processed"
			),
			(
				{
					"warranty", "guarantee", "ওয়ারেন্টি", "গ্যারান্টি", "ওয়ারেন্টি"
				},
				"warranty service support"
			),
		]

		for aliases, expansion in alias_groups:
			if any(alias in lowered for alias in aliases):
				extra_terms.append(expansion)

		if not extra_terms:
			return normalized
		return f"{normalized}\nRelated intent: {'; '.join(extra_terms)}"

	def _score_chunk(self, question: str, chunk: DocumentChunk) -> float:
		query_text = self._normalize_text(question)
		chunk_text = self._normalize_text(chunk.text)
		if not query_text or not chunk_text:
			return 0.0

		query_tokens = self._tokenize(query_text)
		chunk_tokens = self._tokenize(chunk_text)
		if not query_tokens or not chunk_tokens:
			return 0.0

		overlap = len(set(query_tokens) & set(chunk_tokens)) / len(set(query_tokens))
		phrase_ratio = SequenceMatcher(None, query_text, self._normalize_text(chunk.metadata.get("question", ""))).ratio()
		number_bonus = 0.2 if set(re.findall(r"\d+", query_text)) & set(re.findall(r"\d+", chunk_text)) else 0.0
		placeholder_penalty = 0.35 if chunk.metadata.get("is_placeholder") else 0.0
		category_bonus = self._category_boost(query_text, chunk)
		return max(0.0, overlap * 0.7 + phrase_ratio * 0.3 + number_bonus + category_bonus - placeholder_penalty)

	@staticmethod
	def _category_boost(query_text: str, chunk: DocumentChunk) -> float:
		chunk_question = str(chunk.metadata.get("question", "")).lower()
		boost = 0.0

		intent_groups = [
			(
				{
					"kon machine", "which machine", "kon model", "best model", "recommend",
					"newa jai", "nibo", "bhabchilam", "macine", "macines", "mishin", "mishines",
					"machine", "machines", "model", "models", "ki ki", "kon kon", "মেশিন", "মডেল",
					"টাইপ", "আছে", "কি কি", "কোন কোন", "নেব", "কিনব", "list", "available"
				},
				{"model"},
				0.28,
			),
			(
				{
					"dam", "price", "koto", "cost", "quotation", "budget", "taka", "rupee",
					"দাম", "কত", "টাকা", "মূল্য"
				},
				{"price"},
				0.28,
			),
			(
				{
					"emi", "finance", "loan", "bank", "kisti", "কিস্তি", "লোন", "ব্যাংক"
				},
				{"emi", "finance"},
				0.28,
			),
			(
				{
					"training", "shikhiye", "install", "installation", "support", "shikha",
					"ট্রেনিং", "শেখানো", "ইন্সটল", "সাপোর্ট"
				},
				{"training"},
				0.22,
			),
			(
				{
					"warranty", "guarantee", "ওয়ারেন্টি", "গ্যারান্টি", "ওয়ারেন্টি"
				},
				{"warranty"},
				0.22,
			),
			(
				{
					"chair", "furniture", "wood", "kath", "carving", "design", "material",
					"board", "sheet", "plywood", "mdf", "acrylic", "কাঠ", "ডিজাইন", "বোর্ড", "শীট"
				},
				{"materials", "processed"},
				0.18,
			),
		]

		for query_aliases, chunk_markers, amount in intent_groups:
			if any(alias in query_text for alias in query_aliases) and any(marker in chunk_question for marker in chunk_markers):
				boost += amount

		return boost

	@staticmethod
	def _normalize_text(text: str) -> str:
		return re.sub(r"\s+", " ", text.strip().lower())

	@staticmethod
	def _normalize_markdown_text(text: str) -> str:
		text = re.sub(r"^\*+\s*", "- ", text, flags=re.MULTILINE)
		return re.sub(r"\n{3,}", "\n\n", text).strip()

	@staticmethod
	def _tokenize(text: str) -> list[str]:
		stopwords = {
			"a", "an", "and", "are", "can", "do", "for", "from", "how", "i", "in",
			"is", "of", "on", "or", "the", "to", "what", "which", "with", "you",
			"your",
		}
		return [
			token
			for token in re.findall(r"[a-z0-9]+", text.lower())
			if token not in stopwords and len(token) > 1
		]

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

		# Only send images from chunks close to the strongest image-bearing match.
		# Retrieval keeps several loosely-related chunks for text context (e.g.
		# sibling machine models), but attaching every one of their images would
		# spam several product photos for a question about a single model.
		image_candidates = [item for item in retrieved if item.chunk.metadata.get("images")]
		if image_candidates:
			top_image_score = max(item.score for item in image_candidates)
			image_threshold = max(top_image_score * 0.85, top_image_score - 0.05)
			for item in image_candidates:
				if item.score < image_threshold:
					continue
				for img in item.chunk.metadata["images"]:
					if img not in selected:
						selected.append(img)

		for item in retrieved:
			product_tag = item.chunk.metadata.get("product_tag")
			if product_tag:
				product_tags.add(str(product_tag).lower())

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
					"I couldn't reach the live model right now because of temporary traffic limits. "
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

	@staticmethod
	def _should_return_unavailable_answer(retrieved: list[RetrievedChunk], context: str) -> bool:
		if context.strip():
			return False
		return bool(retrieved) and all(item.chunk.metadata.get("is_placeholder") for item in retrieved)

	@staticmethod
	def _build_unavailable_answer(question: str) -> str:
		return (
			"I couldn't find a confirmed answer for that in the current FAQ file yet. "
			"If you want, I can help you add this answer to `data/knowledge.md` once you confirm the exact details."
		)
