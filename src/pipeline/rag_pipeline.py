from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher

from src.config import Settings
from src.memory.context_manager import ContextManager
from src.memory.memory_manager import ConversationMemoryManager
from src.rag.generation import GeneratorRateLimitError, SarvamGenerator
from src.rag.types import DocumentChunk, RAGResponse, RetrievedChunk


class RAGPipeline:
	def __init__(self, settings: Settings) -> None:
		self.settings = settings
		self.generator = SarvamGenerator(
			api_key=settings.sarvam_api_key,
			model=settings.sarvam_generation_model,
			temperature=settings.sarvam_temperature,
			top_p=settings.sarvam_top_p,
			max_tokens=settings.sarvam_max_output_tokens,
		)
		self.context_manager = ContextManager(max_context_chars=settings.max_context_chars)
		self.memory_manager = ConversationMemoryManager(
			max_turns=settings.memory_turns,
			session_dir=settings.session_store_dir,
			encryption_key=settings.session_encryption_key,
		)
		self.knowledge_chunks: list[DocumentChunk] = []

	def initialize(self, force_rebuild: bool = False) -> None:
		self.settings.artifact_dir.mkdir(parents=True, exist_ok=True)
		self.knowledge_chunks = self._load_knowledge_base(self.settings.knowledge_base_path)
		if not self.knowledge_chunks:
			raise ValueError(f"No FAQ entries found in {self.settings.knowledge_base_path}.")

	def query(
		self,
		question: str,
		session_id: str = "default",
		preferred_language: str | None = None,
	) -> dict:
		self.memory_manager.add_user_message(session_id, question)

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

	def _load_knowledge_base(self, knowledge_path: Path) -> list[DocumentChunk]:
		if not knowledge_path.exists():
			raise FileNotFoundError(f"Knowledge base file not found: {knowledge_path}")

		text = knowledge_path.read_text(encoding="utf-8")
		entries = self._parse_faq_entries(text)
		chunks: list[DocumentChunk] = []

		image_map = {}
		mapping_file = self.settings.data_dir / "image_mapping.json"
		if mapping_file.exists():
			image_map = json.loads(mapping_file.read_text(encoding="utf-8"))

		source = str(knowledge_path.relative_to(self.settings.data_dir.parent)).replace("\\", "/")
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

	def _parse_faq_entries(self, text: str) -> list[dict[str, Any]]:
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
			answer = self._normalize_markdown_text(answer)
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
			({"kon machine", "which machine", "kon model", "best model", "recommend", "newa jai", "nibo", "bhabchilam"}, "recommended model machine choice woodworking"),
			({"dam", "price", "koto", "cost", "quotation"}, "price quotation machine cost"),
			({"emi", "finance", "loan", "bank"}, "emi finance bank funding"),
			({"training", "shikhiye", "install", "support"}, "training installation support"),
			({"chair", "furniture", "wood", "kath", "carving", "design"}, "woodworking chair furniture carving"),
			({"warranty", "guarantee"}, "warranty service support"),
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
				{"kon machine", "which machine", "kon model", "best model", "recommend", "newa jai", "nibo", "bhabchilam"},
				{"recommend", "model"},
				0.28,
			),
			(
				{"price", "dam", "koto", "quotation", "cost"},
				{"price"},
				0.28,
			),
			(
				{"emi", "finance", "loan", "bank"},
				{"emi", "finance"},
				0.28,
			),
			(
				{"training", "install", "support"},
				{"training"},
				0.22,
			),
			(
				{"warranty", "guarantee"},
				{"warranty"},
				0.22,
			),
			(
				{"material", "wood", "kath", "chair", "furniture", "acrylic", "plywood"},
				{"materials", "processed"},
				0.18,
			),
		]

		for query_aliases, chunk_markers, amount in intent_groups:
			if any(alias in query_text for alias in query_aliases) and all(marker in chunk_question for marker in chunk_markers):
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

