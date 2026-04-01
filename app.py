from __future__ import annotations

import logging

from dotenv import load_dotenv

from src.config import load_settings
from src.pipeline.rag_pipepline import RAGPipeline


logger = logging.getLogger(__name__)

_rag: RAGPipeline | None = None
_init_error: str | None = None


def _ensure_initialized() -> RAGPipeline:
	global _rag, _init_error
	if _rag is not None:
		return _rag

	load_dotenv()
	try:
		settings = load_settings()
		rag = RAGPipeline(settings)
		rag.initialize(force_rebuild=False)
		_rag = rag
		_init_error = None
		return rag
	except Exception as exc:
		_init_error = f"{exc.__class__.__name__}: {exc}"
		logger.exception("Lazy RAG initialization failed")
		raise RuntimeError("RAG system is not available right now.") from exc


def get_status() -> dict:
	if _rag is not None:
		return {"status": "ready"}
	if _init_error:
		return {"status": "not_ready", "detail": _init_error}
	return {"status": "not_initialized"}


def ask(question: str, session_id: str = "default") -> dict:
	rag = _ensure_initialized()
	return rag.query(question=question, session_id=session_id)

