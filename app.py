from __future__ import annotations

from dotenv import load_dotenv

from src.config import load_settings
from src.pipeline.rag_pipepline import RAGPipeline


load_dotenv()
_settings = load_settings()
_rag = RAGPipeline(_settings)
_rag.initialize(force_rebuild=False)


def ask(question: str, session_id: str = "default") -> dict:
	return _rag.query(question=question, session_id=session_id)

