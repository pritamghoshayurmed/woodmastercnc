from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_embedding_model: str
    gemini_generation_model: str
    gemini_timeout_seconds: int
    gemini_max_retries: int
    gemini_backoff_seconds: float
    gemini_temperature: float
    gemini_top_p: float
    gemini_max_output_tokens: int
    gemini_thinking_budget: int
    data_dir: Path
    image_dir: Path
    artifact_dir: Path
    session_store_dir: Path
    faiss_index_path: Path
    faiss_metadata_path: Path
    session_encryption_key: str | None
    chunk_size: int
    chunk_overlap: int
    top_k: int
    memory_turns: int
    max_context_chars: int


def load_settings() -> Settings:
    base_dir = Path(__file__).resolve().parent.parent
    artifact_dir = base_dir / "artifacts"

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is missing in environment.")

    return Settings(
        gemini_api_key=gemini_api_key,
        gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        gemini_generation_model=os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash"),
        gemini_timeout_seconds=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "25")),
        gemini_max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "1")),
        gemini_backoff_seconds=float(os.getenv("GEMINI_BACKOFF_SECONDS", "0.6")),
        gemini_temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.35")),
        gemini_top_p=float(os.getenv("GEMINI_TOP_P", "0.9")),
        gemini_max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "600")),
        gemini_thinking_budget=int(os.getenv("GEMINI_THINKING_BUDGET", "0")),
        data_dir=base_dir / "data",
        image_dir=base_dir / "data" / "images",
        artifact_dir=artifact_dir,
        session_store_dir=artifact_dir / "sessions_private",
        faiss_index_path=artifact_dir / "faiss.index",
        faiss_metadata_path=artifact_dir / "faiss_meta.json",
        session_encryption_key=(os.getenv("SESSION_ENCRYPTION_KEY", "").strip() or gemini_api_key),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "700")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        memory_turns=int(os.getenv("RAG_MEMORY_TURNS", "8")),
        max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "5000")),
    )
