from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
<<<<<<< HEAD
    sarvam_api_key: str
    sarvam_generation_model: str
    sarvam_temperature: float
    sarvam_top_p: float
    sarvam_max_output_tokens: int
=======
    gemini_api_key: str
    sarvam_api_key: str
    gemini_embedding_model: str
    gemini_generation_model: str
    gemini_timeout_seconds: int
    gemini_max_retries: int
    gemini_backoff_seconds: float
    gemini_temperature: float
    gemini_top_p: float
    gemini_max_output_tokens: int
    gemini_thinking_budget: int
>>>>>>> 0459f549af78c8063fbe9d00dd04b416dad88c04
    data_dir: Path
    image_dir: Path
    knowledge_base_path: Path
    artifact_dir: Path
    session_store_dir: Path
    session_encryption_key: str | None
    top_k: int
    memory_turns: int
    max_context_chars: int


def load_settings() -> Settings:
    base_dir = Path(__file__).resolve().parent.parent
    artifact_dir = base_dir / "artifacts"

    sarvam_api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not sarvam_api_key:
        raise ValueError("SARVAM_API_KEY is missing in environment.")

    sarvam_api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not sarvam_api_key:
        raise ValueError("SARVAM_API_KEY is missing in environment.")

    return Settings(
<<<<<<< HEAD
        sarvam_api_key=sarvam_api_key,
        sarvam_generation_model=os.getenv("SARVAM_GENERATION_MODEL", "sarvam-m").strip() or "sarvam-m",
        sarvam_temperature=float(os.getenv("SARVAM_TEMPERATURE", "0.3")),
        sarvam_top_p=float(os.getenv("SARVAM_TOP_P", "0.85")),
        sarvam_max_output_tokens=int(os.getenv("SARVAM_MAX_OUTPUT_TOKENS", "220")),
=======
        gemini_api_key=gemini_api_key,
        sarvam_api_key=sarvam_api_key,
        gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        gemini_generation_model=os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash"),
        gemini_timeout_seconds=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "25")),
        gemini_max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "1")),
        gemini_backoff_seconds=float(os.getenv("GEMINI_BACKOFF_SECONDS", "0.6")),
        gemini_temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.35")),
        gemini_top_p=float(os.getenv("GEMINI_TOP_P", "0.9")),
        gemini_max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "600")),
        gemini_thinking_budget=int(os.getenv("GEMINI_THINKING_BUDGET", "0")),
>>>>>>> 0459f549af78c8063fbe9d00dd04b416dad88c04
        data_dir=base_dir / "data",
        image_dir=base_dir / "data" / "images",
        knowledge_base_path=base_dir / "data" / "knowledge.md",
        artifact_dir=artifact_dir,
        session_store_dir=artifact_dir / "sessions_private",
        session_encryption_key=(os.getenv("SESSION_ENCRYPTION_KEY", "").strip() or sarvam_api_key),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        memory_turns=int(os.getenv("RAG_MEMORY_TURNS", "8")),
        max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "5000")),
    )
