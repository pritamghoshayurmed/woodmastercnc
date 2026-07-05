from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # ─── LiteLLM / NVIDIA NIM ──────────────────────────────────────────────────
    nvidia_nim_api_key: str
    llm_model: str
    llm_fallback_model: str
    llm_temperature: float
    llm_top_p: float
    llm_max_output_tokens: int

    # ─── RAG / Data ────────────────────────────────────────────────────────────
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

    nvidia_nim_api_key = os.getenv("NVIDIA_NIM_API_KEY", "").strip()
    if not nvidia_nim_api_key:
        raise ValueError(
            "NVIDIA_NIM_API_KEY is missing in environment. "
            "Set it to your NVIDIA NIM API key."
        )

    return Settings(
        nvidia_nim_api_key=nvidia_nim_api_key,
        llm_model=os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b").strip(),
        llm_fallback_model=os.getenv("LLM_FALLBACK_MODEL", "meta/llama-3.3-70b-instruct").strip(),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        llm_top_p=float(os.getenv("LLM_TOP_P", "0.85")),
        llm_max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "512")),
        data_dir=base_dir / "data",
        image_dir=base_dir / "data" / "images",
        knowledge_base_path=base_dir / "data" / "knowledge.md",
        artifact_dir=artifact_dir,
        session_store_dir=artifact_dir / "sessions_private",
        session_encryption_key=(
            os.getenv("SESSION_ENCRYPTION_KEY", "").strip() or nvidia_nim_api_key
        ),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        memory_turns=int(os.getenv("RAG_MEMORY_TURNS", "8")),
        max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "5000")),
    )
