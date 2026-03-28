from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


@dataclass
class RAGResponse:
    answer: str
    sources: list[str]
    images: list[str]
    retrieved_chunks: list[RetrievedChunk]
