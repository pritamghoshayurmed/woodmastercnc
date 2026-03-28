from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.rag.types import DocumentChunk, RetrievedChunk


class FaissVectorStore:
    def __init__(self, index_path: Path, metadata_path: Path) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: faiss.Index | None = None
        self.chunks: list[DocumentChunk] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def build(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            raise ValueError("Cannot build index with no chunks.")
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count and embedding count must match.")

        vectors = np.asarray(embeddings, dtype=np.float32)
        vectors = self._normalize(vectors)

        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)
        self.chunks = chunks

    def save(self) -> None:
        if self.index is None:
            raise ValueError("Index is not initialized.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with self.metadata_path.open("w", encoding="utf-8") as file:
            json.dump([chunk.to_dict() for chunk in self.chunks], file, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Index or metadata not found in artifacts folder.")

        self.index = faiss.read_index(str(self.index_path))
        with self.metadata_path.open("r", encoding="utf-8") as file:
            raw_chunks = json.load(file)
        self.chunks = [DocumentChunk(**item) for item in raw_chunks]

    def exists(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if self.index is None:
            raise ValueError("Index is not initialized.")

        query = np.asarray([query_embedding], dtype=np.float32)
        query = self._normalize(query)

        scores, indices = self.index.search(query, top_k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))
        return results
