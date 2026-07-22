from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.db import qna as db_qna

logger = logging.getLogger(__name__)

router = APIRouter()


class QnaCreateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    status: bool = True


class QnaUpdateRequest(BaseModel):
    question: str | None = Field(None, min_length=1)
    answer: str | None = Field(None, min_length=1)
    status: bool | None = None
    priority: int | None = None


class QnaReorderRequest(BaseModel):
    orderedIds: list[str]


def _to_payload(row: dict) -> dict:
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "status": row["enabled"],
        "priority": row["priority"],
    }


def _reload_pipeline() -> None:
    """Refresh the live RAG pipeline's FAQ chunks so edits apply without a restart."""
    try:
        from src.pipeline.rag_pipeline import get_active_pipeline

        pipeline = get_active_pipeline()
        if pipeline is not None:
            pipeline.reload_knowledge_base()
    except Exception:
        logger.exception("Failed to reload RAG pipeline after QnA change")


@router.get("")
def get_entries() -> dict:
    return {"qna": [_to_payload(row) for row in db_qna.get_all_entries()]}


@router.post("")
def create_entry(body: QnaCreateRequest) -> dict:
    priority = len(db_qna.get_all_entries()) + 1
    row = db_qna.create_entry(body.question, body.answer, priority, body.status)
    _reload_pipeline()
    return {"qna": _to_payload(row)}


@router.put("/{entry_id}")
def update_entry(entry_id: str, body: QnaUpdateRequest) -> dict:
    row = db_qna.update_entry(
        entry_id,
        {
            "question": body.question,
            "answer": body.answer,
            "enabled": body.status,
            "priority": body.priority,
        },
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Q&A entry not found.")
    _reload_pipeline()
    return {"qna": _to_payload(row)}


@router.delete("/{entry_id}")
def delete_entry(entry_id: str) -> dict:
    deleted = db_qna.delete_entry(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Q&A entry not found.")
    _reload_pipeline()
    return {"deleted": True}


@router.post("/reorder")
def reorder_entries(body: QnaReorderRequest) -> dict:
    db_qna.reorder_entries(body.orderedIds)
    _reload_pipeline()
    return {"qna": [_to_payload(row) for row in db_qna.get_all_entries()]}
