from __future__ import annotations

from fastapi import APIRouter

from src.api.analytics import router as analytics_router
from src.api.conversations import router as conversations_router
from src.api.leads import router as leads_router
from src.api.scoring_rules import router as scoring_rules_router


api_router = APIRouter()
api_router.include_router(leads_router, prefix="/leads", tags=["leads"])
api_router.include_router(conversations_router, prefix="/conversations", tags=["conversations"])
api_router.include_router(scoring_rules_router, prefix="/scoring-rules", tags=["scoring-rules"])
api_router.include_router(analytics_router, prefix="/analytics", tags=["analytics"])
