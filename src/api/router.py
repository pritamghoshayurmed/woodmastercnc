from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.ai_settings import router as ai_settings_router
from src.api.auth import require_dashboard_auth
from src.api.auth import router as auth_router
from src.api.conversations import router as conversations_router
from src.api.dashboard import router as dashboard_router
from src.api.leads import router as leads_router
from src.api.qna import router as qna_router
from src.api.scoring_rules import router as scoring_rules_router


api_router = APIRouter()
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])

_protected = APIRouter(dependencies=[Depends(require_dashboard_auth)])
_protected.include_router(leads_router, prefix="/leads", tags=["leads"])
_protected.include_router(conversations_router, prefix="/conversations", tags=["conversations"])
_protected.include_router(scoring_rules_router, prefix="/scoring-rules", tags=["scoring-rules"])
_protected.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])
_protected.include_router(qna_router, prefix="/qna", tags=["qna"])
_protected.include_router(ai_settings_router, prefix="/ai-settings", tags=["ai-settings"])
api_router.include_router(_protected)
