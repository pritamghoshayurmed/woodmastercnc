from __future__ import annotations

import hashlib
import hmac
import logging
import os

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class LoginRequest(BaseModel):
    password: str


def _password_hash() -> str:
    return os.getenv("DASHBOARD_PASSWORD_HASH", "").strip().lower()


def _session_token() -> str:
    """A stable, non-expiring bearer token derived from server-side secrets.

    Recomputed from env on every request instead of persisted anywhere, so
    restarting the process with the same .env keeps existing browser sessions
    valid ("stay logged in") while changing the password or secret revokes them.
    """
    secret = os.getenv("DASHBOARD_AUTH_SECRET", "").strip() or _password_hash()
    return hmac.new(secret.encode("utf-8"), b"wmcnc-dashboard-session", hashlib.sha256).hexdigest()


def auth_configured() -> bool:
    return bool(_password_hash())


@router.get("/status")
def status() -> dict:
    return {"configured": auth_configured()}


@router.post("/login")
def login(body: LoginRequest) -> dict:
    expected_hash = _password_hash()
    if not expected_hash:
        raise HTTPException(status_code=503, detail="Dashboard password is not configured on the server.")

    submitted_hash = hashlib.sha256(body.password.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(submitted_hash, expected_hash):
        raise HTTPException(status_code=401, detail="Incorrect password.")

    return {"token": _session_token()}


def require_dashboard_auth(authorization: str | None = Header(default=None)) -> None:
    """Dependency gating every /api/* route (except /api/auth/login).

    If no password has been configured on the server, auth is left open —
    matches the rest of this codebase's "REQUIRE_*" opt-in pattern for local
    development without full config.
    """
    if not auth_configured():
        return

    token = (authorization or "").removeprefix("Bearer ").strip()
    if not token or not hmac.compare_digest(token, _session_token()):
        raise HTTPException(status_code=401, detail="Not authenticated.")
