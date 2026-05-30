"""WhatsApp Cloud API helpers.

Provides functions to send text, image, and interactive button messages
via the WhatsApp Business Cloud API.  All helpers are no-ops when the
required environment variables are not configured.
"""
from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)


def _settings() -> tuple[str, str, str]:
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()
    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
    graph_version = os.getenv("WHATSAPP_GRAPH_VERSION", "v23.0").strip()
    return phone_number_id, access_token, graph_version


def is_configured() -> bool:
    phone_number_id, access_token, _ = _settings()
    return bool(phone_number_id and access_token)


def _base_url() -> str:
    phone_number_id, _, graph_version = _settings()
    return f"https://graph.facebook.com/{graph_version}/{phone_number_id}/messages"


def _headers() -> dict[str, str]:
    _, access_token, _ = _settings()
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def send_text(to_number: str, text: str) -> None:
    """Send a plain-text message (max 4096 chars)."""
    if not is_configured():
        logger.error("WhatsApp env vars missing — cannot send text message.")
        return
    resp = requests.post(
        _base_url(),
        headers=_headers(),
        json={
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {"body": text[:4096]},
        },
        timeout=20,
    )
    if resp.status_code >= 300:
        logger.error("WhatsApp send_text failed: %s %s", resp.status_code, resp.text)


def send_image(to_number: str, image_url: str) -> None:
    """Send an image message using a public HTTPS URL."""
    if not is_configured():
        logger.error("WhatsApp env vars missing — cannot send image message.")
        return
    resp = requests.post(
        _base_url(),
        headers=_headers(),
        json={
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "image",
            "image": {"link": image_url},
        },
        timeout=20,
    )
    if resp.status_code >= 300:
        logger.error("WhatsApp send_image failed: %s %s", resp.status_code, resp.text)


def send_interactive_buttons(to_number: str, body_text: str, options: list[dict]) -> None:
    """Send interactive quick-reply buttons (max 3 per WhatsApp spec)."""
    if not is_configured():
        logger.error("WhatsApp env vars missing — cannot send interactive message.")
        return
    buttons = [
        {"type": "reply", "reply": {"id": str(opt["value"])[:256], "title": str(opt["label"])[:20]}}
        for opt in options[:3]
    ]
    resp = requests.post(
        _base_url(),
        headers=_headers(),
        json={
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": body_text},
                "action": {"buttons": buttons},
            },
        },
        timeout=20,
    )
    if resp.status_code >= 300:
        logger.error("WhatsApp send_interactive_buttons failed: %s %s", resp.status_code, resp.text)
