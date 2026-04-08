from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any

import requests
from dotenv import load_dotenv

DEFAULT_RECIPIENT = "+916295716352"
DEFAULT_MESSAGE = "Welcome! Your WhatsApp assistant is connected and ready to chat."


def normalize_phone_number(phone_number: str) -> str:
    digits_only = re.sub(r"\D+", "", phone_number)
    if not digits_only:
        raise ValueError("Recipient number is empty after normalization.")
    return digits_only


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_payload(to_number: str, message: str) -> dict[str, Any]:
    return {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message[:4096]},
    }


def parse_meta_error(response: requests.Response) -> tuple[int | None, int | None, str]:
    try:
        data = response.json()
    except ValueError:
        return None, None, response.text

    if not isinstance(data, dict):
        return None, None, response.text

    error = data.get("error", {})
    if not isinstance(error, dict):
        return None, None, response.text

    code = error.get("code")
    subcode = error.get("error_subcode")
    message = str(error.get("message") or response.text)
    return code, subcode, message


def send_message(to_number: str, message: str, dry_run: bool = False) -> None:
    load_dotenv()

    phone_number_id = require_env("WHATSAPP_PHONE_NUMBER_ID")
    access_token = require_env("WHATSAPP_ACCESS_TOKEN")
    graph_version = os.getenv("WHATSAPP_GRAPH_VERSION", "v23.0").strip() or "v23.0"

    normalized_number = normalize_phone_number(to_number)
    payload = build_payload(normalized_number, message)
    url = f"https://graph.facebook.com/{graph_version}/{phone_number_id}/messages"

    if dry_run:
        print("Dry run enabled. Request not sent.")
        print(f"POST {url}")
        print(payload)
        return

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=25)
    if response.status_code >= 300:
        code, subcode, message = parse_meta_error(response)
        if code == 190 and subcode == 463:
            raise RuntimeError(
                "WHATSAPP_ACCESS_TOKEN has expired. Generate a fresh token in Meta WhatsApp API Setup, "
                "update WHATSAPP_ACCESS_TOKEN in .env, and run this script again."
            )
        if code == 190:
            raise RuntimeError(
                "WHATSAPP_ACCESS_TOKEN is invalid or expired. Generate a valid token in Meta, update .env, "
                "and retry."
            )
        raise RuntimeError(
            "WhatsApp API request failed with "
            f"HTTP {response.status_code} (code={code}, subcode={subcode}): {message}"
        )

    print("Welcome message sent successfully.")
    print(response.text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a WhatsApp welcome message from your configured test number."
    )
    parser.add_argument(
        "--to",
        default=DEFAULT_RECIPIENT,
        help=f"Recipient WhatsApp number in international format. Default: {DEFAULT_RECIPIENT}",
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_MESSAGE,
        help="Message body to send.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show request details without sending a WhatsApp message.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        send_message(to_number=args.to, message=args.message, dry_run=args.dry_run)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())