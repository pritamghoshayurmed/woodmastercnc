from __future__ import annotations

import getpass
import hashlib
import secrets
from pathlib import Path

from dotenv import set_key

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def main() -> None:
    password = getpass.getpass("New dashboard password: ")
    confirm = getpass.getpass("Confirm password: ")
    if not password:
        print("Password cannot be empty.")
        raise SystemExit(1)
    if password != confirm:
        print("Passwords did not match.")
        raise SystemExit(1)

    ENV_PATH.touch(exist_ok=True)
    password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    set_key(str(ENV_PATH), "DASHBOARD_PASSWORD_HASH", password_hash)

    existing_secret = _read_existing_secret()
    if not existing_secret:
        set_key(str(ENV_PATH), "DASHBOARD_AUTH_SECRET", secrets.token_hex(32))

    print(f"Updated DASHBOARD_PASSWORD_HASH in {ENV_PATH}. Restart the server for it to take effect.")


def _read_existing_secret() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("DASHBOARD_AUTH_SECRET="):
            return line.split("=", 1)[1].strip()
    return ""


if __name__ == "__main__":
    main()
