from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from collections import deque
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


logger = logging.getLogger(__name__)


class ConversationMemoryManager:
	def __init__(
		self,
		max_turns: int = 8,
		session_dir: str | Path = "artifacts/sessions_private",
		encryption_key: str | None = None,
	) -> None:
		self.max_turns = max_turns
		self._sessions: dict[str, deque[dict[str, str]]] = {}
		self.session_dir = Path(session_dir)
		self.session_dir.mkdir(parents=True, exist_ok=True)
		self._fernet: Fernet | None = None

		if encryption_key:
			self._fernet = Fernet(self._normalize_fernet_key(encryption_key))
		else:
			logger.warning(
				"Session encryption is disabled. Set SESSION_ENCRYPTION_KEY to enable encryption at rest."
			)

	@staticmethod
	def _normalize_fernet_key(raw_key: str) -> bytes:
		key_bytes = raw_key.encode("utf-8")
		try:
			Fernet(key_bytes)
			return key_bytes
		except Exception:
			pass

		digest = hashlib.sha256(key_bytes).digest()
		return base64.urlsafe_b64encode(digest)

	def _get_session_file(self, session_id: str) -> Path:
		normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", session_id).strip("-_").lower()
		if not normalized:
			normalized = "session"
		normalized = normalized[:48]
		hash_suffix = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
		return self.session_dir / f"{normalized}-{hash_suffix}.json"

	def _serialize_payload(self, payload: list[dict[str, str]]) -> bytes:
		encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
		if self._fernet:
			return self._fernet.encrypt(encoded)
		return encoded

	def _deserialize_payload(self, blob: bytes) -> list[dict[str, str]]:
		data = blob
		if self._fernet:
			try:
				data = self._fernet.decrypt(blob)
			except InvalidToken as exc:
				raise RuntimeError("Failed to decrypt session file. Check SESSION_ENCRYPTION_KEY.") from exc
		return json.loads(data.decode("utf-8"))

	def _load_session(self, session_id: str) -> deque[dict[str, str]]:
		if session_id in self._sessions:
			return self._sessions[session_id]

		session_file = self._get_session_file(session_id)
		history = deque(maxlen=self.max_turns * 2)
		if session_file.exists():
			try:
				blob = session_file.read_bytes()
				data = self._deserialize_payload(blob)
				for item in data:
					history.append(item)
			except (OSError, json.JSONDecodeError, RuntimeError) as exc:
				logger.exception("Failed to load session history", extra={"session_id": session_id, "file": str(session_file)})
				raise RuntimeError(f"Unable to load session '{session_id}'.") from exc
		
		self._sessions[session_id] = history
		return history

	def _save_session(self, session_id: str) -> None:
		session_file = self._get_session_file(session_id)
		history = self._sessions.get(session_id, [])
		try:
			payload = self._serialize_payload(list(history))
			session_file.write_bytes(payload)
		except OSError as exc:
			logger.exception("Failed to save session history", extra={"session_id": session_id, "file": str(session_file)})
			raise RuntimeError(f"Unable to persist session '{session_id}'.") from exc

	def add_user_message(self, session_id: str, message: str) -> None:
		history = self._load_session(session_id)
		history.append({"role": "user", "content": message})
		self._save_session(session_id)

	def add_assistant_message(self, session_id: str, message: str) -> None:
		history = self._load_session(session_id)
		history.append({"role": "assistant", "content": message})
		self._save_session(session_id)

	def get_recent_history(self, session_id: str) -> list[dict[str, str]]:
		return list(self._load_session(session_id))

	def clear(self, session_id: str) -> None:
		self._sessions.pop(session_id, None)
		session_file = self._get_session_file(session_id)
		if session_file.exists():
			try:
				session_file.unlink()
			except OSError as exc:
				logger.exception("Failed to delete session history", extra={"session_id": session_id, "file": str(session_file)})
				raise RuntimeError(f"Unable to delete session '{session_id}'.") from exc

