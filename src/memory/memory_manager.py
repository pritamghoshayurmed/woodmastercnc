from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from collections import deque
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from src.db import conversations as db_conversations
from src.db import messages as db_messages
from src.db import users as db_users
from src.db.client import is_db_enabled


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
		# Durable deployments keep conversation history in PostgreSQL. File-backed
		# history is deliberately a development-only fallback (DB_ENABLED=false).
		if is_db_enabled():
			return deque(self._database_history(session_id), maxlen=self.max_turns * 2)
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
				logger.warning(
					"Failed to load session history; starting a fresh session instead.",
					extra={"session_id": session_id, "file": str(session_file), "error_type": exc.__class__.__name__},
				)
				self._archive_unreadable_session(session_file)
		
		self._sessions[session_id] = history
		return history

	def _database_history(self, session_id: str) -> list[dict[str, str]]:
		"""Return the active conversation transcript in the format used by RAG."""
		user = db_users.get_user_by_phone(session_id)
		if not user:
			return []
		conversation = db_conversations.get_active_conversation(user["id"])
		if not conversation:
			return []
		rows = db_messages.get_conversation_messages(conversation["id"], limit=self.max_turns * 2)
		return [
			{
				"role": "assistant" if row.get("sender") == "BOT" else "user",
				"content": self._normalize_message(str(row.get("message") or "")),
			}
			for row in rows
			if row.get("sender") in {"USER", "BOT"} and str(row.get("message") or "").strip()
		]

	def _archive_unreadable_session(self, session_file: Path) -> None:
		if not session_file.exists():
			return

		backup_path = session_file.with_suffix(f"{session_file.suffix}.invalid")
		try:
			if backup_path.exists():
				backup_path.unlink()
			session_file.replace(backup_path)
		except OSError:
			logger.warning("Failed to archive unreadable session file", extra={"file": str(session_file)})

	def _save_session(self, session_id: str) -> None:
		session_file = self._get_session_file(session_id)
		history = self._sessions.get(session_id, [])
		try:
			payload = self._serialize_payload(list(history))
			session_file.write_bytes(payload)
		except OSError as exc:
			logger.exception("Failed to save session history", extra={"session_id": session_id, "file": str(session_file)})
			raise RuntimeError(f"Unable to persist session '{session_id}'.") from exc

	@staticmethod
	def _normalize_message(message: str) -> str:
		return re.sub(r"\s+", " ", message).strip()

	def _append_message(self, session_id: str, role: str, message: str) -> None:
		# Request handlers record USER and BOT messages in the database. Do not
		# duplicate them or create local transcripts while DB persistence is active.
		if is_db_enabled():
			return
		normalized = self._normalize_message(message)
		if not normalized:
			return

		history = self._load_session(session_id)
		if history and history[-1].get("role") == role and history[-1].get("content") == normalized:
			return

		history.append({"role": role, "content": normalized})
		self._save_session(session_id)

	def add_user_message(self, session_id: str, message: str) -> None:
		self._append_message(session_id, "user", message)

	def add_assistant_message(self, session_id: str, message: str) -> None:
		self._append_message(session_id, "assistant", message)

	def get_recent_history(self, session_id: str, limit: int | None = None) -> list[dict[str, str]]:
		history = list(self._load_session(session_id))
		if limit is None or limit >= len(history):
			return history
		return history[-limit:]

	def get_context_window(
		self,
		session_id: str,
		question: str,
		max_messages: int = 6,
	) -> list[dict[str, str]]:
		history = list(self._load_session(session_id))
		normalized_question = self._normalize_message(question)
		if history and history[-1].get("role") == "user" and history[-1].get("content") == normalized_question:
			history = history[:-1]

		filtered = [
			turn
			for turn in history
			if turn.get("role") in {"user", "assistant"} and turn.get("content")
		]
		return filtered[-max_messages:]

	def clear(self, session_id: str) -> None:
		# Timeouts create a new conversation; prior DB messages stay for audit.
		if is_db_enabled():
			return
		self._sessions.pop(session_id, None)
		session_file = self._get_session_file(session_id)
		if session_file.exists():
			try:
				session_file.unlink()
			except OSError as exc:
				logger.exception("Failed to delete session history", extra={"session_id": session_id, "file": str(session_file)})
				raise RuntimeError(f"Unable to delete session '{session_id}'.") from exc

