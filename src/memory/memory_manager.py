from __future__ import annotations

import json
from collections import deque
from pathlib import Path


class ConversationMemoryManager:
	def __init__(self, max_turns: int = 8, session_dir: str = "data/sessions") -> None:
		self.max_turns = max_turns
		self._sessions: dict[str, deque[dict[str, str]]] = {}
		self.session_dir = Path(session_dir)
		self.session_dir.mkdir(parents=True, exist_ok=True)

	def _get_session_file(self, session_id: str) -> Path:
		# Use basic sanitization to ensure valid filename
		safe_id = "".join(c for c in session_id if c.isalnum() or c in ('-', '_')).strip()
		if not safe_id:
			safe_id = "default"
		return self.session_dir / f"{safe_id}.json"

	def _load_session(self, session_id: str) -> deque[dict[str, str]]:
		if session_id in self._sessions:
			return self._sessions[session_id]

		session_file = self._get_session_file(session_id)
		history = deque(maxlen=self.max_turns * 2)
		if session_file.exists():
			try:
				with open(session_file, "r", encoding="utf-8") as f:
					data = json.load(f)
					for item in data:
						history.append(item)
			except Exception:
				pass
		
		self._sessions[session_id] = history
		return history

	def _save_session(self, session_id: str) -> None:
		session_file = self._get_session_file(session_id)
		history = self._sessions.get(session_id, [])
		try:
			with open(session_file, "w", encoding="utf-8") as f:
				json.dump(list(history), f, ensure_ascii=False, indent=2)
		except Exception:
			pass

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
			except OSError:
				pass

