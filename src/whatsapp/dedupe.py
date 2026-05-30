from __future__ import annotations

import time


class RecentMessageCache:
    def __init__(self, ttl_seconds: int = 180) -> None:
        self.ttl_seconds = ttl_seconds
        self._seen: dict[str, float] = {}

    def seen_recently(self, key: str) -> bool:
        now = time.time()
        self._prune(now)
        previous = self._seen.get(key)
        self._seen[key] = now
        return previous is not None

    def _prune(self, now: float) -> None:
        expired = [key for key, ts in self._seen.items() if now - ts > self.ttl_seconds]
        for key in expired:
            self._seen.pop(key, None)
