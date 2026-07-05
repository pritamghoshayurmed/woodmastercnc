from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool


logger = logging.getLogger(__name__)


class DatabaseDisabledError(RuntimeError):
    """Raised when database-backed features are used while DB support is disabled."""


class DatabaseClient:
    def __init__(self, database_url: str, *, minconn: int = 1, maxconn: int = 8) -> None:
        self.database_url = database_url
        self._pool = ThreadedConnectionPool(
            minconn,
            maxconn,
            dsn=database_url,
            cursor_factory=RealDictCursor,
        )

    @contextmanager
    def connection(self) -> Iterator[Any]:
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
        return dict(row) if row else None

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)

    def execute_returning(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        self._pool.closeall()


_db_client: DatabaseClient | None = None
_db_lock = Lock()


def is_db_enabled() -> bool:
    return os.getenv("DB_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def get_db_client() -> DatabaseClient:
    global _db_client
    if not is_db_enabled():
        raise DatabaseDisabledError("Database support is disabled. Set DB_ENABLED=true.")

    database_url = get_database_url()
    if not database_url:
        raise DatabaseDisabledError("DATABASE_URL is missing.")

    if _db_client is None:
        with _db_lock:
            if _db_client is None:
                logger.info("Initializing PostgreSQL connection pool.")
                _db_client = DatabaseClient(database_url)
    return _db_client


def close_db_client() -> None:
    global _db_client
    with _db_lock:
        if _db_client is not None:
            _db_client.close()
            _db_client = None


def migration_path(filename: str = "001_initial_schema.sql") -> Path:
    return Path(__file__).resolve().parent / "migrations" / filename


def run_sql_file(path: Path | None = None) -> None:
    db = get_db_client()
    sql_path = path or migration_path()
    sql = sql_path.read_text(encoding="utf-8")
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
