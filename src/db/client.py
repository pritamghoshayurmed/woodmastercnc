from __future__ import annotations

import logging
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Iterator
from urllib.parse import urlsplit

import psycopg2
from psycopg2 import OperationalError
from psycopg2.extensions import parse_dsn
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool


logger = logging.getLogger(__name__)


class DatabaseDisabledError(RuntimeError):
    """Raised when database-backed features are used while DB support is disabled."""


class DatabaseClient:
    def __init__(self, database_url: str, *, minconn: int = 1, maxconn: int = 8) -> None:
        self.database_url = database_url
        self._pool = self._create_pool(minconn=minconn, maxconn=maxconn)

    def _create_pool(self, *, minconn: int, maxconn: int) -> ThreadedConnectionPool:
        connect_kwargs = _build_connect_kwargs(self.database_url)
        try:
            return ThreadedConnectionPool(
                minconn,
                maxconn,
                cursor_factory=RealDictCursor,
                **connect_kwargs,
            )
        except OperationalError as exc:
            fallback_kwargs = _build_ipv4_fallback_kwargs(self.database_url, exc)
            if fallback_kwargs is None:
                raise

            logger.warning(
                "PostgreSQL connection failed with IPv6/network error; retrying with IPv4 hostaddr "
                "for host '%s'.",
                fallback_kwargs.get("host"),
            )
            return ThreadedConnectionPool(
                minconn,
                maxconn,
                cursor_factory=RealDictCursor,
                **fallback_kwargs,
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
    pooler_url = os.getenv("DATABASE_POOLER_URL", "").strip()
    if pooler_url:
        return pooler_url
    return os.getenv("DATABASE_URL", "").strip()


def _build_connect_kwargs(database_url: str) -> dict[str, Any]:
    return dict(parse_dsn(database_url))


def _parse_database_target(database_url: str) -> tuple[str | None, int | None]:
    parsed = urlsplit(database_url)
    return parsed.hostname, parsed.port


def _is_supabase_direct_connection(hostname: str | None, port: int | None) -> bool:
    return bool(hostname and hostname.startswith("db.") and hostname.endswith(".supabase.co") and port == 5432)


def _resolve_ipv4_address(hostname: str, port: int | None) -> str | None:
    try:
        addrinfo = socket.getaddrinfo(hostname, port, socket.AF_INET, socket.SOCK_STREAM)
    except socket.gaierror:
        return None

    for _, _, _, _, sockaddr in addrinfo:
        hostaddr = sockaddr[0]
        if hostaddr:
            return hostaddr
    return None


def _build_ipv4_fallback_kwargs(database_url: str, error: OperationalError) -> dict[str, Any] | None:
    error_message = str(error).lower()
    if "network is unreachable" not in error_message:
        return None

    connect_kwargs = _build_connect_kwargs(database_url)
    if connect_kwargs.get("hostaddr"):
        return None

    hostname, port = _parse_database_target(database_url)
    if not hostname:
        return None

    port = port or (int(connect_kwargs["port"]) if connect_kwargs.get("port") else None)
    hostaddr = _resolve_ipv4_address(hostname, port)
    if not hostaddr:
        return None

    fallback_kwargs = dict(connect_kwargs)
    fallback_kwargs["hostaddr"] = hostaddr
    return fallback_kwargs


def get_db_client() -> DatabaseClient:
    global _db_client
    if not is_db_enabled():
        raise DatabaseDisabledError("Database support is disabled. Set DB_ENABLED=true.")

    database_url = get_database_url()
    if not database_url:
        raise DatabaseDisabledError("DATABASE_URL is missing.")

    hostname, port = _parse_database_target(database_url)
    if _is_supabase_direct_connection(hostname, port):
        logger.warning(
            "DATABASE_URL is using Supabase's direct host '%s:%s'. On Render this often fails because "
            "that endpoint can be IPv6-only. Prefer setting DATABASE_POOLER_URL to your Supabase pooler "
            "connection string on port 6543.",
            hostname,
            port,
        )

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
