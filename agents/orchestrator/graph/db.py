"""
db.py

Thin asyncpg wrapper for direct SQL execution in the executor.
Used for structured lookups (wallets, categories) where we already
have the query — no LLM needed.
"""
import logging
import os
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import asyncpg

logger = logging.getLogger(__name__)

WALLETS_DB_NAME = os.getenv("WALLETS_DB_NAME", "wallet_db").strip() or "wallet_db"
TRANSACTIONS_DB_NAME = os.getenv("TRANSACTIONS_DB_NAME", "transaction_db").strip() or "transaction_db"

_TABLE_DB_MAP = {
    "wallets": WALLETS_DB_NAME,
    "transactions": TRANSACTIONS_DB_NAME,
    "categories": TRANSACTIONS_DB_NAME,
}

_pools: dict[str, asyncpg.Pool] = {}


def _build_database_dsn(database_name: str) -> str:
    postgres_uri = os.getenv("POSTGRES_URI", "").strip()
    if not postgres_uri:
        raise RuntimeError("POSTGRES_URI is not set")

    parts = urlsplit(postgres_uri)
    if not parts.scheme or not parts.netloc:
        raise RuntimeError(
            "POSTGRES_URI must be a valid PostgreSQL URI like "
            "'postgresql://user:pass@host/database' or 'postgresql://user:pass@host'"
        )

    return urlunsplit((parts.scheme, parts.netloc, f"/{database_name}", parts.query, parts.fragment))


def _choose_database_name_for_sql(sql: str) -> str:
    normalized = re.sub(r"\s+", " ", sql.strip(), flags=re.MULTILINE)
    referenced_tables = {
        table_name
        for table_name in _TABLE_DB_MAP
        if re.search(rf"\b(?:FROM|JOIN)\s+{table_name}\b", normalized, re.IGNORECASE)
    }

    if not referenced_tables:
        raise ValueError(f"db: could not infer target database from SQL: {sql[:120]}")

    database_names = {_TABLE_DB_MAP[table_name] for table_name in referenced_tables}
    if len(database_names) != 1:
        raise ValueError(
            "db: query references tables from multiple databases, which direct SQL does not support"
        )

    return next(iter(database_names))


async def _get_pool(database_name: str) -> asyncpg.Pool:
    pool = _pools.get(database_name)
    if pool is None:
        dsn = _build_database_dsn(database_name)
        logger.info("db: creating asyncpg pool for database=%s", database_name)
        pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=5,
            command_timeout=10,
        )
        _pools[database_name] = pool
    return pool


async def execute_query(sql: str) -> list[dict[str, Any]]:
    """
    Execute a SELECT query and return rows as plain dicts.

    Only SELECT statements are allowed — anything else raises ValueError
    as a safety guard (mirrors the SQL agent's read-only constraint).
    """
    normalized = sql.strip().lstrip(";").strip().upper()
    if not normalized.startswith("SELECT"):
        raise ValueError(f"db: only SELECT queries allowed, got: {sql[:80]}")

    database_name = _choose_database_name_for_sql(sql)
    pool = await _get_pool(database_name)
    async with pool.acquire() as conn:
        try:
            records = await conn.fetch(sql)
        except asyncpg.PostgresError as exc:
            logger.error("db: query failed: %s | sql: %s", exc, sql[:200])
            raise

    rows = [dict(r) for r in records]
    logger.info("db: query returned %d rows | sql: %s", len(rows), sql[:120])
    return rows


async def close_pool() -> None:
    """Call on shutdown to release connections cleanly."""
    global _pools
    for database_name, pool in list(_pools.items()):
        await pool.close()
        logger.info("db: pool closed for database=%s", database_name)
    _pools = {}
