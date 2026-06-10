"""
db.py

Thin asyncpg wrapper for direct SQL execution in the executor.
Used for structured lookups (wallets, categories) where we already
have the query — no LLM needed.
"""
import logging
import os
import re
import asyncio
from typing import Any
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

WALLETS_DB_NAME = os.getenv("WALLETS_DB_NAME", "wallet_db").strip() or "wallet_db"
TRANSACTIONS_DB_NAME = os.getenv("TRANSACTIONS_DB_NAME", "transaction_db").strip() or "transaction_db"

_TABLE_DB_MAP = {
    "wallets": WALLETS_DB_NAME,
    "transactions": TRANSACTIONS_DB_NAME,
    "categories": TRANSACTIONS_DB_NAME,
}

_pools: dict[str, Any] = {}
_mongo_client: Any | None = None


def _get_mongo_uri() -> str:
    return (
        os.getenv("MONGO_URL", "").strip()
        or os.getenv("mongo_url", "").strip()
        or os.getenv("MONGO_URI", "").strip()
    )


def _get_mongo_collection(name: str):
    global _mongo_client
    mongo_uri = _get_mongo_uri()
    if not mongo_uri:
        raise RuntimeError("MONGO_URL/mongo_url/MONGO_URI is not set")
    if _mongo_client is None:
        from pymongo import MongoClient
        _mongo_client = MongoClient(mongo_uri)
    db_name = os.getenv("MONGO_DB_NAME", "budget-tracker").strip() or "budget-tracker"
    return _mongo_client[db_name][name]


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


async def _get_pool(database_name: str):
    pool = _pools.get(database_name)
    if pool is None:
        dsn = _build_database_dsn(database_name)
        logger.info("db: creating asyncpg pool for database=%s", database_name)
        import asyncpg
        pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=5,
            command_timeout=10,
        )
        _pools[database_name] = pool
    return pool


def _normalize_mongo_row(collection_name: str, document: dict[str, Any]) -> dict[str, Any]:
    if collection_name == "wallets":
        return {
            "id": str(document.get("walletId") or document.get("_id") or ""),
            "name": document.get("name", ""),
            "balance": document.get("balance", 0),
            "currency": document.get("currency", ""),
        }

    if collection_name == "categories":
        return {
            "id": str(document.get("categoryId") or document.get("_id") or ""),
            "name": document.get("name", ""),
            "icon": document.get("icon", ""),
            "budget_limit": document.get("budgetLimit"),
            "period": document.get("period"),
        }

    if collection_name == "transactions":
        return {
            "id": str(document.get("id") or document.get("_id") or ""),
            "amount": document.get("amount"),
            "type": document.get("type"),
            "category_id": document.get("category_id"),
            "category_name": document.get("category_name"),
            "description": document.get("description"),
            "wallet_id": document.get("wallet_id"),
            "user_id": document.get("user_id"),
            "transaction_date": document.get("transaction_date"),
            "created_at": document.get("created_at"),
            "updated_at": document.get("updated_at"),
            "image_url": document.get("image_url"),
        }

    return dict(document)


def _sort_rows(rows: list[dict[str, Any]], field_name: str, descending: bool) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[int, Any]:
        value = row.get(field_name)
        if value is None:
            return (1, None)
        if isinstance(value, str):
            stripped = value.strip().replace(",", "")
            try:
                return (0, float(stripped))
            except ValueError:
                return (0, value.lower())
        return (0, value)

    return sorted(rows, key=sort_key, reverse=descending)


def _execute_mongo_query_sync(sql: str) -> list[dict[str, Any]]:
    normalized = re.sub(r"\s+", " ", sql.strip(), flags=re.MULTILINE)
    table_match = re.search(r"\bFROM\s+(wallets|categories|transactions)\b", normalized, re.IGNORECASE)
    if not table_match:
        raise ValueError(f"db: could not infer Mongo collection from SQL: {sql[:120]}")

    collection_name = table_match.group(1).lower()
    query: dict[str, Any] = {}

    user_match = re.search(r"user_id\s*=\s*'([^']+)'", normalized, re.IGNORECASE)
    if user_match:
        user_key = "userId" if collection_name in {"wallets", "categories"} else "user_id"
        query[user_key] = user_match.group(1)

    name_match = re.search(r"name\s*=\s*'([^']+)'", normalized, re.IGNORECASE)
    if name_match:
        query["name"] = name_match.group(1)

    active_match = re.search(r"active\s*=\s*(true|false)", normalized, re.IGNORECASE)
    if active_match and collection_name in {"wallets", "categories"}:
        query["active"] = active_match.group(1).lower() == "true"

    collection = _get_mongo_collection(collection_name)
    rows = [_normalize_mongo_row(collection_name, doc) for doc in collection.find(query)]

    order_match = re.search(r"\bORDER\s+BY\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(ASC|DESC)?\b", normalized, re.IGNORECASE)
    if order_match:
        order_field = order_match.group(1)
        descending = str(order_match.group(2) or "ASC").upper() == "DESC"
        rows = _sort_rows(rows, order_field, descending)

    limit_match = re.search(r"\bLIMIT\s+(\d+)\b", normalized, re.IGNORECASE)
    if limit_match:
        rows = rows[: int(limit_match.group(1))]
    logger.info("db: mongo query returned %d rows | sql: %s", len(rows), sql[:120])
    return rows


async def _execute_mongo_query(sql: str) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_execute_mongo_query_sync, sql)


async def execute_query(sql: str) -> list[dict[str, Any]]:
    """
    Execute a SELECT query and return rows as plain dicts.

    Only SELECT statements are allowed — anything else raises ValueError
    as a safety guard (mirrors the SQL agent's read-only constraint).
    """
    normalized = sql.strip().lstrip(";").strip().upper()
    if not normalized.startswith("SELECT"):
        raise ValueError(f"db: only SELECT queries allowed, got: {sql[:80]}")

    postgres_uri = os.getenv("POSTGRES_URI", "").strip()
    if _get_mongo_uri() or not postgres_uri:
        return await _execute_mongo_query(sql)

    database_name = _choose_database_name_for_sql(sql)
    pool = await _get_pool(database_name)
    async with pool.acquire() as conn:
        try:
            records = await conn.fetch(sql)
        except Exception as exc:
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
