import os
import re
import logging
import json
import base64
import uuid
from typing import Any, Dict, Optional
from contextvars import ContextVar
from datetime import UTC, datetime

from .candidate_binding import select_candidate_binding
from .graph.state import State
import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# Context vars – populated by CredentialMiddleware for every request
# ---------------------------------------------------------------------------

current_token: ContextVar[str] = ContextVar("current_token", default="")
current_user_id: ContextVar[str] = ContextVar("current_user_id", default="")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRANSACTION_API_URL: str = os.getenv("TRANSACTION_API_URL", "")
CATEGORY_API_URL: str = os.getenv("CATEGORY_API_URL", "")
DEFAULT_USER_ID: str = os.getenv("TRANSACTION_USER_ID", "")
REPORT_API_URL: str = os.getenv("REPORT_API_URL", "")
MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "budget-tracker").strip() or "budget-tracker"
_mongo_client: Any | None = None
_candidate_binding_llm: Any | None = None

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _strip_bearer(value: str) -> str:
    token = str(value or "").strip()
    return token.removeprefix("Bearer ").removeprefix("bearer ").strip()


def _parse_amount(value: Any) -> Optional[int]:
    """Normalize shorthand amounts to int.

    Examples: "50k" → 50000 | "20,000" → 20000 | 20000 → 20000
    Returns None when the value cannot be parsed.
    """
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return int(value)
        s = str(value).strip().lower().replace(",", "")
        if not s:
            return None
        if s.endswith("k"):
            return int(float(s[:-1]) * 1_000)
        m = re.search(r"[0-9]+(?:\.[0-9]+)?", s)
        if m:
            return int(float(m.group(0)))
    except Exception:
        pass
    return None


def _resolve_token(tool_token: str) -> str:
    """Return the best available bearer token (tool arg > context var)."""
    return _strip_bearer(tool_token) or _strip_bearer(current_token.get())


def _resolve_user_id(tool_user_id: str) -> str:
    """Return the best available user-id (tool arg > context var > env)."""
    return (
        tool_user_id.strip()
        or current_user_id.get().strip()
        or DEFAULT_USER_ID
    )


def _get_mongo_uri() -> str:
    return (
        os.getenv("MONGO_URL", "").strip()
        or os.getenv("mongo_url", "").strip()
        or os.getenv("MONGO_URI", "").strip()
    )


def _get_mongo_client():
    global _mongo_client
    if _mongo_client is None:
        mongo_uri = _get_mongo_uri()
        if not mongo_uri:
            raise RuntimeError("MONGO_URL/mongo_url/MONGO_URI is not set")
        from pymongo import MongoClient
        _mongo_client = MongoClient(mongo_uri)
    return _mongo_client


def _get_candidate_binding_llm():
    global _candidate_binding_llm
    if _candidate_binding_llm is None:
        from .llm import get_classifier_llm
        _candidate_binding_llm = get_classifier_llm()
    return _candidate_binding_llm


def _get_collection(name: str):
    return _get_mongo_client()[MONGO_DB_NAME][name]


# ---------------------------------------------------------------------------
# Standardized API client
# ---------------------------------------------------------------------------

class APIError(Exception):
    """Raised when the backend returns a 4xx / 5xx status."""
    def __init__(self, status_code: int, data: Any):
        self.status_code = status_code
        self.data = data
        super().__init__(f"Backend error {status_code}: {data}")


mcp = FastMCP(name="transaction-service")





async def _api_call(
    method: str,
    url: str,
    *,
    bearer: str = "",
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Single async entry-point for every outbound API call.

    Returns a normalised dict:
        {"status": "success", "status_code": int, "data": <parsed json>}
    Raises APIError on 4xx/5xx and httpx.HTTPError on network failure.
    """
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    #log contextual info
    
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.request(
            method.upper(),
            url,
            json=payload,
            params=params,
            headers=headers,
        )

    try:
        data = response.json()
    except ValueError:
        data = {"raw": response.text}

    logger.info(
        "_api_call %s %s → status=%s body=%s",
        method.upper(), url, response.status_code, str(data)[:400],
    )

    if response.status_code >= 400:
        raise APIError(response.status_code, data)

    return {"status": "success", "status_code": response.status_code, "data": data}


def _error_response(message: str, **extra) -> Dict[str, Any]:
    """Uniform error dict returned to the MCP caller."""
    return {"status": "error", "message": message, **extra}


def _auth_error_message(status_code: int) -> str:
    if status_code == 401:
        return "Unauthorized (401). Provide a valid token via the tool arg or Authorization header."
    if status_code == 403:
        return "Forbidden (403). The token does not have permission for this operation."
    return f"Backend returned an error (HTTP {status_code})."


def _normalize_document_id(document: dict[str, Any], field_name: str) -> dict[str, Any]:
    normalized = dict(document)
    if "_id" in normalized and field_name not in normalized:
        normalized[field_name] = str(normalized["_id"])
    normalized.pop("_id", None)
    return normalized


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        from bson import ObjectId
        if isinstance(value, ObjectId):
            return str(value)
    except Exception:
        pass
    return value


def _wallet_candidates_for_user(user_id: str) -> list[dict[str, Any]]:
    wallets = []
    for wallet in _get_collection("wallets").find({"userId": user_id}):
        if wallet.get("active") is False:
            continue
        wallet_id = str(wallet.get("walletId") or wallet.get("wallet_id") or wallet.get("_id") or "")
        if not wallet_id:
            continue
        wallets.append(
            {
                "id": wallet_id,
                "walletId": wallet_id,
                "name": wallet.get("name") or wallet.get("label") or wallet_id,
                "balance": wallet.get("balance"),
                "currency": wallet.get("currency"),
            }
        )
    return wallets

async def _select_wallet_id_with_llm(
    wallets: list[dict[str, Any]],
    *,
    wallet_name: str = "",
    text_hint: str = "",
    description: str = "",
) -> str | None:
    user_text = wallet_name.strip() or text_hint.strip() or description.strip()
    if not user_text:
        return None

    decision = await select_candidate_binding(
        _get_candidate_binding_llm(),
        user_text=user_text,
        current_context={
            "name": "create_transaction",
            "description": "Resolve wallet_id for a transaction before writing it.",
        },
        arg_name="wallet_id",
        rows=wallets,
    )
    if not decision or decision.get("route") != "execute":
        return None
    value = decision.get("value")
    return str(value) if value else None


# ---------------------------------------------------------------------------
# MCP server & tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_wallet_summary(token: str = "", user_id: str = "") -> Dict[str, Any]:
    """Get a summary of the user's wallets and balances used to give advice on their financial health and spending habits based on their current financial situation.
    Mention the user's financial health score and provide insights on their spending habits based on the wallet summary. For example, if the user has a low financial health score, you can suggest ways to improve it, such as reducing expenses or increasing income. If the user has a high balance in one wallet but low balances in others, you can suggest redistributing funds to optimize their financial situation.

    Parameters:
    - token (str): bearer token for authentication
    Returns a dict with wallet summaries. Example:
    {
        "wallets": [
            {
                "walletId": "f0945566-9b1c-4c81-88e0-2da9d32806e6",
                "name": "Ước có học bổng",
                "walletType": "MANUAL",
                "balance": -501236033.00,
                "currency": "VND"
            },
            ...
        ],
        "totalBalance": -506874168.00
    }
    """
    resolved_user_id = _resolve_user_id(user_id)
    if not resolved_user_id:
        return _error_response(
            "Missing user_id. Set TRANSACTION_USER_ID or pass user_id in tool args."
        )

    try:
        wallets_cursor = _get_collection("wallets").find({"userId": resolved_user_id})
        wallets = [
            _normalize_document_id(wallet, "walletId")
            for wallet in wallets_cursor
        ]
        total_balance = 0
        for wallet in wallets:
            try:
                total_balance += float(wallet.get("balance", 0) or 0)
            except (TypeError, ValueError):
                continue

        return {
            "status": "success",
            "wallet_summary": _json_safe({
                "wallets": wallets,
                "totalBalance": total_balance,
            }),
            "financial_health_score": None,
        }
    except Exception as exc:
        return _error_response(f"Mongo request failed: {exc}")

@mcp.tool()
async def create_wallet(
    name: str,
    balance: str = "0",
    currency: str = "VND",
    token: str = "",
    user_id: str = "",
) -> Dict[str, Any]:
    """Create a manual wallet.

    Parameters:
    - name      (str): wallet name, e.g. "Cash", "MB Bank"
    - balance   (str, optional): starting balance, supports shorthand like "50k"
    - currency  (str, optional): wallet currency, defaults to "VND"
    - token     (str, optional): bearer token
    - user_id   (str, optional): user UUID
    """
    name = name.strip()
    if not name:
        return _error_response("Wallet name is required.")

    resolved_user_id = _resolve_user_id(user_id)
    if not resolved_user_id:
        return _error_response(
            "Missing user_id. Set TRANSACTION_USER_ID or pass user_id in tool args."
        )

    parsed_balance = _parse_amount(balance)
    logger.info(
        "create_wallet: name=%s balance=%s→%s currency=%s user_id=%s",
        name, balance, parsed_balance, currency, resolved_user_id,
    )

    payload = {
        "walletId": str(uuid.uuid4()),
        "userId": resolved_user_id,
        "name": name,
        "balance": parsed_balance if parsed_balance is not None else 0,
        "currency": (currency or "VND").strip().upper() or "VND",
        "walletType": "MANUAL",
        "active": True,
        "createdAt": datetime.now(UTC),
    }

    try:
        existing = _get_collection("wallets").find_one(
            {"userId": resolved_user_id, "name": name}
        )
        if existing:
            return _error_response("Wallet already exists.")

        result = _get_collection("wallets").insert_one(payload)
        created = dict(payload)
        created["id"] = str(result.inserted_id)
        return {
            "status": "success",
            "status_code": 201,
            "data": _json_safe(created),
        }
    except Exception as exc:
        return _error_response(f"Mongo request failed: {exc}")


@mcp.tool()
async def create_transaction(
    amount: str,
    description: str,
    wallet_id: str = "",
    wallet_name: str = "",
    request_text: str = "",
    category_name: str = "",
    category_id: str = "",
    transaction_date: str = "",
    image_url: str = "",
    token: str = "",
    user_id: str = "",
) -> Dict[str, Any]:
    """Create a financial expense transaction.

    Parameters:
    - amount      (str): transaction amount – supports shorthand like "50k", "20,000"
    - description (str): what the money was spent on, e.g. "ăn sáng", "cà phê"
    - wallet_id   (str, optional): wallet UUID to record the expense against
    - wallet_name (str, optional): exact wallet name to resolve before prompting
    - request_text (str, optional): raw user utterance used for wallet-name disambiguation
    - category_name (str, optional): category label for backend auto-matching
    - category_id (str, optional): category UUID when already resolved
    - token       (str, optional): bearer token (falls back to request context / env)
    - user_id     (str, optional): user UUID     (falls back to request context / env)
    """
    # --- validation ---
    if not str(amount).strip():
        return _error_response("Amount is required.")
    if not str(description).strip():
        return _error_response("Description is required.")

    resolved_user_id = _resolve_user_id(user_id)
    if not resolved_user_id:
        return _error_response(
            "Missing user_id. Set TRANSACTION_USER_ID or pass user_id in tool args."
        )

    parsed_amount = _parse_amount(amount)
    logger.info(
        "create_transaction: amount=%s→%s description=%s user_id=%s",
        amount, parsed_amount, description, resolved_user_id,
    )

    now = datetime.now(UTC)
    timestamp = transaction_date.strip() or now.isoformat()
    payload = {
        "id": str(uuid.uuid4()),
        "amount": parsed_amount if parsed_amount is not None else amount,
        "type": "EXPENSE",
        "category_id": str(category_id).strip() or None,
        "category_name": str(category_name).strip() or None,
        "description": description,
        "wallet_id": None,
        "user_id": resolved_user_id,
        "transaction_date": timestamp,
        "created_at": now,
        "updated_at": now,
        "image_url": image_url.strip() or None,
    }
    if str(wallet_id).strip():
        payload["wallet_id"] = str(wallet_id).strip()
    try:
        if payload["wallet_id"] and str(payload["wallet_id"]).startswith("$"):
            return _error_response("wallet_id placeholder was not resolved before tool execution.")

        if not payload["wallet_id"]:
            wallets = _wallet_candidates_for_user(resolved_user_id)
            if not wallets:
                return _error_response("No wallet found for this user.")
            if len(wallets) == 1:
                payload["wallet_id"] = wallets[0]["walletId"]
            else:
                resolved_wallet_id = await _select_wallet_id_with_llm(
                    wallets,
                    wallet_name=wallet_name,
                    text_hint=request_text or description,
                    description=description,
                )
                if resolved_wallet_id:
                    payload["wallet_id"] = resolved_wallet_id
                else:
                    return {
                        "status": "needs_input",
                        "prompt": "Please choose which wallet to use for this transaction:",
                        "selection_field": "wallet_id",
                        "value_field": "walletId",
                        "candidates": wallets,
                    }

        result = _get_collection("transactions").insert_one(payload)
        created = dict(payload)
        created.setdefault("mongo_id", str(result.inserted_id))
        return {
            "status": "success",
            "status_code": 201,
            "data": _json_safe(created),
        }
    except Exception as exc:
        return _error_response(f"Mongo request failed: {exc}")


def extract_user_id_from_token(token: str) -> Optional[str]:
    """Extract user id from JWT payload using the `sub` claim.

    This performs payload decoding only (no signature verification).
    """
    raw = _strip_bearer(token)
    if not raw:
        return None

    parts = raw.split(".")
    if len(parts) != 3:
        return None

    payload_b64 = parts[1]
    # JWT uses base64url without padding; restore padding for decoding.
    padding = "=" * (-len(payload_b64) % 4)
    try:
        payload_bytes = base64.urlsafe_b64decode(payload_b64 + padding)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        logger.exception("extract_user_id_from_token: failed to decode JWT payload")
        return None

    sub = payload.get("sub")
    if isinstance(sub, str) and sub.strip():
        return sub.strip()
    return None

@mcp.tool()
async def get_user_id(token: str = "") -> Dict[str, Any]:
    """Get the user ID associated with the provided token. Alg: HS256"""
    resolved_token = _resolve_token(token)
    if not resolved_token:
        return _error_response("No token provided.")

    extracted_user_id = extract_user_id_from_token(resolved_token)
    if not extracted_user_id:
        return _error_response("Could not extract user_id from token `sub` claim.")

    logger.info("get_user_id: extracted user_id=%s", extracted_user_id)
    return {"status": "success", "user_id": extracted_user_id}

@mcp.tool()
async def create_category(
    name: str,
    icon: str = "",
    budget_limit: str = "100000",
    period: str = "MONTHLY",
    token: str = "",
    user_id: str = "",
) -> Dict[str, Any]:
    """Create a financial category.

    Parameters:
    - name         (str): category name, e.g. "Ăn sáng", "Hóa đơn" (may be used for auto-categorization, so be specific)
    - icon         (str, optional): icon identifier, e.g. "coffee"
    - budget_limit (str, optional): budget cap – supports shorthand like "10k"
    - period       (str, optional): "MONTHLY" | "WEEKLY"
    - token        (str, optional): bearer token
    - user_id      (str, optional): user UUID
    """
    name = name.strip()
    if not name:
        return _error_response("Category name is required.")

    resolved_user_id = _resolve_user_id(user_id)
    logger.info("create_category: token=%s", _resolve_token(token))
    parsed_limit = _parse_amount(budget_limit)
    logger.info(
        "create_category: name=%s icon=%s budget_limit=%s→%s period=%s user_id=%s",
        name, icon, budget_limit, parsed_limit, period, resolved_user_id,
    )

    payload = {
        "userId": resolved_user_id,
        "name": name,
        "icon": icon,
        "budgetLimit": parsed_limit if parsed_limit is not None else budget_limit,
        "period": period,
    }

    try:
        existing = _get_collection("categories").find_one(
            {"userId": resolved_user_id, "name": name}
        )
        if existing:
            return _error_response("Category already exists.")

        payload["categoryId"] = str(uuid.uuid4())
        payload["createdAt"] = datetime.now(UTC)
        result = _get_collection("categories").insert_one(payload)
        created = dict(payload)
        created["id"] = str(result.inserted_id)
        return {
            "status": "success",
            "status_code": 201,
            "data": _json_safe(created),
        }
    except Exception as exc:
        return _error_response(f"Mongo request failed: {exc}")


if __name__ == "__main__":
    mcp.run()
