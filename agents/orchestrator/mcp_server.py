import os
import re
import logging
import json
import base64
from typing import Any, Dict, Optional
from contextvars import ContextVar
from datetime import datetime

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


# ---------------------------------------------------------------------------
# MCP server & tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_wallet_summary(token: str = "", user_id: str = "") -> Dict[str, Any]:
    """Get a summary of the user's wallets and balances used to give advice on their financial health and spending habits based on their current financial situation.

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
    # `user_id` is accepted for executor compatibility; the token remains the primary auth input.
    try:
        wallet_summary = await _api_call(
            "GET", f"{REPORT_API_URL}/wallet-summary",
            bearer=_resolve_token(token),
        )
        month = datetime.now().month
        year = datetime.now().year
        date = f"{year}-{month:02d}" 
        financial_health = await _api_call(
            "GET", f"{REPORT_API_URL}/financial-health-score?month={date}",
            bearer=_resolve_token(token),
        )
        return {
            "status": "success",
            "wallet_summary": wallet_summary.get("data", {}),
            "financial_health_score": financial_health.get("data", {}).get("score"),
        }
    except APIError as exc:
        return _error_response(_auth_error_message(exc.status_code), status_code=exc.status_code, data=exc.data)
    except httpx.HTTPError as exc:
        return _error_response(f"Request failed: {exc}")

@mcp.tool()
async def create_transaction(
    amount: str,
    description: str,
    wallet_id: str = "",
    category_name: str = "",
    category_id: str = "",
    token: str = "",
    user_id: str = "",
) -> Dict[str, Any]:
    """Create a financial expense transaction.

    Parameters:
    - amount      (str): transaction amount – supports shorthand like "50k", "20,000"
    - description (str): what the money was spent on, e.g. "ăn sáng", "cà phê"
    - wallet_id   (str, optional): wallet UUID to record the expense against
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

    payload = {
        "userId": "af1eb20b-54e5-4910-b974-60151185e48f",
        "amount": parsed_amount if parsed_amount is not None else amount,
        "description": description,
        "type": "EXPENSE",
    }
    if str(wallet_id).strip():
        payload["walletId"] = str(wallet_id).strip()
    if str(category_name).strip():
        payload["categoryName"] = str(category_name).strip()
    if str(category_id).strip():
        payload["categoryId"] = str(category_id).strip()
    logger.info("create_transaction: token=%s", _resolve_token(token))
    try:
        return await _api_call(
            "POST", TRANSACTION_API_URL,
            bearer=_resolve_token(token),
            payload=payload,
        )
    #logging token
    
    except APIError as exc:
        return _error_response(_auth_error_message(exc.status_code), status_code=exc.status_code, data=exc.data)
    except httpx.HTTPError as exc:
        return _error_response(f"Request failed: {exc}")


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
        "userId": "af1eb20b-54e5-4910-b974-60151185e48f",
        "name": name,
        "icon": icon,
        "budgetLimit": parsed_limit if parsed_limit is not None else budget_limit,
        "period": period,
    }

    try:
        return await _api_call(
            "POST", CATEGORY_API_URL,
            bearer=_resolve_token(token),
            payload=payload,
        )
    except APIError as exc:
        return _error_response(_auth_error_message(exc.status_code), status_code=exc.status_code, data=exc.data)
    except httpx.HTTPError as exc:
        return _error_response(f"Request failed: {exc}")


if __name__ == "__main__":
    mcp.run()
