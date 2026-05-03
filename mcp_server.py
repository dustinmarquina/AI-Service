import os
import re
from typing import Any, Dict, Optional

import requests
from fastmcp import FastMCP


TRANSACTION_API_BASE_URL = os.getenv("TRANSACTION_API_BASE_URL", "http://localhost:8080")
TRANSACTION_API_TOKEN = os.getenv("TRANSACTION_API_TOKEN", "")

mcp = FastMCP("ai-service")


def parse_simple_transaction_input(user_input: str) -> Optional[Dict[str, Any]]:
    text = user_input.strip()
    match = re.match(r"^(\d+(?:[\.,]\d+)?\s*[kK]?)\s+(.+)$", text)
    if not match:
        return None

    amount_text = match.group(1).replace(" ", "").lower()
    description = match.group(2).strip()

    try:
        if amount_text.endswith("k"):
            amount = float(amount_text[:-1]) * 1000
        else:
            amount = float(amount_text.replace(",", "."))
    except ValueError:
        return None

    if amount.is_integer():
        amount = int(amount)

    return {
        "intent": "transaction",
        "required_slots": ["amount", "description"],
        "filled_slots": {
            "amount": amount,
            "description": description,
        },
        "missing_slots": [],
        "next_action": "execute",
        "status": "ready",
    }


def build_transaction_payload(slots: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    return {
        "amount": slots.get("amount"),
        "description": slots.get("description"),
        "type": slots.get("type") or "EXPENSE",
        "userId": user_id,
    }


def submit_transaction(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if TRANSACTION_API_TOKEN:
        headers["Authorization"] = f"Bearer {TRANSACTION_API_TOKEN}"

    response = requests.post(
        f"{TRANSACTION_API_BASE_URL}/api/transactions",
        json=payload,
        headers=headers,
        timeout=5,
    )

    if response.status_code not in (200, 201):
        return {
            "status": "error",
            "message": f"Lỗi backend: {response.text}",
            "status_code": response.status_code,
        }

    try:
        data = response.json()
    except ValueError:
        data = response.text

    return {
        "status": "success",
        "message": "Transaction created successfully",
        "data": data,
    }


@mcp.tool()
def parse_transaction_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a short Vietnamese transaction message like '20k xôi gà'."""
    user_input = str(payload.get("message", ""))
    parsed = parse_simple_transaction_input(user_input)

    if not parsed:
        return {
            "status": "unparsed",
            "message": "Không nhận diện được mẫu giao dịch ngắn.",
        }

    return {
        "status": "success",
        "data": parsed,
    }


@mcp.tool()
def create_transaction(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create a transaction by calling the Spring Boot backend."""
    user_id = str(payload.get("userId", "")).strip()
    slots = payload.get("slots") or {}

    if not user_id:
        return {
            "status": "error",
            "message": "Missing userId",
        }

    transaction_payload = build_transaction_payload(slots, user_id)
    return submit_transaction(transaction_payload)


@mcp.tool()
def process_transaction_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Parse short transaction text and submit it in one step."""
    user_id = str(payload.get("userId", "")).strip()
    user_input = str(payload.get("message", ""))

    if not user_id:
        return {
            "status": "error",
            "message": "Missing userId",
        }

    parsed = parse_simple_transaction_input(user_input)
    if not parsed:
        return {
            "status": "unparsed",
            "message": "Không nhận diện được mẫu giao dịch ngắn.",
        }

    return submit_transaction(build_transaction_payload(parsed["filled_slots"], user_id))


@mcp.tool()
def get_status(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "status": "success",
        "message": "ai-service MCP server is running",
        "base_url": TRANSACTION_API_BASE_URL,
    }


if __name__ == "__main__":
    mcp.run()