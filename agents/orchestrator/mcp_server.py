import os
from typing import Any, Dict

import httpx
from fastmcp import FastMCP


TRANSACTION_API_URL = os.getenv(
    "TRANSACTION_API_URL",
    "https://api-budget-tracker.manportfolio.id.vn/api/v1/transactions",
)
DEFAULT_USER_ID = os.getenv("TRANSACTION_USER_ID", "af1eb20b-54e5-4910-b974-60151185e48f")
DEFAULT_TOKEN = os.getenv("TRANSACTION_API_TOKEN", "")

mcp = FastMCP("transaction-service")




@mcp.tool()
async def create_transaction(
    amount: str,
    description: str,
    token: str = "",
    user_id: str = "",
) -> Dict[str, Any]:
    """
Create a financial expense transaction.

Parameters:
- amount (string): transaction amount (e.g. "50k", "20000")
- description (string): what the money was spent on (e.g. "ăn sáng", "cà phê")

Behavior:
- Converts amount like "50k" → 50000
- Sends request to backend API

Examples:
- "50k ăn sáng"
- "100k cà phê"
"""
    if not amount or not str(amount).strip():
        return {
            "status": "error",
            "message": "Amount is required.",
        }

    if not description or not str(description).strip():
        return {
            "status": "error",
            "message": "Description is required.",
        }

    payload = {
        "amount": amount,
        "description": str(description).strip(),
        "type": "EXPENSE",
        "userId": user_id.strip() or DEFAULT_USER_ID,
    }

    headers = {
        "Content-Type": "application/json",
    }
    bearer = token.strip() or DEFAULT_TOKEN
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(
                TRANSACTION_API_URL,
                json=payload,
                headers=headers,
            )

        try:
            data = res.json()
        except ValueError:
            data = {"raw": res.text}

        if res.status_code >= 400:
            return {
                "status": "error",
                "status_code": res.status_code,
                "message": "Backend returned an error",
                "data": data,
            }

        return {
            "status": "success",
            "status_code": res.status_code,
            "data": data,
        }
    except httpx.HTTPError as exc:
        return {
            "status": "error",
            "message": f"Request failed: {exc}",
        }


if __name__ == "__main__":
    mcp.run()
    
