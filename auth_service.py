import base64
import hashlib
import hmac
import json
import os
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any


def _get_mongo_uri() -> str:
    return (
        os.getenv("MONGO_URL", "").strip()
        or os.getenv("mongo_url", "").strip()
        or os.getenv("MONGO_URI", "").strip()
    )


AUTH_DB_NAME = os.getenv("AUTH_DB_NAME", "budget-tracker").strip() or "budget-tracker"
JWT_SECRET = os.getenv("JWT_SECRET", "local-dev-secret").strip() or "local-dev-secret"
JWT_EXP_DAYS = int(os.getenv("JWT_EXP_DAYS", "7"))

_mongo_client: Any | None = None


def _get_mongo_client():
    global _mongo_client
    if _mongo_client is None:
        mongo_uri = _get_mongo_uri()
        if not mongo_uri:
            raise RuntimeError("MONGO_URL/mongo_url/MONGO_URI is not set")
        from pymongo import MongoClient
        _mongo_client = MongoClient(mongo_uri)
    return _mongo_client


def get_users_collection():
    client = _get_mongo_client()
    db = client[AUTH_DB_NAME]
    return db["users"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _build_jwt(payload: dict[str, Any]) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    encoded_header = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    encoded_payload = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    signature = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{encoded_header}.{encoded_payload}.{_b64url_encode(signature)}"


def _hash_password(password: str, *, salt: str | None = None) -> str:
    password_bytes = password.encode("utf-8")
    salt_value = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password_bytes, salt_value.encode("utf-8"), 100_000)
    return f"{salt_value}${digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, _digest = stored_hash.split("$", 1)
    except ValueError:
        return False
    expected = _hash_password(password, salt=salt)
    return hmac.compare_digest(expected, stored_hash)


def _serialize_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "userId": user["userId"],
        "accountId": user["accountId"],
        "email": user["email"],
        "roles": user["roles"],
    }


def _issue_token(user: dict[str, Any]) -> str:
    now = _utcnow()
    payload = {
        "sub": user["userId"],
        "accountId": user["accountId"],
        "email": user["email"],
        "roles": user["roles"],
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=JWT_EXP_DAYS)).timestamp()),
    }
    return _build_jwt(payload)


def register_user(email: str, password: str) -> dict[str, Any]:
    users = get_users_collection()
    normalized_email = email.strip().lower()
    if users.find_one({"email": normalized_email}):
        raise ValueError("Email already exists")

    user = {
        "userId": str(uuid.uuid4()),
        "accountId": str(uuid.uuid4()),
        "email": normalized_email,
        "passwordHash": _hash_password(password),
        "roles": "ROLE_USER",
        "createdAt": _utcnow(),
    }
    users.insert_one(user)
    return {
        "token": _issue_token(user),
        "user": _serialize_user(user),
    }


def authenticate_user(email: str, password: str) -> dict[str, Any]:
    users = get_users_collection()
    normalized_email = email.strip().lower()
    user = users.find_one({"email": normalized_email})
    if not user or not _verify_password(password, str(user.get("passwordHash", ""))):
        raise ValueError("Invalid credentials")

    return {
        "token": _issue_token(user),
        "user": _serialize_user(user),
    }
