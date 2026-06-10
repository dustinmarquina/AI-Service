import base64
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUTH_SERVICE = _load_module("auth_service", ROOT / "auth_service.py")


class _FakeCollection:
    def __init__(self):
        self.docs = []

    def find_one(self, query):
        email = query.get("email")
        return next((doc for doc in self.docs if doc.get("email") == email), None)

    def insert_one(self, document):
        self.docs.append(dict(document))
        return type("InsertResult", (), {"inserted_id": "1"})()


def _decode_payload(token: str) -> dict:
    payload_b64 = token.split(".")[1]
    padding = "=" * (-len(payload_b64) % 4)
    return json.loads(base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8"))


class AuthServiceTests(unittest.TestCase):
    def test_register_user_returns_spring_like_jwt_claims(self):
        collection = _FakeCollection()

        with patch.object(AUTH_SERVICE, "get_users_collection", return_value=collection):
            result = AUTH_SERVICE.register_user("demo@example.com", "secret123")

        payload = _decode_payload(result["token"])
        self.assertEqual(payload["sub"], result["user"]["userId"])
        self.assertEqual(payload["accountId"], result["user"]["accountId"])
        self.assertEqual(payload["email"], "demo@example.com")
        self.assertEqual(payload["roles"], "ROLE_USER")
        self.assertIn("iat", payload)
        self.assertIn("exp", payload)

    def test_authenticate_user_rejects_wrong_password(self):
        collection = _FakeCollection()
        password_hash = AUTH_SERVICE._hash_password("secret123")
        collection.docs.append(
            {
                "userId": "user-1",
                "accountId": "account-1",
                "email": "demo@example.com",
                "passwordHash": password_hash,
                "roles": "ROLE_USER",
            }
        )

        with patch.object(AUTH_SERVICE, "get_users_collection", return_value=collection):
            with self.assertRaises(ValueError):
                AUTH_SERVICE.authenticate_user("demo@example.com", "wrongpass")


if __name__ == "__main__":
    unittest.main()
