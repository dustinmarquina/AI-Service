import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException


ROOT = Path(__file__).resolve().parents[1]


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("models", ROOT / "models")
_ensure_package("routes", ROOT / "routes")

SCHEMAS_MODULE = _load_module("models.schemas", ROOT / "models" / "schemas.py")
AUTH_MODULE = _load_module("routes.auth", ROOT / "routes" / "auth.py")

SignUpRequest = SCHEMAS_MODULE.SignUpRequest
SignInRequest = SCHEMAS_MODULE.SignInRequest


class AuthRouteTests(unittest.TestCase):
    def test_signup_returns_token_and_user_profile(self):
        fake_result = {
            "token": "jwt-token",
            "user": {
                "userId": "user-1",
                "accountId": "account-1",
                "email": "demo@example.com",
                "roles": "ROLE_USER",
            },
        }

        with patch.object(AUTH_MODULE, "register_user", return_value=fake_result):
            response = AUTH_MODULE.signup(
                SignUpRequest(email="demo@example.com", password="secret123")
            )

        self.assertEqual(response["token"], "jwt-token")
        self.assertEqual(response["user"]["email"], "demo@example.com")

    def test_signup_maps_duplicate_email_to_http_400(self):
        with patch.object(
            AUTH_MODULE,
            "register_user",
            side_effect=ValueError("Email already exists"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                AUTH_MODULE.signup(
                    SignUpRequest(email="demo@example.com", password="secret123")
                )

        self.assertEqual(ctx.exception.status_code, 400)

    def test_signin_returns_token_for_valid_credentials(self):
        fake_result = {
            "token": "jwt-token",
            "user": {
                "userId": "user-1",
                "accountId": "account-1",
                "email": "demo@example.com",
                "roles": "ROLE_USER",
            },
        }

        with patch.object(AUTH_MODULE, "authenticate_user", return_value=fake_result):
            response = AUTH_MODULE.signin(
                SignInRequest(email="demo@example.com", password="secret123")
            )

        self.assertEqual(response["token"], "jwt-token")
        self.assertEqual(response["user"]["userId"], "user-1")

    def test_signin_maps_invalid_credentials_to_http_401(self):
        with patch.object(
            AUTH_MODULE,
            "authenticate_user",
            side_effect=ValueError("Invalid credentials"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                AUTH_MODULE.signin(
                    SignInRequest(email="demo@example.com", password="bad-pass")
                )

        self.assertEqual(ctx.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
