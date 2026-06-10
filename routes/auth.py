import logging

from fastapi import APIRouter, HTTPException

from auth_service import authenticate_user, register_user
from models.schemas import SignInRequest, SignInResponse, SignUpRequest, SignUpResponse


router = APIRouter(prefix="/api/auth", tags=["Auth"])
logger = logging.getLogger(__name__)


@router.post("/signup", response_model=SignUpResponse, summary="Create a local account")
def signup(request: SignUpRequest):
    try:
        return register_user(request.email, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("signup failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create account") from exc


@router.post("/signin", response_model=SignInResponse, summary="Sign in to a local account")
def signin(request: SignInRequest):
    try:
        return authenticate_user(request.email, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("signin failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to sign in") from exc
