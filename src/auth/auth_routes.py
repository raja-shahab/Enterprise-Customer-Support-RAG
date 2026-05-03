"""
src/auth/routes.py  –  Authentication endpoints.

POST /auth/register   – create account
POST /auth/login      – get JWT token
GET  /auth/me         – current user info
GET  /auth/usage      – daily usage stats
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, field_validator

from src.auth.jwt_auth import (authenticate_user, create_token, create_user,
                                get_current_user, user_exists)
from src.auth.rate_limiter import get_usage

auth_router = APIRouter(prefix="/auth", tags=["Auth"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def strong_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str


# ── Routes ────────────────────────────────────────────────────────────────────

@auth_router.post("/register", response_model=TokenResponse, status_code=201)
async def register(req: RegisterRequest):
    """Create a new account and return a JWT token."""
    if user_exists(req.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Email already registered")
    create_user(req.email, req.password)
    token = create_token(req.email)
    return TokenResponse(access_token=token, email=req.email)


@auth_router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """Authenticate and return a JWT token."""
    email = authenticate_user(req.email, req.password)
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid email or password")
    token = create_token(email)
    return TokenResponse(access_token=token, email=email)


@auth_router.get("/me")
async def me(user: str = Depends(get_current_user)):
    """Return current user info + usage stats."""
    usage = await get_usage(user)
    return {"email": user, **usage}


@auth_router.get("/usage")
async def usage(user: str = Depends(get_current_user)):
    """Return daily usage stats for the current user."""
    return await get_usage(user)