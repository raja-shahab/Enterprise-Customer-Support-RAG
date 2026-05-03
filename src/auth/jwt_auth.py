"""
src/auth/jwt_auth.py  –  JWT token creation, verification, password hashing.

Users are stored in Redis as  user:<email>  →  JSON{email, hashed_password}
No extra database needed — Redis is already in the stack.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# ── Config (override via .env) ────────────────────────────────────────────────
JWT_SECRET      = os.getenv("JWT_SECRET", "asa-super-secret-change-in-production-2024")
JWT_ALGORITHM   = "HS256"
JWT_EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "7"))

_bearer = HTTPBearer()


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── Token helpers ─────────────────────────────────────────────────────────────

def create_token(email: str) -> str:
    payload = {
        "sub": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# ── FastAPI dependency — extracts current user from Bearer token ──────────────

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    """Returns the user's email. Raises 401 if token is missing/invalid."""
    payload = decode_token(creds.credentials)
    return payload["sub"]


# ── Redis user store ──────────────────────────────────────────────────────────

def _get_redis():
    import redis as _redis
    from src.config import get_settings
    s = get_settings()
    return _redis.Redis(host=s.redis_host, port=s.redis_port,
                        password=s.redis_password or None,
                        decode_responses=True)


def user_exists(email: str) -> bool:
    r = _get_redis()
    return r.exists(f"user:{email}") == 1


def create_user(email: str, plain_password: str) -> None:
    r = _get_redis()
    data = {"email": email, "hashed_password": hash_password(plain_password)}
    r.set(f"user:{email}", json.dumps(data))


def authenticate_user(email: str, plain_password: str) -> Optional[str]:
    """Returns email on success, None on failure."""
    r = _get_redis()
    raw = r.get(f"user:{email}")
    if not raw:
        return None
    data = json.loads(raw)
    if not verify_password(plain_password, data["hashed_password"]):
        return None
    return email