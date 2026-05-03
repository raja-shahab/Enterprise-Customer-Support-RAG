"""
src/auth/rate_limiter.py  –  Per-user daily request limit via Redis.

Key pattern:  rl:<email>:<YYYY-MM-DD>  →  integer count
TTL set to 25 hours so keys auto-expire after the day rolls over.
"""
from __future__ import annotations

from datetime import datetime, timezone

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, status

from src.auth.jwt_auth import get_current_user
from src.config import get_settings

DAILY_LIMIT = int(__import__("os").getenv("DAILY_REQUEST_LIMIT", "5"))


def _redis_url() -> str:
    s = get_settings()
    if s.redis_password:
        return f"redis://:{s.redis_password}@{s.redis_host}:{s.redis_port}/0"
    return f"redis://{s.redis_host}:{s.redis_port}/0"


async def check_rate_limit(user: str = Depends(get_current_user)) -> str:
    """
    FastAPI dependency.
    - Increments the user's daily counter.
    - Raises 429 if limit exceeded.
    - Returns the user email (so routes can use it too).
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"rl:{user}:{today}"

    r = aioredis.from_url(_redis_url(), decode_responses=True)
    async with r:
        count = await r.incr(key)
        if count == 1:
            # First request today — set TTL of 25 hours
            await r.expire(key, 25 * 3600)

        remaining = max(0, DAILY_LIMIT - count)

        if count > DAILY_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Daily limit reached",
                    "limit": DAILY_LIMIT,
                    "remaining": 0,
                    "resets": "midnight UTC",
                },
                headers={"X-RateLimit-Limit": str(DAILY_LIMIT),
                         "X-RateLimit-Remaining": "0"},
            )

    return user


async def get_usage(user: str) -> dict:
    """Returns current usage stats for a user."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"rl:{user}:{today}"
    r = aioredis.from_url(_redis_url(), decode_responses=True)
    async with r:
        count = int(await r.get(key) or 0)
    return {
        "used": count,
        "limit": DAILY_LIMIT,
        "remaining": max(0, DAILY_LIMIT - count),
        "resets": "midnight UTC",
    }