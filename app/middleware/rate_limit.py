"""
Per-user rate limiting for the AI chat endpoint.

WHY RATE LIMIT:
  - Cost: every chat request may trigger a paid LLM inference.
  - Latency: unbounded concurrent LLM calls blow the <4s NFR.
  - Abuse: an anonymous bot would otherwise be unbounded.

WHY PER-USER:
  - A global limit lets one noisy user starve everyone else.
  - Keying by the verified token subject ("sub") gives fair independent
    budgets + a per-customer abuse signal.

WHY THE STORE IS AN INTERFACE:
  - In-memory is correct for ONE dev process, but with N uvicorn workers each
    worker would keep its OWN counter (real limit = N x configured).
    Redis keeps one shared counter. Isolating the store behind
    `RateLimitStore` makes that swap a one-line change in `check_rate_limit`.

IMPLEMENTATION:
  Sliding-window counter. Per user we keep a deque of request timestamps.
  On each request: drop timestamps older than the window, admit if fewer
  than `limit` remain, else deny -> RateLimitExceeded (429).
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict

from fastapi import Depends, Request

from app.core.client_ip import client_ip
from app.core.config import settings
from app.core.dependencies import get_current_user_id
from app.middleware.error_handler import RateLimitExceeded


# Tuning knobs — config-driven since Day 10 (RATE_LIMIT_PER_MINUTE in .env).
# The module-level names are kept so tests and the 429 detail message read one
# single source of truth. Values are read ONCE at import time — changing them
# requires a restart (same as any env-var configuration).
RATE_LIMIT_LIMIT: int = settings.rate_limit_per_minute  # requests per user per window
RATE_LIMIT_WINDOW_SECONDS: float = 60.0  # sliding-window length for 1 minute

logger = logging.getLogger(__name__)


class RateLimitResult:
    """
    Immutable result of a rate-limit check.

    Attributes:
        allowed: Whether the request may proceed.
        remaining: How many requests remain in the current window.
        limit: The configured per-window budget.
        window_seconds: The sliding-window length the limit applies to.
    """

    __slots__ = ("allowed", "remaining", "limit", "window_seconds")

    def __init__(
        self,
        allowed: bool,
        remaining: int,
        limit: int,
        window_seconds: float,
    ) -> None:
        self.allowed = allowed
        self.remaining = remaining
        self.limit = limit
        self.window_seconds = window_seconds


class RateLimitStore(ABC):
    """Storage contract for rate-limit counters.

    Production swaps `InMemoryRateLimitStore` for a Redis-backed store behind
    the SAME async `allow_request` signature — callers never change.
    """

    @abstractmethod
    async def allow_request(
        self,
        key: str,
        limit: int,
        window_seconds: float,
    ) -> RateLimitResult:
        """
        Record one request for `key` and report whether it may proceed.

        Implementations must be atomic and thread-safe.
        """
        raise NotImplementedError


class InMemoryRateLimitStore(RateLimitStore):
    """
    Sliding-window limiter held in process memory.

    Thread-safe via a lock (requests may arrive from the async loop's thread
    pool). Kept intentionally small; Redis is the production replacement.
    """

    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    async def allow_request(
        self,
        key: str,
        limit: int,
        window_seconds: float,
    ) -> RateLimitResult:
        """
        Record one in-memory request for `key` under the lock.

        Slides the window (drops expired timestamps), admits the request if
        the budget remains, and reclaims the key when a denied bucket drains.

        Returns:
            A RateLimitResult describing the decision and remaining budget.
        """
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            timestamps = self._hits.setdefault(key, deque())
            # 1. Slide the window: drop every timestamp older than `cutoff`.
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            # 2. Decide, then record (admitted requests append their time).
            allowed = len(timestamps) < limit
            if allowed:
                timestamps.append(now)
            else:
                # Denied + empty bucket -> reclaim the dict entry (bounds memory).
                if not timestamps:
                    self._hits.pop(key, None)
            remaining = limit - len(timestamps) if allowed else 0
        return RateLimitResult(
            allowed=allowed, remaining=remaining, limit=limit, window_seconds=window_seconds
        )

    def reset(self) -> None:
        """Clear all counters (used by tests between cases)."""
        with self._lock:
            self._hits.clear()


# Module-level singleton so every request shares ONE counter set per process.
rate_limit_store: RateLimitStore = InMemoryRateLimitStore()


async def check_rate_limit(
    request: Request,
    user_id: str = Depends(get_current_user_id),
) -> None:
    """
    FastAPI dependency: enforce the chat budgets before the handler runs.

    Because this depends on get_current_user_id, the resolution order is:
      1. missing / invalid bearer token  -> AuthenticationError (401)
      2. over the per-user budget        -> RateLimitExceeded   (429)
      3. over the per-client-IP budget   -> RateLimitExceeded   (429)
      4. otherwise                       -> None (handler proceeds)

    The per-user key is the VERIFIED token subject, so a client cannot dodge
    it by changing a header. The per-IP cap (CHAT_IP_RATE_PER_MINUTE) stops
    one client from multiplying its budget by rotating demo tokens; the IP
    is proxy-aware (app/core/client_ip.py).
    """
    result = await rate_limit_store.allow_request(
        key=f"user:{user_id}",
        limit=RATE_LIMIT_LIMIT,
        window_seconds=RATE_LIMIT_WINDOW_SECONDS,
    )
    if not result.allowed:
        raise RateLimitExceeded(
            detail=f"{RATE_LIMIT_LIMIT} requests per minute allowed"
        )

    ip = client_ip(request)
    ip_limit = settings.chat_ip_rate_per_minute
    ip_result = await rate_limit_store.allow_request(
        key=f"chat-ip:{ip}",
        limit=ip_limit,
        window_seconds=RATE_LIMIT_WINDOW_SECONDS,
    )
    if not ip_result.allowed:
        logger.warning("Rate limit exceeded for client IP %s", ip)
        raise RateLimitExceeded(
            detail=f"{ip_limit} requests per minute allowed per client IP"
        )
    return None