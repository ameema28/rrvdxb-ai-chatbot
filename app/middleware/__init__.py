"""
Middleware package — cross-cutting HTTP concerns for the RRVDXB chatbot.

Day 8:
  - rate_limit.py    Per-user rate limiting for AI endpoints. In-memory store
                     today; the RateLimitStore interface makes a Redis-backed
                     store a drop-in swap later.
  - error_handler.py Centralized exception types + request handlers that keep
                     every error response in the same canonical JSON shape.
"""