"""
Client IP for per-IP rate limits (demo-token issuance and the chat IP cap).

With TRUST_PROXY=false (the default) the socket peer is used and any
X-Forwarded-For header is ignored.

With TRUST_PROXY=true the IP is the X-Forwarded-For entry TRUST_PROXY_HOPS
places from the right (default 1 = right-most, the entry the trusted proxy
appended). Entries further left are client-supplied and ignored, so a
client cannot pick its own rate-limit key by sending the header. If the
header has fewer entries than that, the socket peer is used.
"""

from fastapi import Request

from app.core.config import settings


def client_ip(request: Request) -> str:
    """Return the client IP used as the per-IP rate-limit key."""
    if settings.trust_proxy:
        joined = ",".join(request.headers.getlist("x-forwarded-for"))
        entries = [entry.strip() for entry in joined.split(",") if entry.strip()]
        hops = settings.trust_proxy_hops
        if len(entries) >= hops:
            return entries[-hops]
    return request.client.host if request.client else "unknown"
