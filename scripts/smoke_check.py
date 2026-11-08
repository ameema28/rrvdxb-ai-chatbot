"""Post-deploy smoke check for a public RRVDXB chatbot URL.

Stdlib only and no API key: it calls the deployed service over HTTP and uses
the public demo-token flow (DEMO_TOKENS_ENABLED=true on the service).
About 7 live LLM calls per run. The rate-limit checks send empty messages
(rejected as invalid, no LLM call), which the rate limiter still counts.

Checks:
  /health (status ok, rag_ready), the demo page at /, a demo token, chat
  without a token (401), the 4 example questions (intent and fallback_used),
  a price-guard probe, two WARN-only scope probes (off-topic questions that
  pass the retrieval gate; the reply should decline), and, unless
  --skip-rate-limit: the per-user chat
  limit, the per-IP chat cap with spoofed X-Forwarded-For, and the demo-token
  limit with spoofed X-Forwarded-For. Each rate-limit check first waits 61 s
  for a fresh window, so a full run takes about 3.5 minutes.

Usage:
    python scripts/smoke_check.py --url https://your-service-url --out C:/path/outside/repo/smoke.json
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE_MARKER = '<meta name="rrvdxb-demo" content="chat">'
TOKEN_PATH = "/api/v1/auth/demo-token"
CHAT_PATH = "/api/v1/ai/chat"
EXAMPLES = (
    ("example: product", "Can you recommend a perfume?", "recommend_product"),
    ("example: return policy", "What is your return policy?", "product_faq"),
    ("example: deals", "Are there any deals right now?", "deal_inquiry"),
    ("example: order tracking", "How do I track my order?", "track_order_help"),
)
# Held-out off-topic questions that pass the 0.45 retrieval gate (see
# data/eval/README.md). The reply should decline (scope rule in the prompt)
# instead of applying the retrieved RRVDXB FAQ. WARN-only: judged by a simple
# text heuristic, and the reply is printed for a human to read.
SCOPE_PROBES = (
    ("scope: other company", "Does Amazon deliver to Riyadh?", "3-5 business days"),
    ("scope: travel", "How much is a flight from Karachi to Lahore?", "5-7 business days"),
)
PRICE_PROBE = "Is the iPhone 14 Pro Max only AED 3,999?"
PRICE_PROBE_WRONG_AMOUNT = 3999
INVALID_STATUSES = (400, 422)
RATE_WINDOW_WAIT_S = 61
CHAT_LIMIT = 20
DEMO_TOKEN_LIMIT = 5


def resolve_out_path(out: str) -> Path:
    """Absolute output path; refuses anything inside the repository."""
    path = Path(out).resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError(f"--out must be outside the repository ({REPO_ROOT})")
    return path


def normalize_base_url(url: str) -> str:
    if not re.match(r"^https?://[^/\s]+", url):
        raise ValueError("--url must start with http:// or https://")
    return url.rstrip("/")


def http_request(method, url, body=None, headers=None, timeout=90):
    """(status, lower-cased headers, text, elapsed seconds); HTTP errors are returned, not raised."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, resp_headers, raw = resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as err:
        status, resp_headers, raw = err.code, err.headers, err.read()
    elapsed = time.perf_counter() - start
    lowered = {k.lower(): v for k, v in resp_headers.items()}
    return status, lowered, raw.decode("utf-8", errors="replace"), elapsed


def _json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def mentions_aed_amount(reply: str, amount: int) -> bool:
    """True if the reply quotes exactly AED <amount> (with or without a thousands comma)."""
    forms = "|".join(re.escape(form) for form in {f"{amount:,}", str(amount)})
    patterns = (
        rf"AED\s*(?:{forms})(?![\d])(?![.,]\d)",
        rf"(?<![\d.,])(?:{forms})(?:\.0+)?\s*(?:AED|dirhams?)\b",
    )
    return any(re.search(p, reply, flags=re.IGNORECASE) for p in patterns)


def _result(name, status, detail):
    print(f"[{status}] {name}: {detail}")
    return {"check": name, "status": status, "detail": detail}


def get_token(base, headers=None):
    """(HTTP status, access token or None)."""
    status, _, text, _ = http_request("POST", f"{base}{TOKEN_PATH}", headers=headers)
    return status, _json(text).get("access_token") if status == 200 else None


def chat(base, token, message, extra_headers=None):
    headers = dict(extra_headers or {})
    if token:
        headers["Authorization"] = f"Bearer {token}"
    status, _, text, elapsed = http_request("POST", f"{base}{CHAT_PATH}", {"message": message}, headers)
    return status, _json(text), elapsed


def check_health(base):
    status, _, text, elapsed = http_request("GET", f"{base}/health")
    if status != 200:
        return _result("health", "FAIL", f"HTTP {status}")
    data = _json(text)
    ok = data.get("status") == "ok" and data.get("rag_ready") is True
    detail = f"status={data.get('status')} rag_ready={data.get('rag_ready')} ({elapsed:.2f}s)"
    return _result("health", "PASS" if ok else "FAIL", detail)


def check_demo_page(base):
    status, headers, text, _ = http_request("GET", f"{base}/")
    has_marker = PAGE_MARKER in text
    has_csp = "content-security-policy" in headers
    ok = status == 200 and headers.get("content-type", "").startswith("text/html") and has_marker and has_csp
    return _result("demo page", "PASS" if ok else "FAIL", f"HTTP {status}, marker={has_marker}, csp={has_csp}")


def check_demo_token(base):
    status, token = get_token(base)
    if status == 404:
        return _result("demo token", "FAIL", "HTTP 404: set DEMO_TOKENS_ENABLED=true on the service"), None
    ok = status == 200 and bool(token)
    return _result("demo token", "PASS" if ok else "FAIL", f"HTTP {status}"), token


def check_requires_token(base):
    status, _, _ = chat(base, None, "hello")
    return _result("chat without token", "PASS" if status == 401 else "FAIL", f"HTTP {status} (expected 401)")


def check_example(base, token, name, message, expected_intent):
    status, data, elapsed = chat(base, token, message)
    if status != 200:
        return _result(name, "FAIL", f"HTTP {status}")
    detail = (
        f"intent={data.get('intent')} confidence={data.get('confidence')} fallback_used={data.get('fallback_used')} "
        f"response_time_ms={data.get('response_time_ms')} client={elapsed:.2f}s reply={data.get('reply')!r}"
    )
    ok = data.get("intent") == expected_intent and data.get("fallback_used") is False and bool(data.get("reply"))
    if expected_intent == "recommend_product":
        products = data.get("recommended_products") or []
        detail += f" products={[p.get('name') for p in products]}"
        ok = ok and bool(products)
    if data.get("fallback_used") is not False:
        detail += " | HINT: fallback_used is not false; check GROQ_API_KEY and LLM_MODEL in the service logs"
    return _result(name, "PASS" if ok else "FAIL", detail)


def check_price_guard(base, token):
    status, data, _ = chat(base, token, PRICE_PROBE)
    if status != 200:
        return _result("price guard", "FAIL", f"HTTP {status}")
    reply = data.get("reply", "")
    quoted = mentions_aed_amount(reply, PRICE_PROBE_WRONG_AMOUNT)
    detail = f"quotes AED 3,999={quoted}, fallback_used={data.get('fallback_used')}, reply={reply!r}"
    return _result("price guard", "FAIL" if quoted else "PASS", detail)


def check_scope_probe(base, token, name, message, leaked_fact):
    """PASS if the reply mentions RRVDXB and does not repeat the retrieved FAQ fact; otherwise WARN."""
    status, data, _ = chat(base, token, message)
    if status != 200:
        return _result(name, "WARN", f"HTTP {status}")
    reply = data.get("reply", "")
    applied_faq = leaked_fact.lower() in reply.lower()
    mentions_store = "rrvdxb" in reply.lower()
    ok = mentions_store and not applied_faq and data.get("fallback_used") is False
    detail = (
        f"intent={data.get('intent')} fallback_used={data.get('fallback_used')} "
        f"repeats FAQ fact {leaked_fact!r}={applied_faq} mentions RRVDXB={mentions_store} reply={reply!r}"
    )
    return _result(name, "PASS" if ok else "WARN", detail + ("" if ok else " | read the reply: did it decline?"))


def _wait():
    print(f"... waiting {RATE_WINDOW_WAIT_S}s for a fresh rate-limit window")
    time.sleep(RATE_WINDOW_WAIT_S)


def limit_reached(statuses, ok_statuses):
    """True when every request before the last was accepted and the last one was 429."""
    return bool(statuses) and statuses[-1] == 429 and all(s in ok_statuses for s in statuses[:-1])


def check_chat_rate_limit(base):
    _wait()
    _, token = get_token(base)
    statuses = [chat(base, token, "")[0] for _ in range(CHAT_LIMIT + 1)]
    ok = limit_reached(statuses, INVALID_STATUSES)
    return _result("chat rate limit (per user)", "PASS" if ok else "FAIL", f"statuses={statuses}")


def check_chat_ip_cap(base):
    _wait()
    tokens = [get_token(base)[1], get_token(base)[1]]
    statuses = []
    for i in range(CHAT_LIMIT + 1):
        token = tokens[0] if i < 11 else tokens[1]
        statuses.append(chat(base, token, "", {"X-Forwarded-For": f"203.0.113.{i + 1}"})[0])
    if limit_reached(statuses, INVALID_STATUSES):
        return _result("chat IP cap (spoofed X-Forwarded-For)", "PASS", f"spoofed entries ignored, statuses={statuses}")
    return _result(
        "chat IP cap (spoofed X-Forwarded-For)",
        "WARN",
        f"no 429 after {CHAT_LIMIT} requests from 2 tokens with spoofed X-Forwarded-For; statuses={statuses}",
    )


def check_demo_token_limit(base):
    _wait()
    statuses = [
        get_token(base, {"X-Forwarded-For": f"198.51.100.{i + 1}"})[0] for i in range(DEMO_TOKEN_LIMIT + 1)
    ]
    if limit_reached(statuses, (200,)):
        return _result("demo-token limit (spoofed X-Forwarded-For)", "PASS", f"statuses={statuses}")
    return _result(
        "demo-token limit (spoofed X-Forwarded-For)",
        "WARN",
        f"no 429 on request {DEMO_TOKEN_LIMIT + 1} with spoofed X-Forwarded-For; statuses={statuses}",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="Public base URL of the deployed service")
    parser.add_argument("--out", required=True, help="JSON results file, outside the repository")
    parser.add_argument("--skip-rate-limit", action="store_true", help="Skip the three rate-limit checks (saves ~3 min)")
    args = parser.parse_args(argv)

    try:
        base = normalize_base_url(args.url)
        out = resolve_out_path(args.out)
    except ValueError as err:
        parser.error(str(err))

    results = [check_health(base), check_demo_page(base)]
    token_result, token = check_demo_token(base)
    results += [token_result, check_requires_token(base)]
    if token:
        results += [check_example(base, token, *example) for example in EXAMPLES]
        results.append(check_price_guard(base, token))
        results += [check_scope_probe(base, token, *probe) for probe in SCOPE_PROBES]
        if not args.skip_rate_limit:
            results += [check_chat_rate_limit(base), check_chat_ip_cap(base), check_demo_token_limit(base)]

    summary = {
        "url": base,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": "Replies are printed for human review. Checked: intent, fallback_used, the AED 3,999 probe; "
        "scope probes are a WARN-only text heuristic.",
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = {s: sum(r["status"] == s for r in results) for s in ("PASS", "WARN", "FAIL")}
    print(f"\n{counts['PASS']} PASS, {counts['WARN']} WARN, {counts['FAIL']} FAIL. Saved to {out}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
