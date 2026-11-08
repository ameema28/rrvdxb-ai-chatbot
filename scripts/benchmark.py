"""Sequential /api/v1/ai/chat latency benchmark against a public URL.

Stdlib only and no API key: it gets a demo token from the service
(DEMO_TOKENS_ENABLED=true) and renews it on a 401. One warm-up request
(excluded from the stats), then N sequential requests over the 4 demo
example questions. Requests are paced (default 3.5 s start to start) so one
client stays under 20 requests/minute, the per-user and per-IP chat limits.

Client times include the network between you and the service.
response_time_ms is the server's own time for the whole chat request.
Percentiles use the nearest-rank method (p95 of 20 values is the 19th).

Usage:
    python scripts/benchmark.py --url https://your-service-url -n 20 --out C:/path/outside/repo/bench.json
"""

import argparse
import json
import math
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKEN_PATH = "/api/v1/auth/demo-token"
CHAT_PATH = "/api/v1/ai/chat"
MIN_INTERVAL_S = 3.1
QUESTIONS = (
    "Can you recommend a perfume?",
    "What is your return policy?",
    "Are there any deals right now?",
    "How do I track my order?",
)


def resolve_out_path(out: str | None) -> Path:
    """Absolute output path (default: system temp dir); refuses anything inside the repository."""
    if out is None:
        out = str(Path(tempfile.gettempdir()) / f"rrvdxb_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json")
    path = Path(out).resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError(f"--out must be outside the repository ({REPO_ROOT})")
    return path


def validate_interval(interval: float, force: bool) -> float:
    if interval < 0:
        raise ValueError("--interval must be >= 0")
    if interval < MIN_INTERVAL_S and not force:
        raise ValueError(
            f"--interval below {MIN_INTERVAL_S}s would exceed 20 requests/minute; pass --force to override"
        )
    return interval


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def http_json(method: str, url: str, body=None, headers=None, timeout: float = 90) -> tuple:
    """(status or None, parsed JSON body or {}, error text or None)."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=dict(headers or {}))
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as err:
        return err.code, {}, None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
        return None, {}, f"{type(err).__name__}: {err}"


def get_token(base: str, timeout: float) -> str:
    status, body, error = http_json("POST", f"{base}{TOKEN_PATH}", timeout=timeout)
    if status != 200 or not body.get("access_token"):
        raise RuntimeError(f"could not get a demo token (status={status}, error={error})")
    return body["access_token"]


class Client:
    """Holds the demo token in memory and renews it once on a 401."""

    def __init__(self, base: str, timeout: float):
        self.base = base
        self.timeout = timeout
        self.token = get_token(base, timeout)
        self.renewals = 0

    def _post(self, message: str) -> tuple:
        headers = {"Authorization": f"Bearer {self.token}"}
        return http_json("POST", f"{self.base}{CHAT_PATH}", {"message": message}, headers, self.timeout)

    def one_request(self, message: str) -> dict:
        record = {"message": message}
        start = time.perf_counter()
        status, body, error = self._post(message)
        if status == 401:
            self.token = get_token(self.base, self.timeout)
            self.renewals += 1
            record["token_renewed"] = True
            start = time.perf_counter()
            status, body, error = self._post(message)
        record["client_ms"] = round((time.perf_counter() - start) * 1000)
        record["status"] = status
        if error:
            record["error"] = error
        if status == 200:
            record.update(
                fallback_used=body.get("fallback_used"),
                server_ms=body.get("response_time_ms"),
                intent=body.get("intent"),
            )
        return record


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("status") == 200]
    live = [r for r in ok if r.get("fallback_used") is False]
    return {
        "requests": len(records),
        "ok": len(ok),
        "rate_limited_429": sum(r.get("status") == 429 for r in records),
        "errors": sum(r.get("status") not in (200, 429) for r in records),
        "fallback_count": sum(r.get("fallback_used") is True for r in ok),
        "client_ms": stats([r["client_ms"] for r in ok]),
        "server_ms_live": stats([r["server_ms"] for r in live if isinstance(r.get("server_ms"), int)]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="Public base URL of the deployed service")
    parser.add_argument("-n", type=int, default=20, help="Measured requests (default 20)")
    parser.add_argument("--interval", type=float, default=3.5, help="Seconds between request starts (default 3.5)")
    parser.add_argument("--force", action="store_true", help=f"Allow --interval below {MIN_INTERVAL_S}s")
    parser.add_argument("--timeout", type=float, default=90, help="Per-request timeout in seconds")
    parser.add_argument("--out", help="JSON results file, outside the repository (default: system temp dir)")
    args = parser.parse_args(argv)

    try:
        if args.n < 1:
            raise ValueError("-n must be >= 1")
        if not args.url.startswith(("http://", "https://")):
            raise ValueError("--url must start with http:// or https://")
        interval = validate_interval(args.interval, args.force)
        out = resolve_out_path(args.out)
    except ValueError as err:
        parser.error(str(err))
    base = args.url.rstrip("/")

    try:
        client = Client(base, args.timeout)
    except RuntimeError as err:
        print(f"Stopped: {err}. Is DEMO_TOKENS_ENABLED=true on the service?")
        return 1

    print(f"Warm-up request to {base} (excluded from stats)")
    warmup = client.one_request(QUESTIONS[0])
    print(f"  warm-up: status={warmup.get('status')} client_ms={warmup['client_ms']}")

    records = []
    next_start = time.perf_counter() + interval
    for i in range(args.n):
        delay = next_start - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        next_start = time.perf_counter() + interval
        record = client.one_request(QUESTIONS[i % len(QUESTIONS)])
        records.append(record)
        print(
            f"  {i + 1:>2}/{args.n}: status={record.get('status')} client_ms={record['client_ms']} "
            f"server_ms={record.get('server_ms')} fallback={record.get('fallback_used')}"
        )

    summary = summarize(records)
    summary["token_renewals"] = client.renewals
    client_stats, server = summary["client_ms"], summary["server_ms_live"]
    print(
        f"\nClient end-to-end ms (includes your network), {client_stats['count']} OK: "
        f"p50={client_stats['p50']} p95={client_stats['p95']} max={client_stats['max']}"
    )
    print(f"Server ms (response_time_ms, live replies only): p50={server['p50']} p95={server['p95']} max={server['max']}")
    print(
        f"Fallbacks: {summary['fallback_count']}, 429s: {summary['rate_limited_429']}, "
        f"errors: {summary['errors']}"
    )

    result = {
        "url": base,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n": args.n,
        "interval_s": interval,
        "method": "sequential, one demo token, nearest-rank percentiles, warm-up excluded",
        "warmup": warmup,
        "summary": summary,
        "requests": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out}")
    if summary["ok"] != args.n:
        print("Not every measured request returned 200; the numbers above are incomplete.")
        if summary["rate_limited_429"]:
            print("Got 429s: wait 60 s after other scripts before benchmarking, or increase --interval.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
