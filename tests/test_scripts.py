"""
scripts/smoke_check.py and scripts/benchmark.py, with the HTTP layer faked
(no network, no API key).
"""

import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load("smoke_check")
bench = _load("benchmark")

PAGE = '<html><head><meta name="rrvdxb-demo" content="chat"></head></html>'
EXAMPLE_INTENTS = {message: intent for _, message, intent in smoke.EXAMPLES}


# --------------------------------------------------------------------------
# Shared guards
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", [smoke, bench])
def test_out_path_inside_repo_is_refused(module, tmp_path):
    with pytest.raises(ValueError):
        module.resolve_out_path(str(ROOT / "out.json"))
    with pytest.raises(ValueError):
        module.resolve_out_path(str(ROOT / "scripts" / "out.json"))
    assert module.resolve_out_path(str(tmp_path / "out.json")) == (tmp_path / "out.json").resolve()


def test_benchmark_default_out_is_the_temp_dir():
    path = bench.resolve_out_path(None)
    assert Path(tempfile.gettempdir()).resolve() in path.parents


@pytest.mark.parametrize("url", ["ftp://x", "example.com", "http://", ""])
def test_smoke_rejects_bad_urls(url):
    with pytest.raises(ValueError):
        smoke.normalize_base_url(url)


def test_smoke_normalizes_trailing_slash():
    assert smoke.normalize_base_url("https://svc.example.run.app/") == "https://svc.example.run.app"


# --------------------------------------------------------------------------
# smoke_check helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reply, expected",
    [
        ("Yes, it's AED 3,999 today.", True),
        ("It costs 3999 AED.", True),
        ("Only 3,999 dirhams!", True),
        ("aed3999 is the price", True),
        ("It is AED 4,699.", False),
        ("It is AED 3,999.50.", False),
        ("It is AED 13,999.", False),
        ("Our free-shipping threshold is AED 489.", False),
    ],
)
def test_price_probe_matcher(reply, expected):
    assert smoke.mentions_aed_amount(reply, 3999) is expected


def test_limit_reached():
    assert smoke.limit_reached([422] * 20 + [429], (400, 422))
    assert smoke.limit_reached([400] * 20 + [429], (400, 422))
    assert not smoke.limit_reached([422] * 21, (400, 422))
    assert not smoke.limit_reached([422, 500, 429], (400, 422))
    assert not smoke.limit_reached([], (200,))


class FakeService:
    """Minimal stand-in for the deployed API, driven through smoke.http_request."""

    def __init__(self, fallback=False, rate_limits=True, quote_wrong_price=False, leak_scope=False):
        self.fallback = fallback
        self.rate_limits = rate_limits
        self.quote_wrong_price = quote_wrong_price
        self.leak_scope = leak_scope
        self.tokens = 0
        self.new_window()

    def new_window(self):
        """A fresh one-minute rate-limit window (what smoke._wait waits for)."""
        self.token_requests = 0
        self.per_token = {}
        self.chats = 0

    def __call__(self, method, url, body=None, headers=None, timeout=90):
        headers = headers or {}
        path = url.split("://", 1)[1].split("/", 1)[1]
        path = "/" + path
        if path == "/health":
            return 200, {}, json.dumps({"status": "ok", "service": "x", "rag_ready": True}), 0.01
        if path == "/":
            return 200, {"content-type": "text/html", "content-security-policy": "x"}, PAGE, 0.01
        if path == smoke.TOKEN_PATH:
            self.token_requests += 1
            if self.rate_limits and self.token_requests > smoke.DEMO_TOKEN_LIMIT:
                return 429, {}, "{}", 0.01
            self.tokens += 1
            return 200, {}, json.dumps({"access_token": f"tok-{self.tokens}"}), 0.01
        if path == smoke.CHAT_PATH:
            auth = headers.get("Authorization")
            if not auth:
                return 401, {}, "{}", 0.01
            message = body["message"]
            if not message:
                count = self.per_token.get(auth, 0) + 1
                self.per_token[auth] = count
                self.chats += 1
                if self.rate_limits and (count > 20 or self.chats > 20):
                    return 429, {}, "{}", 0.01
                return 400, {}, "{}", 0.01
            intent = EXAMPLE_INTENTS.get(message, "general_chat")
            reply = "It is AED 3,999." if (self.quote_wrong_price and message == smoke.PRICE_PROBE) else "Sure!"
            for _, probe, fact in smoke.SCOPE_PROBES:
                if message == probe:
                    reply = (f"Yes, delivery takes {fact}." if self.leak_scope
                             else "Sorry, I can only help with RRVDXB shopping.")
            data = {
                "reply": reply,
                "intent": intent,
                "confidence": 1.0,
                "fallback_used": self.fallback,
                "response_time_ms": 120,
                "recommended_products": [{"name": "Ladies Leather Handbag"}] if intent == "recommend_product" else None,
            }
            return 200, {}, json.dumps(data), 0.2
        return 404, {}, "{}", 0.01


def test_smoke_main_passes_against_a_healthy_service(monkeypatch, tmp_path):
    service = FakeService()
    monkeypatch.setattr(smoke, "http_request", service)
    monkeypatch.setattr(smoke, "_wait", service.new_window)
    out = tmp_path / "smoke.json"
    assert smoke.main(["--url", "https://svc.example", "--out", str(out)]) == 0
    results = json.loads(out.read_text(encoding="utf-8"))["results"]
    assert [r["status"] for r in results] == ["PASS"] * len(results)
    assert len(results) == 4 + len(smoke.EXAMPLES) + 1 + len(smoke.SCOPE_PROBES) + 3


def test_scope_probes_only_warn_when_the_reply_applies_the_faq(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "http_request", FakeService(leak_scope=True))
    out = tmp_path / "smoke.json"
    assert smoke.main(["--url", "https://svc.example", "--out", str(out), "--skip-rate-limit"]) == 0
    results = {r["check"]: r for r in json.loads(out.read_text(encoding="utf-8"))["results"]}
    for name, _, fact in smoke.SCOPE_PROBES:
        assert results[name]["status"] == "WARN"
        assert repr(fact) in results[name]["detail"]
    assert {r["status"] for r in results.values()} == {"PASS", "WARN"}


def test_smoke_fails_on_fallback_and_wrong_price(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "http_request", FakeService(fallback=True, quote_wrong_price=True))
    out = tmp_path / "smoke.json"
    assert smoke.main(["--url", "https://svc.example", "--out", str(out), "--skip-rate-limit"]) == 1
    results = {r["check"]: r for r in json.loads(out.read_text(encoding="utf-8"))["results"]}
    assert results["example: return policy"]["status"] == "FAIL"
    assert "HINT" in results["example: return policy"]["detail"]
    assert results["price guard"]["status"] == "FAIL"


def test_smoke_warns_when_spoofed_headers_bypass_the_limits(monkeypatch, tmp_path):
    service = FakeService(rate_limits=False)
    monkeypatch.setattr(smoke, "http_request", service)
    monkeypatch.setattr(smoke, "_wait", service.new_window)
    out = tmp_path / "smoke.json"
    assert smoke.main(["--url", "https://svc.example", "--out", str(out)]) == 1
    results = {r["check"]: r["status"] for r in json.loads(out.read_text(encoding="utf-8"))["results"]}
    assert results["chat rate limit (per user)"] == "FAIL"
    assert results["chat IP cap (spoofed X-Forwarded-For)"] == "WARN"
    assert results["demo-token limit (spoofed X-Forwarded-For)"] == "WARN"


def test_smoke_reports_disabled_demo_tokens(monkeypatch, tmp_path):
    def service(method, url, body=None, headers=None, timeout=90):
        if url.endswith(smoke.TOKEN_PATH):
            return 404, {}, "{}", 0.01
        return FakeService()(method, url, body, headers, timeout)

    monkeypatch.setattr(smoke, "http_request", service)
    out = tmp_path / "smoke.json"
    assert smoke.main(["--url", "https://svc.example", "--out", str(out)]) == 1
    results = json.loads(out.read_text(encoding="utf-8"))["results"]
    assert "DEMO_TOKENS_ENABLED" in results[2]["detail"]
    assert all(r["check"] != smoke.SCOPE_PROBES[0][0] for r in results)
    assert len(results) == 4


def test_smoke_refuses_out_inside_repo():
    with pytest.raises(SystemExit):
        smoke.main(["--url", "https://svc.example", "--out", str(ROOT / "smoke.json")])


# --------------------------------------------------------------------------
# benchmark helpers
# --------------------------------------------------------------------------

def test_percentile_nearest_rank():
    values = list(range(1, 21))
    assert bench.percentile(values, 50) == 10
    assert bench.percentile(values, 95) == 19
    assert bench.percentile([7], 95) == 7
    assert bench.percentile([], 50) is None
    assert bench.stats([]) == {"count": 0, "p50": None, "p95": None, "max": None}


def test_validate_interval():
    assert bench.validate_interval(3.5, False) == 3.5
    with pytest.raises(ValueError):
        bench.validate_interval(1.0, False)
    assert bench.validate_interval(1.0, True) == 1.0
    with pytest.raises(ValueError):
        bench.validate_interval(-1, True)


def test_summarize_counts():
    records = [
        {"status": 200, "fallback_used": False, "server_ms": 100, "client_ms": 300},
        {"status": 200, "fallback_used": True, "server_ms": 3000, "client_ms": 3100},
        {"status": 429, "client_ms": 20},
        {"status": None, "client_ms": 5, "error": "URLError"},
    ]
    summary = bench.summarize(records)
    assert summary["ok"] == 2
    assert summary["rate_limited_429"] == 1
    assert summary["errors"] == 1
    assert summary["fallback_count"] == 1
    assert summary["client_ms"]["count"] == 2
    assert summary["server_ms_live"] == {"count": 1, "p50": 100, "p95": 100, "max": 100}


class FakeJson:
    def __init__(self, expire_first_token=False, token_status=200):
        self.issued = 0
        self.expire_first_token = expire_first_token
        self.token_status = token_status
        self.calls = []

    def __call__(self, method, url, body=None, headers=None, timeout=90):
        self.calls.append(url)
        if url.endswith(bench.TOKEN_PATH):
            if self.token_status != 200:
                return self.token_status, {}, None
            self.issued += 1
            return 200, {"access_token": f"tok-{self.issued}"}, None
        if self.expire_first_token and headers["Authorization"] == "Bearer tok-1":
            return 401, {}, None
        return 200, {"reply": "ok", "fallback_used": False, "response_time_ms": 150, "intent": "x"}, None


def test_client_renews_the_token_once_on_401(monkeypatch):
    fake = FakeJson(expire_first_token=True)
    monkeypatch.setattr(bench, "http_json", fake)
    client = bench.Client("https://svc.example", 5)
    record = client.one_request("hello")
    assert record["status"] == 200
    assert record["token_renewed"] is True
    assert client.renewals == 1
    assert client.token == "tok-2"
    assert client.one_request("again").get("token_renewed") is None


def test_benchmark_main_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "http_json", FakeJson())
    out = tmp_path / "bench.json"
    code = bench.main(["--url", "https://svc.example", "-n", "3", "--interval", "0", "--force", "--out", str(out)])
    assert code == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["summary"]["ok"] == 3
    assert result["summary"]["server_ms_live"]["p50"] == 150
    assert [r["message"] for r in result["requests"]] == list(bench.QUESTIONS[:3])


def test_benchmark_stops_when_demo_tokens_are_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "http_json", FakeJson(token_status=404))
    code = bench.main(["--url", "https://svc.example", "-n", "1", "--out", str(tmp_path / "b.json")])
    assert code == 1


@pytest.mark.parametrize(
    "args",
    [
        ["--url", "https://svc.example", "-n", "0"],
        ["--url", "svc.example"],
        ["--url", "https://svc.example", "--interval", "1"],
        ["--url", "https://svc.example", "--out", str(ROOT / "b.json")],
    ],
)
def test_benchmark_rejects_bad_arguments(args):
    with pytest.raises(SystemExit):
        bench.main(args)
