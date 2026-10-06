"""Access codes: closed when configured, open when not, never leaky."""
from __future__ import annotations

from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from app.config import get_settings
from app.main import app
from app.middleware import AccessCodeMiddleware


def _client(codes):
    async def ok(request):
        return PlainTextResponse("ok")

    inner = Starlette(routes=[Route("/thing", ok), Route("/health", ok), Route("/auth/status", ok)])
    inner.add_middleware(AccessCodeMiddleware, codes=codes)
    return TestClient(inner.build_middleware_stack())


def test_no_codes_configured_means_open():
    assert _client([]).get("/thing").status_code == 200


def test_missing_or_wrong_code_is_401():
    client = _client(["letmein-123"])
    assert client.get("/thing").status_code == 401
    assert client.get("/thing", headers={"X-Access-Code": "nope"}).status_code == 401


def test_right_code_passes_and_any_listed_code_works():
    client = _client(["alpha-code", "beta-code"])
    assert client.get("/thing", headers={"X-Access-Code": "beta-code"}).status_code == 200


def test_health_and_status_stay_open():
    client = _client(["letmein-123"])
    assert client.get("/health").status_code == 200
    assert client.get("/auth/status").status_code == 200


def test_401_body_does_not_reveal_the_code():
    body = _client(["super-secret-code"]).get("/thing").text
    assert "super-secret-code" not in body


def test_status_endpoint_reports_whether_a_code_is_needed(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(get_settings(), "access_codes", "")
    assert client.get("/auth/status").json() == {"required": False}
    monkeypatch.setattr(get_settings(), "access_codes", "abc")
    assert client.get("/auth/status").json() == {"required": True}
