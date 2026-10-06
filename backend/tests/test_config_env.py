"""The environment formats docker-compose actually passes."""
from __future__ import annotations

import pytest

from app.config import Settings


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("http://localhost:3000", ["http://localhost:3000"]),
        ("https://a.example.com, https://b.example.com", ["https://a.example.com", "https://b.example.com"]),
        ('["https://a.example.com"]', ["https://a.example.com"]),
        ("", []),
    ],
)
def test_cors_origins_parse_from_the_environment(monkeypatch, raw, expected):
    monkeypatch.setenv("API_CORS_ORIGINS", raw)
    assert Settings().api_cors_origins == expected
