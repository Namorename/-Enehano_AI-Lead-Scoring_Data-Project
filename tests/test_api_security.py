"""
Security tests for api.py.

Uses FastAPI's TestClient (backed by httpx). The model startup is patched out
so tests run without trained artifacts on disk.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    TestClient with model loading suppressed.
    The startup event calls _load_assets(); we patch it to a no-op so tests
    don't require trained pkl files.
    """
    import api as api_module
    from fastapi.testclient import TestClient

    with patch.object(api_module, "_load_assets"):
        with TestClient(api_module.app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture(scope="module")
def authed_client(monkeypatch_module):
    """
    TestClient with a fixed SCORING_API_KEY so auth-success paths are testable
    without needing the model loaded (auth is checked before model code runs).
    """
    import api as api_module
    from fastapi.testclient import TestClient

    with patch.object(api_module, "_load_assets"), \
         patch.object(api_module, "SCORING_API_KEY", "test-secret-key"):
        with TestClient(api_module.app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch shim (not needed directly but satisfies authed_client)."""
    return None


# ── /health — always public ───────────────────────────────────────────────────

def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_payload(client):
    body = client.get("/health").json()
    assert "status" in body
    assert "version" in body


# ── security headers ──────────────────────────────────────────────────────────

def test_security_headers_on_health(client):
    r = client.get("/health")
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert r.headers.get("x-xss-protection") == "1; mode=block"
    assert r.headers.get("referrer-policy") == "strict-origin-when-cross-origin"


# ── /metrics — requires API key ───────────────────────────────────────────────

def test_metrics_requires_key_when_env_set():
    """When SCORING_API_KEY is set, /metrics must return 401 without a key."""
    import api as api_module
    from fastapi.testclient import TestClient

    with patch.object(api_module, "_load_assets"), \
         patch.object(api_module, "SCORING_API_KEY", "secret"):
        with TestClient(api_module.app, raise_server_exceptions=False) as c:
            r = c.get("/metrics")
            assert r.status_code == 401


def test_metrics_public_when_no_key_configured(client):
    """/metrics is accessible without a key when SCORING_API_KEY is not set."""
    import api as api_module
    with patch.object(api_module, "SCORING_API_KEY", None):
        # METRICS is empty in test env → 503, not 401
        r = client.get("/metrics")
        assert r.status_code in (200, 503)
        assert r.status_code != 401


# ── /score — requires API key ─────────────────────────────────────────────────

def test_score_returns_401_without_key():
    import api as api_module
    from fastapi.testclient import TestClient

    with patch.object(api_module, "_load_assets"), \
         patch.object(api_module, "SCORING_API_KEY", "real-key"):
        with TestClient(api_module.app, raise_server_exceptions=False) as c:
            r = c.post("/score", json={})
            assert r.status_code == 401


def test_score_returns_401_with_wrong_key():
    import api as api_module
    from fastapi.testclient import TestClient

    with patch.object(api_module, "_load_assets"), \
         patch.object(api_module, "SCORING_API_KEY", "real-key"):
        with TestClient(api_module.app, raise_server_exceptions=False) as c:
            r = c.post("/score", json={}, headers={"x-api-key": "wrong-key"})
            assert r.status_code == 401


def test_score_passes_auth_with_correct_key():
    """Correct key gets past auth (model not loaded → 500, not 401/422)."""
    import api as api_module
    from fastapi.testclient import TestClient

    with patch.object(api_module, "_load_assets"), \
         patch.object(api_module, "SCORING_API_KEY", "test-secret"):
        with TestClient(api_module.app, raise_server_exceptions=False) as c:
            r = c.post("/score", json={}, headers={"x-api-key": "test-secret"})
            # Not 401 — auth succeeded; model code may error (500) or validate (422)
            assert r.status_code != 401


# ── /score/batch — requires API key ──────────────────────────────────────────

def test_batch_returns_401_without_key():
    import api as api_module
    from fastapi.testclient import TestClient

    with patch.object(api_module, "_load_assets"), \
         patch.object(api_module, "SCORING_API_KEY", "key"):
        with TestClient(api_module.app, raise_server_exceptions=False) as c:
            r = c.post("/score/batch", json={"leads": []})
            assert r.status_code == 401


# ── input validation — LeadInput bounds ───────────────────────────────────────

def test_email_open_rate_above_1_rejected(client):
    import api as api_module
    with patch.object(api_module, "SCORING_API_KEY", None):
        r = client.post("/score", json={"Email_Open_Rate__c": 1.5})
        assert r.status_code == 422


def test_negative_web_interactions_rejected(client):
    import api as api_module
    with patch.object(api_module, "SCORING_API_KEY", None):
        r = client.post("/score", json={"Web_Interactions__c": -5})
        assert r.status_code == 422


def test_demo_flag_above_1_rejected(client):
    import api as api_module
    with patch.object(api_module, "SCORING_API_KEY", None):
        r = client.post("/score", json={"Demo_Requested__c": 2})
        assert r.status_code == 422


def test_days_in_pipeline_above_max_rejected(client):
    import api as api_module
    with patch.object(api_module, "SCORING_API_KEY", None):
        r = client.post("/score", json={"Days_in_Pipeline__c": 99999})
        assert r.status_code == 422


def test_time_to_first_response_negative_rejected(client):
    import api as api_module
    with patch.object(api_module, "SCORING_API_KEY", None):
        r = client.post("/score", json={"Time_to_First_Response_h__c": -1.0})
        assert r.status_code == 422


def test_founding_year_below_min_rejected(client):
    import api as api_module
    with patch.object(api_module, "SCORING_API_KEY", None):
        r = client.post("/score", json={"Founding_Year": 1200})
        assert r.status_code == 422


def test_valid_defaults_pass_validation(client):
    """An empty payload (all defaults) passes Pydantic validation."""
    import api as api_module
    with patch.object(api_module, "SCORING_API_KEY", None):
        r = client.post("/score", json={})
        # 422 would mean validation failed — any other code is fine here
        assert r.status_code != 422


# ── batch size limit ──────────────────────────────────────────────────────────

def test_batch_over_500_rejected():
    import api as api_module
    from fastapi.testclient import TestClient

    with patch.object(api_module, "_load_assets"), \
         patch.object(api_module, "SCORING_API_KEY", None):
        with TestClient(api_module.app, raise_server_exceptions=False) as c:
            leads = [{}] * 501
            r = c.post("/score/batch", json={"leads": leads})
            assert r.status_code == 400
