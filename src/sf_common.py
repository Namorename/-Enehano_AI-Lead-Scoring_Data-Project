"""
sf_common.py
============
Shared Salesforce configuration and connection helpers, used by both
sf_sync.py (scoring sync) and create_test_leads.py (seeder).

Kept separate so the seeder does not have to import the ML stack (`api`), and so
the connection logic / field config live in exactly one place.

Env vars (optionally via a .env file — see .env.example):
  SF_USER, SF_PASS, SF_TOKEN   Salesforce username / password / security token
  SF_DOMAIN                    'login' (prod/Dev Edition, default) or 'test' (sandbox)
  SF_ICO_FIELD                 Lead field holding the IČO (default 'ICO__c')
  SF_BEHAVIORAL_FIELDS         Comma-separated Lead fields to feed the model
"""
from __future__ import annotations

import os
import sys


def enable_utf8_stdout() -> None:
    """Force UTF-8 console output. Czech firmographics (IČO, kraj names) contain
    non-ASCII characters that crash the default cp1252 Windows console."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - stdout may not be reconfigurable (e.g. piped)
        pass


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


# Load .env before reading config below, so env-driven field names resolve.
_load_env()

ICO_FIELD = os.getenv("SF_ICO_FIELD", "ICO__c")
BEHAVIORAL_FIELDS = [
    f.strip() for f in os.getenv("SF_BEHAVIORAL_FIELDS", "").split(",") if f.strip()
]


def connect():
    """Authenticate to Salesforce from env vars. Imported lazily so modules that
    only need the config constants don't require simple-salesforce installed."""
    from simple_salesforce import Salesforce

    missing = [v for v in ("SF_USER", "SF_PASS", "SF_TOKEN") if not os.getenv(v)]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")
    return Salesforce(
        username=os.environ["SF_USER"],
        password=os.environ["SF_PASS"],
        security_token=os.environ["SF_TOKEN"],
        domain=os.getenv("SF_DOMAIN", "login"),
    )
