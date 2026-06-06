"""
Salesforce connection and shared field config.

Used by sf_sync.py and create_test_leads.py. Credentials are read from
environment variables (a .env file works too):

    SF_USER, SF_PASS, SF_TOKEN   login / password / security token
    SF_DOMAIN                    "login" for prod or Dev Edition, "test" for a sandbox
    SF_ICO_FIELD                 Lead field that holds the IČO (default ICO__c)
    SF_BEHAVIORAL_FIELDS         comma-separated Lead fields to feed the model
"""
from __future__ import annotations

import os
import sys


def enable_utf8_stdout() -> None:
    """Switch stdout to UTF-8 so Czech names (IČO, kraj) print on a Windows console."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


# Load .env first so the field names below pick up any overrides.
_load_env()

ICO_FIELD = os.getenv("SF_ICO_FIELD", "ICO__c")
BEHAVIORAL_FIELDS = [
    f.strip() for f in os.getenv("SF_BEHAVIORAL_FIELDS", "").split(",") if f.strip()
]


def connect():
    """Log in to Salesforce with the credentials from the environment."""
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
