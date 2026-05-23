"""
ares.py
=======
Thin wrapper around the public ARES REST API for company enrichment.

Endpoint: https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/{ico}
Used to fetch authoritative firmographics (address, legal form, NACE) for a
given IČO, supplementing the dataset built from the RES open-data export.
"""

from __future__ import annotations

from typing import Optional

import requests

ARES_URL = "https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/{ico}"
TIMEOUT_S = 5


def lookup(ico: str) -> Optional[dict]:
    """Fetch and flatten ARES data for an 8-digit IČO. Returns None on failure."""
    ico = str(ico).strip().zfill(8)
    try:
        r = requests.get(ARES_URL.format(ico=ico), timeout=TIMEOUT_S)
        if r.status_code != 200:
            return None
        data = r.json()
    except (requests.RequestException, ValueError):
        return None

    addr = data.get("sidlo", {}) or {}
    nace_list = data.get("czNace") or []
    return {
        "ico": data.get("ico"),
        "name": data.get("obchodniJmeno"),
        "legal_form": data.get("pravniForma"),
        "registration_date": data.get("datumVzniku"),
        "termination_date": data.get("datumZaniku"),
        "address": addr.get("textovaAdresa"),
        "city": addr.get("nazevObce"),
        "region": addr.get("nazevKraje"),
        "zip": addr.get("psc"),
        "primary_nace": nace_list[0] if nace_list else None,
        "nace_count": len(nace_list),
        "active": data.get("datumZaniku") is None,
    }

