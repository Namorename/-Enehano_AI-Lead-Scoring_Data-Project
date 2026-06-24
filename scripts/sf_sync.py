"""
scripts/sf_sync.py
==================
Nightly Salesforce ↔ scoring API synchronisation.

What this script does
---------------------
1. PULL   — query Salesforce for all open Leads (IsConverted = false).
2. SCORE  — send them to POST /score/batch via the local API.
3. PUSH   — upsert AI_Score__c, AI_Segment__c, AI_Top_Driver__c back to Salesforce.
4. COLLECT — query recently closed Leads/Opportunities and POST each real
             outcome to POST /feedback so the model can learn from them.

This is the script that turns the project from a demo into a product.
Run it as a nightly cron job:

    0 2 * * * /usr/bin/python3 /app/scripts/sf_sync.py >> /var/log/sf_sync.log 2>&1

Required environment variables (put in .env or export before running):
    SF_USERNAME         Salesforce username
    SF_PASSWORD         Salesforce password
    SF_SECURITY_TOKEN   Salesforce security token (from Settings > My Personal Info)
    SF_DOMAIN           "login" for production, "test" for sandbox (default: login)
    API_URL             Base URL of the scoring API (default: http://localhost:8000)
    BATCH_SIZE          Leads per /score/batch call (default: 200, max 500)
    DAYS_BACK_OUTCOMES  How many days of closes to pull for /feedback (default: 7)

Install dependency (already in requirements.txt):
    pip install simple-salesforce python-dotenv
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sf_sync")

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # python-dotenv is optional at runtime; env vars can be set directly

# ── Configuration ─────────────────────────────────────────────────────────────
SF_USERNAME       = os.environ.get("SF_USERNAME", "")
SF_PASSWORD       = os.environ.get("SF_PASSWORD", "")
SF_SECURITY_TOKEN = os.environ.get("SF_SECURITY_TOKEN", "")
SF_DOMAIN         = os.environ.get("SF_DOMAIN", "login")
API_URL           = os.environ.get("API_URL", "http://localhost:8000")
BATCH_SIZE        = int(os.environ.get("BATCH_SIZE", "200"))
DAYS_BACK         = int(os.environ.get("DAYS_BACK_OUTCOMES", "7"))


# ── Salesforce helpers ────────────────────────────────────────────────────────

def _sf_client():
    """Return an authenticated simple_salesforce.Salesforce instance."""
    try:
        from simple_salesforce import Salesforce
    except ImportError:
        log.error("simple-salesforce not installed. Run: pip install simple-salesforce")
        sys.exit(1)

    if not all([SF_USERNAME, SF_PASSWORD, SF_SECURITY_TOKEN]):
        log.error(
            "Missing Salesforce credentials. Set SF_USERNAME, SF_PASSWORD, "
            "SF_SECURITY_TOKEN in .env or environment."
        )
        sys.exit(1)

    log.info("Authenticating to Salesforce (%s)...", SF_DOMAIN)
    return Salesforce(
        username=SF_USERNAME,
        password=SF_PASSWORD,
        security_token=SF_SECURITY_TOKEN,
        domain=SF_DOMAIN,
    )


def _query_open_leads(sf) -> list[dict]:
    """Return all open (unconverted) leads with the fields needed for scoring."""
    soql = """
        SELECT
            Id, ICO__c,
            Industry, Rating, LeadSource,
            NumberOfEmployees, AnnualRevenue,
            Time_to_First_Response_h__c,
            Web_Interactions__c, Email_Opens__c, Email_Clicks__c,
            Emails_Sent__c, Email_Open_Rate__c, Form_Submissions__c,
            Demo_Requested__c, Calls_Made__c, Meetings_Held__c,
            Proposal_Sent__c, Content_Downloads__c,
            Chatbot_Interactions__c, LinkedIn_Viewed__c,
            Days_Since_Last_Activity__c, Days_in_Pipeline__c
        FROM Lead
        WHERE IsConverted = false
        AND ICO__c != null
        ORDER BY CreatedDate DESC
    """
    result = sf.query_all(soql)
    leads = result.get("records", [])
    log.info("Fetched %d open leads from Salesforce.", len(leads))
    return leads


def _query_recent_closes(sf) -> list[dict]:
    """Return leads closed in the last DAYS_BACK days for /feedback."""
    since = (datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    soql = f"""
        SELECT Id, IsConverted, ConvertedOpportunityId, LastModifiedDate
        FROM Lead
        WHERE LastModifiedDate >= {since}
        AND IsConverted = true
    """
    leads = sf.query_all(soql).get("records", [])

    # Also fetch won opportunities.
    soql_opp = f"""
        SELECT Id, ConvertedLeadId, IsClosed, IsWon, CloseDate
        FROM Opportunity
        WHERE LastModifiedDate >= {since}
        AND IsClosed = true
    """
    opps = {
        r["ConvertedLeadId"]: r
        for r in sf.query_all(soql_opp).get("records", [])
        if r.get("ConvertedLeadId")
    }

    closes = []
    for lead in leads:
        opp = opps.get(lead["Id"])
        closes.append({
            "lead_id":    lead["Id"],
            "converted":  True,
            "closed_won": bool(opp and opp.get("IsWon")),
            "closed_date": (opp or {}).get("CloseDate"),
        })
    log.info("Found %d recently closed leads.", len(closes))
    return closes


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _sf_lead_to_payload(lead: dict) -> dict:
    """Map a Salesforce Lead record to an EnrichLeadInput payload."""
    def _int(v, default=0):
        try: return int(float(v or default))
        except (TypeError, ValueError): return default

    def _float(v, default=0.0):
        try: return float(v or default)
        except (TypeError, ValueError): return default

    return {
        "lead_id": lead.get("Id"),
        "ico":     str(lead.get("ICO__c", "")).strip().zfill(8),
        "Industry":               lead.get("Industry", "Technology"),
        "Rating":                 lead.get("Rating", "Warm"),
        "LeadSource":             lead.get("LeadSource", "Web"),
        "NumberOfEmployees":      _int(lead.get("NumberOfEmployees")),
        "Annual_Revenue_MCZK__c": round(_float(lead.get("AnnualRevenue")) / 1_000_000, 2),
        "Time_to_First_Response_h__c": _float(lead.get("Time_to_First_Response_h__c"), 24.0),
        "Web_Interactions__c":    _int(lead.get("Web_Interactions__c")),
        "Email_Opens__c":         _int(lead.get("Email_Opens__c")),
        "Email_Clicks__c":        _int(lead.get("Email_Clicks__c")),
        "Emails_Sent__c":         _int(lead.get("Emails_Sent__c")),
        "Email_Open_Rate__c":     _float(lead.get("Email_Open_Rate__c")),
        "Form_Submissions__c":    _int(lead.get("Form_Submissions__c")),
        "Demo_Requested__c":      _int(lead.get("Demo_Requested__c")),
        "Calls_Made__c":          _int(lead.get("Calls_Made__c")),
        "Meetings_Held__c":       _int(lead.get("Meetings_Held__c")),
        "Proposal_Sent__c":       _int(lead.get("Proposal_Sent__c")),
        "Content_Downloads__c":   _int(lead.get("Content_Downloads__c")),
        "Chatbot_Interactions__c":_int(lead.get("Chatbot_Interactions__c")),
        "LinkedIn_Viewed__c":     _int(lead.get("LinkedIn_Viewed__c")),
        "Days_Since_Last_Activity__c": _int(lead.get("Days_Since_Last_Activity__c"), 30),
        "Days_in_Pipeline__c":    _int(lead.get("Days_in_Pipeline__c"), 30),
    }


def _score_batch(payloads: list[dict]) -> list[dict]:
    """POST to /score/enrich for each lead (or /score/batch for full payloads)."""
    results = []
    for i in range(0, len(payloads), BATCH_SIZE):
        chunk = payloads[i : i + BATCH_SIZE]
        scored_chunk = []
        for payload in chunk:
            # Use /score/enrich so ARES resolves firmographics automatically.
            try:
                r = requests.post(
                    f"{API_URL}/score/enrich",
                    json=payload,
                    timeout=10,
                )
                r.raise_for_status()
                scored_chunk.append(r.json())
            except Exception as exc:
                log.warning("Score failed for lead %s: %s", payload.get("lead_id"), exc)
                scored_chunk.append(None)
        results.extend(scored_chunk)
        log.info("Scored %d/%d leads.", min(i + BATCH_SIZE, len(payloads)), len(payloads))
        time.sleep(0.1)  # gentle rate-limit on ARES calls
    return results


# ── Salesforce upsert ─────────────────────────────────────────────────────────

def _upsert_scores(sf, leads: list[dict], scores: list[dict | None]) -> None:
    """Write AI_Score__c, AI_Segment__c, AI_Top_Driver__c back to Salesforce."""
    records = []
    for lead, score in zip(leads, scores):
        if score is None:
            continue
        records.append({
            "Id":                lead["Id"],
            "AI_Score__c":       score.get("ai_score"),
            "AI_Segment__c":     score.get("segment"),
            "AI_Top_Driver__c":  (score.get("top_drivers") or [""])[0],
        })

    if not records:
        log.warning("No records to upsert.")
        return

    # Bulk update in chunks of 200 (Salesforce bulk API limit per call).
    for i in range(0, len(records), 200):
        chunk = records[i : i + 200]
        result = sf.bulk.Lead.update(chunk, batch_size=200)
        errors = [r for r in result if not r.get("success")]
        if errors:
            log.warning("%d upsert errors in chunk %d: %s", len(errors), i // 200, errors[:3])
        else:
            log.info("Upserted %d lead scores.", len(chunk))


# ── Feedback collection ───────────────────────────────────────────────────────

def _push_feedback(closes: list[dict]) -> None:
    """POST each real close outcome to /feedback."""
    success = 0
    for close in closes:
        try:
            r = requests.post(f"{API_URL}/feedback", json=close, timeout=5)
            r.raise_for_status()
            success += 1
        except Exception as exc:
            log.warning("Feedback failed for %s: %s", close.get("lead_id"), exc)
    log.info("Pushed %d/%d outcomes to /feedback.", success, len(closes))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Salesforce sync started ===")
    start = time.time()

    sf = _sf_client()

    # Step 1 & 2 & 3: pull → score → upsert.
    leads = _query_open_leads(sf)
    if leads:
        payloads = [_sf_lead_to_payload(l) for l in leads]
        scores   = _score_batch(payloads)
        _upsert_scores(sf, leads, scores)
    else:
        log.info("No open leads to score.")

    # Step 4: collect real outcomes and feed them back.
    closes = _query_recent_closes(sf)
    if closes:
        _push_feedback(closes)
    else:
        log.info("No recent closes to feed back.")

    elapsed = time.time() - start
    log.info("=== Salesforce sync complete in %.1fs ===", elapsed)

    # Report retrain readiness.
    try:
        r = requests.get(f"{API_URL}/feedback/stats", timeout=5)
        stats = r.json()
        if stats.get("ready_for_retrain"):
            log.info(
                "RETRAIN READY: %d new outcomes since last retrain "
                "(threshold=%d). Run: python src/train.py",
                stats["new_since_last_retrain"],
                stats["retrain_threshold"],
            )
        else:
            log.info(
                "Retrain not yet needed: %d/%d new outcomes since last retrain.",
                stats.get("new_since_last_retrain", 0),
                stats.get("retrain_threshold", 200),
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
