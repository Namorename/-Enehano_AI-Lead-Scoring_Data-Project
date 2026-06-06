"""
sf_sync.py
==========
Batch lead-scoring sync with Salesforce — integration **Path A** (see
docs/SALESFORCE_INTEGRATION.md).

Flow:
  1. Connect to a Salesforce org (Developer Edition works) via simple-salesforce.
  2. Query open Leads that carry a company IČO.
  3. Score each Lead **in-process** with the trained models (no HTTP, no hosting):
     firmographics are resolved from ARES via the IČO, exactly like /score/enrich.
  4. Write the AI fields back to Salesforce in one Bulk API call.

Because scoring runs in-process and the job talks to Salesforce *outbound*, this
path needs no public hosting — it can run from a laptop, a cron box, or a CI job.

Configuration (env vars, optionally via a .env file — see .env.example):
  SF_USER, SF_PASS, SF_TOKEN     Salesforce username / password / security token
  SF_DOMAIN                      'login' (prod/Dev Edition, default) or 'test' (sandbox)
  SF_ICO_FIELD                   Lead field holding the IČO (default 'ICO__c')
  SF_BEHAVIORAL_FIELDS           Comma-separated Lead fields to feed the model
                                 (must be valid EnrichLeadInput names). Empty by
                                 default so a fresh org with only ICO__c still works.

Usage:
  python src/sf_sync.py --dry-run          # score + print, write nothing
  python src/sf_sync.py --limit 50         # only the first 50 leads
  python src/sf_sync.py                     # full run, writes back to Salesforce
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import api  # noqa: E402  (in-process models + scoring pipeline)
from sf_common import (  # noqa: E402
    BEHAVIORAL_FIELDS,
    ICO_FIELD,
    connect,
    enable_utf8_stdout,
)

# Note: ICO_FIELD / BEHAVIORAL_FIELDS / connect() come from sf_common so the
# seeder (create_test_leads.py) can share them without importing the ML stack.
# BEHAVIORAL_FIELDS is empty by default (SF_BEHAVIORAL_FIELDS) so a fresh org
# with only ICO__c added still works; the model then scores on firmographics
# plus default behaviour. Opt in once those custom fields exist, e.g.:
#   SF_BEHAVIORAL_FIELDS="Web_Interactions__c,Email_Opens__c,Meetings_Held__c,Demo_Requested__c,Proposal_Sent__c"

# AI result fields written back to the Lead. Override here if your API names differ.
OUT_SCORE = "AI_Score__c"
OUT_SEGMENT = "AI_Segment__c"
OUT_EXPECTED_WIN = "Expected_Win__c"
OUT_TOP_DRIVER = "AI_Top_Driver__c"
OUT_SCORED_DATE = "AI_Scored_Date__c"
OUT_MODEL_VERSION = "AI_Model_Version__c"


def build_soql(limit: int | None = None) -> str:
    fields = ["Id", ICO_FIELD, *BEHAVIORAL_FIELDS]
    soql = (
        f"SELECT {', '.join(dict.fromkeys(fields))} FROM Lead "
        f"WHERE {ICO_FIELD} != null AND IsConverted = false"
    )
    if limit:
        soql += f" LIMIT {int(limit)}"
    return soql


def score_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Pure, offline-testable core: map Salesforce Lead dicts → Lead update dicts.

    Returns (updates, failures). A failure (e.g. ARES could not resolve the IČO)
    skips that one lead instead of aborting the batch. No Salesforce or network
    state is required beyond ARES, which `api.score_enrich` handles + caches.
    """
    updates: list[dict] = []
    failures: list[dict] = []
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    for rec in records:
        ico = rec.get(ICO_FIELD)
        try:
            kwargs = {
                field: rec[field]
                for field in BEHAVIORAL_FIELDS
                if rec.get(field) is not None
            }
            payload = api.EnrichLeadInput(ico=str(ico), **kwargs)
            score = api.score_enrich(payload)
            updates.append({
                "Id": rec["Id"],
                OUT_SCORE: score.ai_score,
                OUT_SEGMENT: score.segment,
                OUT_EXPECTED_WIN: score.expected_win_pct,
                OUT_TOP_DRIVER: (score.top_drivers or [None])[0],
                OUT_SCORED_DATE: now,
                OUT_MODEL_VERSION: api.app.version,
            })
        except Exception as exc:  # noqa: BLE001 - one bad lead must not kill the batch
            failures.append({"Id": rec.get("Id"), "ico": ico, "error": str(exc)})
    return updates, failures


def main() -> None:
    enable_utf8_stdout()
    parser = argparse.ArgumentParser(description="Score Salesforce leads in batch.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score and print, but do not write back to Salesforce.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N leads.")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Bulk API batch size for the write-back.")
    args = parser.parse_args()

    api._load_assets()  # load models once
    sf = connect()

    soql = build_soql(args.limit)
    print(f"Querying: {soql}")
    records = sf.query_all(soql)["records"]
    print(f"Fetched {len(records)} lead(s).")

    updates, failures = score_records(records)
    print(f"Scored {len(updates)} | skipped {len(failures)}.")
    for f in failures[:10]:
        print(f"  [skip] {f['ico']}: {f['error'][:80]}")

    if not updates:
        print("Nothing to write.")
        return

    if args.dry_run:
        print("\n--- DRY RUN (no write) — sample of first 5 updates ---")
        for u in updates[:5]:
            print(f"  {u['Id']}  score={u[OUT_SCORE]}  seg={u[OUT_SEGMENT]}  "
                  f"win={u[OUT_EXPECTED_WIN]}  driver={u[OUT_TOP_DRIVER]}")
        return

    sf.bulk.Lead.update(updates, batch_size=args.batch_size)
    print(f"Wrote {len(updates)} lead score(s) back to Salesforce.")


if __name__ == "__main__":
    main()
