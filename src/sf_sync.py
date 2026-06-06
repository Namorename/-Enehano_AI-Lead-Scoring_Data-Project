"""
Batch-score Salesforce leads and write the results back.

Pulls open leads, scores them with the trained models (firmographics come from
ARES via the IČO, the same way the /score/enrich endpoint works), and updates the
AI fields through the Bulk API. Runs anywhere that can reach Salesforce; nothing
needs to be hosted.

    python src/sf_sync.py --dry-run     # score and print only
    python src/sf_sync.py --limit 50    # first 50 leads
    python src/sf_sync.py               # full run
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import api
from sf_common import (
    BEHAVIORAL_FIELDS,
    ICO_FIELD,
    connect,
    enable_utf8_stdout,
)

# BEHAVIORAL_FIELDS is empty unless SF_BEHAVIORAL_FIELDS is set, so a lead with
# only an IČO still scores (on firmographics). Set it once the engagement fields
# exist on the Lead object, e.g.:
#   SF_BEHAVIORAL_FIELDS="Web_Interactions__c,Email_Opens__c,Meetings_Held__c"

# Lead fields we write the result into.
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
    """Turn Salesforce lead dicts into update dicts.

    Returns (updates, failures). A lead whose IČO ARES can't resolve is added to
    failures and skipped, so one bad row doesn't stop the rest. This has no
    Salesforce dependency, so it can be tested on its own.
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
        except Exception as exc:  # skip this lead, keep going
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
        print("\n--- DRY RUN (no write), first 5 updates ---")
        for u in updates[:5]:
            print(f"  {u['Id']}  score={u[OUT_SCORE]}  seg={u[OUT_SEGMENT]}  "
                  f"win={u[OUT_EXPECTED_WIN]}  driver={u[OUT_TOP_DRIVER]}")
        return

    sf.bulk.Lead.update(updates, batch_size=args.batch_size)
    print(f"Wrote {len(updates)} lead score(s) back to Salesforce.")


if __name__ == "__main__":
    main()
