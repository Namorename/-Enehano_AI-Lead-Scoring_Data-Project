"""
create_test_leads.py
====================
Seed a Salesforce org with sample Leads so there is realistic data to score
(see docs/SALESFORCE_INTEGRATION.md, Phase 1).

Reads rows from data/lead_train.csv (real Czech companies + behavioural signals)
and inserts them as Leads via the Bulk API. Each Lead gets the required standard
fields (LastName, Company), the IČO custom field (so ARES enrichment works), and
optionally the behavioural fields listed in SF_BEHAVIORAL_FIELDS.

Usage:
  python src/create_test_leads.py --count 20 --dry-run   # preview, insert nothing
  python src/create_test_leads.py --count 20             # insert into Salesforce

Prereqs: the Lead custom field ICO__c must exist (plus any behavioural fields you
list in SF_BEHAVIORAL_FIELDS). Credentials come from the same env vars as sf_sync.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from paths import LEAD_TRAIN  # noqa: E402
from sf_common import (  # noqa: E402
    BEHAVIORAL_FIELDS,
    ICO_FIELD,
    connect,
    enable_utf8_stdout,
)

# Column names in data/lead_train.csv.
ICO_CSV_COL = "IČO"
COMPANY_CSV_COL = "Company_Name"


def _coerce(value):
    """Return a JSON-friendly scalar: int when whole, else float."""
    f = float(value)
    return int(f) if f.is_integer() else f


def build_leads(count: int) -> list[dict]:
    """Build Salesforce Lead dicts from the training data. Pure / offline —
    needs only the CSV, so it is testable without a Salesforce connection."""
    df = pd.read_csv(LEAD_TRAIN, dtype={ICO_CSV_COL: str})
    df = df.dropna(subset=[ICO_CSV_COL]).head(count)

    leads: list[dict] = []
    # to_dict("records") preserves exact column names, including the non-ASCII
    # "IČO" header that itertuples would rename to a positional alias.
    for i, row_d in enumerate(df.to_dict("records"), start=1):
        company = str(row_d.get(COMPANY_CSV_COL) or f"Test Co {i}")[:255]
        lead = {
            "LastName": f"AI Test {i}",
            "Company": company,
            ICO_FIELD: str(row_d[ICO_CSV_COL]).strip().zfill(8),
        }
        for field in BEHAVIORAL_FIELDS:
            val = row_d.get(field)
            if val is not None and pd.notna(val):
                lead[field] = _coerce(val)
        leads.append(lead)
    return leads


def main() -> None:
    enable_utf8_stdout()
    parser = argparse.ArgumentParser(description="Seed Salesforce with test Leads.")
    parser.add_argument("--count", type=int, default=20, help="How many leads to create.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and print, but do not insert into Salesforce.")
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()

    leads = build_leads(args.count)
    print(f"Prepared {len(leads)} test lead(s) "
          f"(behavioural fields: {BEHAVIORAL_FIELDS or 'none'}).")

    if args.dry_run:
        print("\n--- DRY RUN (no insert) — sample of first 5 ---")
        for lead in leads[:5]:
            print(f"  {lead['LastName']:12s} | {lead[ICO_FIELD]} | {lead['Company']}")
        return

    sf = connect()
    results = sf.bulk.Lead.insert(leads, batch_size=args.batch_size)
    ok = sum(1 for r in results if r.get("success"))
    print(f"Inserted {ok}/{len(results)} lead(s).")
    for r in results:
        if not r.get("success"):
            print(f"  [error] {r.get('errors')}")
            break  # show the first error (usually a missing field or FLS issue)


if __name__ == "__main__":
    main()
