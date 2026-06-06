# Salesforce Integration Plan — Enehano Lead Intelligence

How to make the lead-scoring model usable **inside Salesforce**, using a free
Salesforce org. This builds on README §8 but goes from idea to step-by-step.

---

## 0. TL;DR

- **Yes, there is a genuinely free Salesforce: the Developer Edition (DE) org.**
  It is free forever (as long as you log in periodically), and includes full
  **Apex, Flow, Lightning Web Components, and REST/Bulk API access** — everything
  this integration needs. No credit card, no trial clock.
- The model runs as an **external service** (the existing FastAPI app). Salesforce
  is in the cloud and **cannot reach `localhost`**, so the scoring service must be
  reachable over public HTTPS (a tunnel for dev, a free host for staging).
- There are **three integration paths**. Recommended order:
  1. **Batch sync (Python)** — fastest to working, *no hosting needed*. MVP.
  2. **Real-time Apex callout** via Named Credential — scores on lead create/edit.
  3. **No-Apex External Services** — import the API's OpenAPI schema, call it from Flow.
- Two prerequisites before any cloud exposure: **add an API key to the FastAPI
  service** (it is currently unauthenticated) and **add an `ICO__c` field** to Lead.

---

## 1. The free Salesforce — verified

| Option | Free? | Good for | Notes |
|---|---|---|---|
| **Developer Edition (recommended)** | ✅ Free, non-expiring | Building this integration | Full Apex/Flow/LWC + API. Sign up: developer.salesforce.com/signup |
| Trailhead Playground | ✅ Free | Learning, throwaway tests | Tied to a Trailhead account |
| 30-day Sales/Platform trial | ⚠️ Time-boxed | Demoing paid features | Expires; not for ongoing dev |

**Use a Developer Edition org.** It is the "free version" you were told about and
is the standard way developers build and test Salesforce integrations.

> ⚠️ Edition caveat: API + Apex callouts are fully available in **Developer,
> Enterprise, Unlimited, and Performance** editions. **Professional/Group/Essentials**
> editions restrict API access. So the DE org works for development, but if the
> client's *production* org is Professional Edition, the real-time API path may
> require an API add-on — confirm the client's edition early.

---

## 2. Architecture & the connectivity gap

```
   Salesforce (cloud)                         Your scoring service (external)
 ┌────────────────────┐   HTTPS POST        ┌──────────────────────────────┐
 │ Lead record        │  /score/enrich      │ FastAPI (src/api.py)          │
 │  + AI_Score__c     │ ──────────────────► │  ├─ ARES enrichment (IČO)     │
 │  + AI_Segment__c   │ ◄────────────────── │  ├─ conv + win models (.pkl)  │
 │  Flow / Apex / ESvc │   JSON score        │  └─ SHAP top_drivers          │
 └────────────────────┘                     └──────────────────────────────┘
        ▲                                              ▲
        │ Bulk API (write back)                        │ must be PUBLIC HTTPS
        └────────── Python batch job (Path A) ─────────┘ (localhost is unreachable
                                                          from Salesforce cloud)
```

**Key constraint:** Salesforce servers cannot call `http://localhost:8000`. The
service must have a public HTTPS URL. Options (all have free tiers):

| Need | Tool | Notes |
|---|---|---|
| Dev / demo (quickest) | **ngrok** or **Cloudflare Tunnel** | One command exposes local `:8000` as `https://xxxx.ngrok.io`. Perfect for testing callouts. |
| Staging (always-on) | **Render** / **Railway** / **Fly.io** | Free tier deploys the Docker image; gives a stable HTTPS URL. May cold-start on free tier. |
| Production | Render/Railway paid, or client cloud | Always-on, custom domain, no cold start. |

> 💡 **Path A (batch) avoids this gap entirely** — the Python job runs the model
> in-process and talks to Salesforce *outbound*, so nothing needs to be hosted.
> That is why it is the recommended MVP.

---

## 3. Salesforce data model

Create these custom fields on the **Lead** object (Setup → Object Manager → Lead →
Fields & Relationships). Mirror them on **Opportunity** if you also score post-conversion.

| Field label | API name | Type | Purpose |
|---|---|---|---|
| Company Reg. No. (IČO) | `ICO__c` | Text(8), External ID, Unique | Key for ARES enrichment + upsert matching |
| AI Score | `AI_Score__c` | Number(3,0) | 0–100 conversion score |
| AI Segment | `AI_Segment__c` | Picklist (High/Medium/Low) | Prioritisation bucket |
| Expected Win % | `Expected_Win__c` | Percent(5,2) | P(convert) × P(win) |
| AI Top Driver | `AI_Top_Driver__c` | Text(255) | `top_drivers[0]` — explainability |
| AI Scored Date | `AI_Scored_Date__c` | DateTime | Freshness / audit |
| AI Model Version | `AI_Model_Version__c` | Text(20) | Which model produced the score |

Marking `ICO__c` as an **External ID** is important: it lets the batch job
**upsert** by IČO without first querying Salesforce IDs.

---

## 4. The three integration paths

| | **A. Batch sync (Python)** | **B. Real-time Apex callout** | **C. External Services (no Apex)** |
|---|---|---|---|
| How | Python job reads Leads via API, scores locally, writes back via Bulk API | Lead trigger/Flow → Apex HTTP callout to FastAPI on create/edit | Import FastAPI OpenAPI → auto-generated Flow action calls the API |
| Hosting needed | ❌ None | ✅ Public HTTPS | ✅ Public HTTPS |
| Apex needed | ❌ (pure Python) | ✅ Apex class + test | ❌ Declarative |
| Latency | Minutes–hours (scheduled) | Seconds (on save) | Seconds (on save) |
| Effort | **Low** | Medium | Medium |
| Best for | MVP, bulk re-scoring, no infra | "Score the moment a rep saves a lead" | Admins who prefer clicks over code |
| Reuses | `src/api.py` model code directly | `/score/enrich` endpoint | `/score/enrich` + `/openapi.json` |

**Recommendation:** ship **Path A** first (working end-to-end in a day, no hosting),
then add **Path B** (or **C**) for real-time scoring once the service is hosted.

The `/score/enrich` endpoint is the ideal Salesforce target for B and C: the caller
sends only the **IČO + behavioural metrics**, and firmographics are resolved
server-side from ARES — so the Salesforce payload stays tiny.

---

## 5. Prerequisite — secure the API (do this first)

`src/api.py` is currently **unauthenticated**. Before exposing it publicly, add a
simple API-key check. Salesforce will send the key via a Named Credential header.

```python
# src/api.py — add near the top
import os
from fastapi import Header, HTTPException

API_KEY = os.getenv("SCORING_API_KEY")  # set in the host's env / .env

def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

# then protect the scoring routes:
# @app.post("/score/enrich", dependencies=[Depends(require_api_key)])
```

(If `SCORING_API_KEY` is unset, the check is skipped — keeps local dev frictionless.)

---

## 6. Step-by-step plan

### Phase 0 — Foundations (½ day)
1. Sign up for a **Developer Edition** org → verify email → set security token
   (Setup → Reset My Security Token).
2. Add the **custom fields** from §3.
3. Add the **API key** guard from §5; set `SCORING_API_KEY`.
4. Create a **Connected App** (Setup → App Manager → New Connected App) with OAuth
   enabled — needed so the Python job and/or Named Credential can authenticate.

### Phase 1 — Path A: Batch sync (MVP, ~1 day)
1. `pip install simple-salesforce` (add to `requirements.txt`).
2. Write `src/sf_sync.py` (stub in Appendix A): authenticate → query open Leads →
   score in-process with the existing models → **bulk upsert** the AI fields.
3. Run manually, verify scores land on Lead records.
4. Schedule it (Windows Task Scheduler / cron / a Render cron job) — e.g. nightly.

✅ **Deliverable:** every open Lead shows an AI score in Salesforce, refreshed nightly.

### Phase 2 — Host the service (½ day, needed for B/C)
1. Dev: `ngrok http 8000` → copy the `https://…ngrok.io` URL.
2. Staging: `docker compose` build → deploy to Render/Railway → stable HTTPS URL.
3. Smoke-test `POST /score/enrich` with the API key from the public URL.

### Phase 3 — Path B: Real-time Apex callout (~1–2 days)
1. **Named Credential** (Setup → Named Credentials → New):
   - URL = the public service URL; add a custom header `X-API-Key` = your key.
   - This avoids hard-coding the URL/secret in Apex and removes the need for a
     Remote Site Setting.
2. Apex class `LeadScoringService` (Appendix B) — `@future(callout=true)` method
   that POSTs to `callout:Lead_Scoring_API/score/enrich` and writes the response
   back to the Lead.
3. Invoke it from a **record-triggered Flow** (preferred) or an Apex trigger on
   Lead create/update when `ICO__c` is present.
4. Write the **Apex test** (mock the callout with `HttpCalloutMock`) — Salesforce
   requires ≥75% coverage to deploy.

✅ **Deliverable:** saving a Lead with an IČO scores it within seconds.

### Phase 3-alt — Path C: External Services (no Apex)
1. Setup → External Services → New → paste the FastAPI **`/openapi.json`** schema
   (or a trimmed version — see caveat below).
2. Salesforce generates an **invocable action** per endpoint.
3. Drop the action into a record-triggered **Flow**; map Lead fields → request,
   response → AI fields.

> ⚠️ External Services caveat: it ingests OpenAPI 2.0/3.0 but is picky about complex/
> nested schemas and has size limits. FastAPI's auto-generated schema may need
> trimming (flatten the response model, remove unused components). If it imports
> cleanly, this is the lowest-maintenance option; if it fights you, use Path B.

### Phase 4 — Surface it in the UI (½ day)
- **List View / Report:** a "Hot Leads" Lead list view sorted by `AI_Score__c`
  desc, filtered `AI_Segment__c = High`. This alone delivers most of the business value.
- **Record page:** show `AI_Score__c`, `AI_Segment__c`, `Expected_Win__c`,
  `AI_Top_Driver__c` in a highlights panel; optionally a small LWC gauge.
- **Dashboard:** segment distribution, avg expected-win by owner/region.

### Phase 5 — Test, limit-check, monitor (½ day)
- Validate scores match the standalone API for the same input.
- Confirm **API limits**: Developer Edition has a modest daily API allocation —
  use **Bulk API** (batches records, not 1 call/record) in Path A to stay well
  under it. Check Setup → Company Information → API usage.
- Log failures (callout errors, auth) and alert.

### Phase 6 — Production hardening (later)
- Move secrets to Named Credential / a secrets manager (never in code).
- Always-on hosting (no cold starts), retries/back-off, idempotency.
- Champion/challenger: keep the rule-based score visible alongside AI for trust.
- Retrain cadence + `AI_Model_Version__c` stamping; monitor score drift.

---

## 7. Effort & cost summary

| Item | Cost | Time |
|---|---|---|
| Salesforce Developer Edition | **Free** | 30 min signup |
| Path A (batch) | **Free** (runs anywhere) | ~1 day |
| Hosting for B/C | Free tier (ngrok/Render/Railway) | ~½ day |
| Path B or C (real-time) | **Free** on DE | ~1–2 days |
| **Total to a working in-Salesforce MVP** | **€0** | **~2–3 days** |

---

## 8. Risks & decisions to confirm

1. **Production edition.** DE proves it works; if the client's prod org is
   *Professional Edition*, real-time API access may cost extra. Confirm early.
2. **Where the service lives.** Batch (Path A) needs no hosting; real-time does.
3. **Data residency / PII.** Sending lead data to an external service may need
   client sign-off; hosting in the EU (Render/Railway EU region) helps.
4. **ARES rate limits.** `/score/enrich` calls ARES; the in-process cache helps,
   but for high volume pre-fetch/cache firmographics.
5. **Auth model.** API key (simple) vs OAuth/JWT on the Named Credential (stronger).

---

## Appendix A — Batch sync stub (`src/sf_sync.py`)

```python
"""Pull open Leads from Salesforce, score them, write the AI fields back."""
import os, datetime as dt
import pandas as pd
from simple_salesforce import Salesforce
import api  # reuse the loaded models + scoring pipeline

api._load_assets()
sf = Salesforce(
    username=os.environ["SF_USER"],
    password=os.environ["SF_PASS"],
    security_token=os.environ["SF_TOKEN"],
)

# 1. Read leads that have an IČO and need (re)scoring
rows = sf.query_all(
    "SELECT Id, ICO__c, Web_Interactions__c, Email_Opens__c, Meetings_Held__c, "
    "Demo_Requested__c, Proposal_Sent__c, Time_to_First_Response_h__c "
    "FROM Lead WHERE ICO__c != null AND Status = 'Open - Not Contacted'"
)["records"]

updates = []
for r in rows:
    payload = api.EnrichLeadInput(
        ico=r["ICO__c"],
        Web_Interactions__c=r.get("Web_Interactions__c") or 0,
        Email_Opens__c=r.get("Email_Opens__c") or 0,
        Meetings_Held__c=r.get("Meetings_Held__c") or 0,
        Demo_Requested__c=int(bool(r.get("Demo_Requested__c"))),
        Proposal_Sent__c=int(bool(r.get("Proposal_Sent__c"))),
        Time_to_First_Response_h__c=r.get("Time_to_First_Response_h__c") or 24.0,
    )
    s = api.score_enrich(payload)             # in-process; no HTTP, no hosting
    updates.append({
        "Id": r["Id"],
        "AI_Score__c": s.ai_score,
        "AI_Segment__c": s.segment,
        "Expected_Win__c": s.expected_win_pct,
        "AI_Top_Driver__c": (s.top_drivers or [None])[0],
        "AI_Scored_Date__c": dt.datetime.utcnow().isoformat(),
        "AI_Model_Version__c": api.app.version,
    })

# 2. Write back in one Bulk API call (efficient, limit-friendly)
if updates:
    sf.bulk.Lead.update(updates, batch_size=10000)
print(f"Scored and updated {len(updates)} leads.")
```

## Appendix B — Apex real-time callout (`LeadScoringService.cls`)

```apex
public with sharing class LeadScoringService {
    public class ScoreResponse {
        public Decimal ai_score; public String segment;
        public Decimal expected_win_pct; public List<String> top_drivers;
    }

    @future(callout=true)
    public static void scoreLead(Id leadId, String ico, Decimal webInteractions,
                                 Decimal emailOpens, Decimal meetings,
                                 Boolean demo, Boolean proposal, Decimal ttfr) {
        HttpRequest req = new HttpRequest();
        req.setEndpoint('callout:Lead_Scoring_API/score/enrich'); // Named Credential
        req.setMethod('POST');
        req.setHeader('Content-Type', 'application/json');
        req.setBody(JSON.serialize(new Map<String, Object>{
            'ico' => ico,
            'Web_Interactions__c' => webInteractions,
            'Email_Opens__c' => emailOpens,
            'Meetings_Held__c' => meetings,
            'Demo_Requested__c' => demo ? 1 : 0,
            'Proposal_Sent__c' => proposal ? 1 : 0,
            'Time_to_First_Response_h__c' => ttfr
        }));

        HttpResponse res = new Http().send(req);
        if (res.getStatusCode() == 200) {
            ScoreResponse s = (ScoreResponse) JSON.deserialize(
                res.getBody(), ScoreResponse.class);
            update new Lead(
                Id = leadId,
                AI_Score__c = s.ai_score,
                AI_Segment__c = s.segment,
                Expected_Win__c = s.expected_win_pct,
                AI_Top_Driver__c = (s.top_drivers == null || s.top_drivers.isEmpty())
                                   ? null : s.top_drivers[0],
                AI_Scored_Date__c = System.now()
            );
        } else {
            System.debug('Scoring callout failed: ' + res.getStatusCode()
                         + ' ' + res.getBody());
        }
    }
}
```

Trigger it from a record-triggered Flow (call this invocable/`@future` method) on
Lead create/update where `ICO__c` is set. Add an `HttpCalloutMock` test for ≥75% coverage.

---

## Sources
- [Sign up for Salesforce Developer Edition (REST API Guide)](https://developer.salesforce.com/docs/atlas.en-us.api_rest.meta/api_rest/quickstart_dev_org.htm)
- [Choose a Salesforce Org for Apex Development](https://developer.salesforce.com/docs/atlas.en-us.apexcode.meta/apexcode/apex_intro_get_dev_account.htm)
- [Salesforce editions with API access](https://help.salesforce.com/s/articleView?id=000385436&language=en_US&type=1)
- [Named Credentials as Callout Endpoints (Apex Guide)](https://developer.salesforce.com/docs/atlas.en-us.apexcode.meta/apexcode/apex_callouts_named_credentials.htm)
- [Invoking HTTP Callouts (Apex Guide)](https://developer.salesforce.com/docs/atlas.en-us.apexcode.meta/apexcode/apex_callouts_http.htm)
- [Intro to External Services (Trailhead)](https://trailhead.salesforce.com/content/learn/modules/external-services/get-started-with-external-services)
- [Using Salesforce External Services in Flow without Apex](https://arrify.com/salesforce-external-services-in-flow/)
- [simple-salesforce (PyPI)](https://pypi.org/project/simple-salesforce/)
- [Using ngrok with FastAPI](https://ngrok.com/docs/using-ngrok-with/fastAPI)
- [FastAPI deployment options (Render)](https://render.com/articles/fastapi-deployment-options)
