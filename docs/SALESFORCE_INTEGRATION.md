# Salesforce integration

How to make the lead-scoring model usable inside Salesforce, on a free org.
This expands on README section 7.

## The free Salesforce

The Developer Edition org is free and does not expire (as long as you log in
now and then). It includes Apex, Flow, Lightning Web Components and full
REST/Bulk API access, which is everything this needs. Sign up at
developer.salesforce.com/signup.

Note: API and Apex callouts work in Developer, Enterprise, Unlimited and
Performance editions. Professional/Group/Essentials restrict API access, so if
the client's production org is Professional Edition the real-time path may need
an API add-on. Worth checking their edition early.

## The connectivity gap

The model runs as an external service (the FastAPI app). Salesforce is in the
cloud and cannot reach `localhost`, so for the real-time paths the service needs
a public HTTPS URL:

- Dev/demo: ngrok or Cloudflare Tunnel exposes local :8000 in one command.
- Staging: Render / Railway / Fly.io free tier deploys the Docker image and
  gives a stable URL (may cold-start on the free tier).
- Production: a paid tier or the client's own cloud.

The batch path (below) avoids this entirely: the job runs the model in-process
and only makes outbound calls to Salesforce, so nothing has to be hosted. That
is why it is the recommended first step.

## Data model

Custom fields to add on the Lead object (Setup > Object Manager > Lead > Fields
& Relationships). Mirror them on Opportunity if you also score after conversion.

| Field label | API name | Type | Purpose |
|---|---|---|---|
| ICO | `ICO__c` | Text(8), External ID, Unique | key for ARES enrichment and upsert |
| AI Score | `AI_Score__c` | Number(3,0) | 0-100 conversion score |
| AI Segment | `AI_Segment__c` | Picklist (High/Medium/Low) | priority bucket |
| Expected Win | `Expected_Win__c` | Percent(5,2) | P(convert) x P(win) |
| AI Top Driver | `AI_Top_Driver__c` | Text(255) | top SHAP feature |
| AI Scored Date | `AI_Scored_Date__c` | DateTime | freshness / audit |
| AI Model Version | `AI_Model_Version__c` | Text(20) | which model produced the score |

Marking `ICO__c` as an External ID lets the batch job upsert by IČO without
first looking up Salesforce IDs.

When you create a field, type the API name WITHOUT the `__c` suffix (Salesforce
adds it). For example, type `AI_Score`, not `AI_Score__c`.

## Three ways to integrate

| | A. Batch (Python) | B. Real-time Apex | C. External Services |
|---|---|---|---|
| How | job reads leads, scores, writes back via Bulk API | Flow/trigger calls the API on save | import the API's OpenAPI, call from Flow |
| Hosting | none | public HTTPS | public HTTPS |
| Apex | none | yes | none |
| Latency | minutes to hours | seconds | seconds |
| Effort | low | medium | medium |

Start with A (working in a day, no hosting), then add B or C for real-time once
the service is hosted. `/score/enrich` is the natural target for B and C: the
caller sends only the IČO plus behavioural fields and ARES fills in the rest, so
the payload stays small.

## Step by step

### Phase 0 - setup
1. Create a Developer Edition org, verify the email, set a password.
2. Get a security token: Setup > Reset My Security Token (it is emailed).
3. Add the custom fields above.
4. Set `SCORING_API_KEY` if you will expose the API (see below).

### Phase 1 - batch (the MVP, already built)
1. `pip install -r requirements.txt`.
2. Fill `.env` (copy from `.env.example`) with SF_USER / SF_PASS / SF_TOKEN.
3. Seed sample leads, then score:
   ```
   python src/create_test_leads.py --count 20
   python src/sf_sync.py --dry-run
   python src/sf_sync.py
   ```
4. Schedule `sf_sync.py` (Task Scheduler / cron) to refresh scores, e.g. nightly.

`sf_sync.py` pulls open leads, scores them in-process (ARES enrichment via the
IČO), and writes `AI_Score__c` and the other fields back. `create_test_leads.py`
seeds an org with real Czech IČOs from the training data.

### Phase 2 - host the service (needed for B and C)
1. Dev: `ngrok http 8000` and copy the HTTPS URL.
2. Staging: deploy the Docker image to Render/Railway for a stable URL.
3. Smoke-test `POST /score/enrich` against the public URL with the API key.

### Phase 3 - real-time scoring (Apex)
1. Create a Named Credential pointing at the service URL, with the API key as a
   custom header. This keeps the URL/secret out of Apex and removes the need for
   a Remote Site Setting.
2. Add an Apex class that POSTs to the endpoint and writes the result back
   (stub below).
3. Call it from a record-triggered Flow (or trigger) on lead create/update when
   `ICO__c` is set.
4. Add an Apex test with an HttpCalloutMock (Salesforce needs >=75% coverage).

Alternative (no Apex): register the API's `/openapi.json` under Setup > External
Services, which generates an invocable action you can drop into a Flow. The
generated schema sometimes needs trimming; if it fights you, use the Apex path.

## Securing the API

`src/api.py` already has an optional API-key check. Set `SCORING_API_KEY` on the
host and send the same value in the `X-API-Key` header (the Named Credential
does this). With the variable unset the check is skipped, so local runs need no
key.

## Limits and cost

- Developer Edition has a modest daily API allocation. The batch job uses the
  Bulk API, which counts per batch rather than per record, so it stays well
  under the limit. Check Setup > Company Information for current usage.
- Prototype cost is zero: free org, open data, open-source stack. Production
  cost is mainly data engineering plus optional hosting for real-time.

## Appendix A - Apex callout stub

```apex
public with sharing class LeadScoringService {
    public class ScoreResponse {
        public Decimal ai_score;
        public String segment;
        public Decimal expected_win_pct;
        public List<String> top_drivers;
    }

    @future(callout=true)
    public static void scoreLead(Id leadId, String ico, Decimal webInteractions,
                                 Decimal meetings, Boolean demo, Boolean proposal) {
        HttpRequest req = new HttpRequest();
        req.setEndpoint('callout:Lead_Scoring_API/score/enrich'); // Named Credential
        req.setMethod('POST');
        req.setHeader('Content-Type', 'application/json');
        req.setBody(JSON.serialize(new Map<String, Object>{
            'ico' => ico,
            'Web_Interactions__c' => webInteractions,
            'Meetings_Held__c' => meetings,
            'Demo_Requested__c' => demo ? 1 : 0,
            'Proposal_Sent__c' => proposal ? 1 : 0
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
        }
    }
}
```

Trigger this from a record-triggered Flow on lead create/update where `ICO__c`
is set, and add an HttpCalloutMock test for coverage.
