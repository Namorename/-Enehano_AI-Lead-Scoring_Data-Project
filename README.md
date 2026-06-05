# Enehano Lead Intelligence Platform

Predictive lead-scoring MVP for **Enehano Solutions** — a Salesforce
consulting firm. Given a marketing/sales lead, the platform predicts:

1. **P(Convert)** — probability the lead becomes an Opportunity.
2. **P(Win | Converted)** — probability that Opportunity is Closed Won.
3. **Expected win % = P(Convert) × P(Win)** — single number used for ranking.

The score is delivered both as a percentage and as a **High / Medium / Low**
segment, ready to drop into Salesforce as a custom field.

---

## 1. Business problem

Enehano currently prioritises leads via simple rules and sales-rep intuition.
This is functional but inefficient: reps spend time on low-fit leads, and
there is no transparent, data-driven explanation for why a lead is "good".

**The question:** can we use historical CRM data + firmographics from the
Czech RES register to build a model that prioritises leads better than a
rule-based scoring sheet? **Answer: yes, and the uplift is measured in
converters reached per unit of sales effort (see §5), not just AUC.**

## 2. Data

| Source | Role |
|---|---|
| **RES open data** (ČSÚ) | Real Czech companies — IČO, legal form, NACE, region, employee category, founding date |
| **Synthetic CRM telemetry** | Web visits, response time, emails, meetings, forms, demo, LinkedIn, etc. |
| **Synthetic outcomes** | `Converted` (Lead → Opportunity) and `Closed_Won` (Opportunity outcome), generated via a noisy logistic model (`N(0, 0.8)` logit noise) so the AI has real signal to learn but cannot trivially memorise a formula |

**NACE filter.** Only sectors with realistic Salesforce-consulting demand
are kept: **C** (Manufacturing), **J** (ICT), **K** (Finance & Insurance),
**M** (Professional services). This shrinks ~90 k RES rows to ~30 k usable
companies.

**ARES enrichment.** `ares.py` calls the public ARES REST API
(`ares.gov.cz`) to fetch authoritative address, registration date and NACE
for any IČO. The `/score/enrich` endpoint accepts an IČO + behavioural
metrics and resolves firmographics server-side (process-level cache).

## 3. Models

Both stages are **probability-calibrated** (`CalibratedClassifierCV`,
isotonic) over a base estimator chosen from five candidates. Candidate
selection uses a **one-standard-error rule on cross-validated AUC**, not raw
held-out AUC: among models statistically tied on CV, the simplest/cheapest is
chosen (rationale persisted to `metrics.json → *_model.selection`).

| Model | Algorithm (auto-selected) | Target | Test AUC | PR-AUC | F1 |
|---|---|---|---|---|---|
| Conversion | Calibrated **HistGradientBoosting** | `Converted` | **0.935** | 0.861 | 0.757 |
| Win | Calibrated **LogisticRegression** (converted leads only) | `Closed_Won` | 0.665 | — | 0.530 |
| **Baseline** (rules) | Hand-written weighted sum (`baseline.py`) | `Converted` | 0.832¹ | — | 0.545 |

**Conversion uplift over the rule baseline: +0.103 AUC, +0.213 F1**, stable
across 5 time-series folds (CV std ≈ 0.003). Calibration error (ECE) drops
from 0.084 → **0.019** after isotonic calibration, so the percentages are
trustworthy probabilities, not just ranks.

¹ This baseline (`0.832`) uses the generator's stored `Rule_Based_Score`,
which folds in the excluded `Status` field. The rules the **API actually
serves** (`baseline.score_frame`, no `Status`) score AUC ≈ 0.64 — the honest
production comparison, used for the lift numbers in §5.

### Top conversion drivers (gain importance)

1. `Days_in_Pipeline__c` — pipeline maturity ⚠️ *leakage proxy (see §4)*
2. `Proposal_Sent__c` — late-funnel commitment
3. `Days_Since_Last_Activity__c` — recency ⚠️ *leakage proxy (see §4)*
4. `rating_encoded` — sales qualification (Hot/Warm/Cold)
5. `engagement_composite` — weighted digital-engagement score
6. `Demo_Requested__c` — high-intent signal
7. `Time_to_First_Response_h__c` — follow-up speed
8. `Email_Open_Rate__c` — marketing engagement

The model is **explainable** via SHAP (`/score` returns `top_drivers`), not a
black box — but the top two drivers carry a leakage caveat, addressed next.

## 4. Model integrity, calibration & leakage control

**Honesty matters more than a flattering number.** Two pipeline-stage features
(`Days_in_Pipeline__c`, `Days_Since_Last_Activity__c`) are mechanically derived
from the synthetic generator's hidden `Status` field, which drives the label.
The model partly reconstructs `Status` from these **leakage proxies**, which
inflates the headline 0.935 AUC on synthetic data but **will not generalise to
real CRM data**.

This is controllable, not hidden:

| Mode | Command | Conversion model | Test AUC | Capture @ top-20% vs **deployed** rules |
|---|---|---|---|---|
| **Default** (synthetic benchmark) | `python src/train.py` | hist_gbm | 0.935 | 64.1% vs 30.9% (**+555 converters, ~2.1×**) |
| **Leakage-controlled** (real-data ready) | `EXCLUDE_STATUS_PROXIES=1 python src/train.py` | logreg | 0.880 | 60.5% vs 30.9% |

Key takeaways:
- **The uplift is real, not just leakage.** Even with all `Status` proxies
  removed, ML reaches ~2× the converters of the deployed rules in the top 20%.
- **On clean features, LogReg ties the GBM** — the 1-SE selection rule
  automatically down-shifts to the simpler, more interpretable model. The
  data-generating process is logistic-in-engineered-features; the GBM's extra
  complexity mostly buys leakage exploitation.
- **Before real-data deployment, train with `EXCLUDE_STATUS_PROXIES=1`.** The
  0.880 / 60.5% figures are the trustworthy ones.

Business prioritisation KPIs (decile lift, capture@K, AI-vs-rules head-to-head)
are computed by `ranking_metrics.py` and stored in
`metrics.json → ranking_lift_vs_live_baseline`.

## 5. Architecture

```
data/res_open_data_sample.csv        # ČSÚ RES export (real Czech companies)
        │
        ▼
src/data_generator.py                # NACE filter + synthetic CRM + targets
        │
        ▼
data/lead_train.csv                  # ~30k labelled leads
        │
        ▼
src/train.py ──► src/models/*.pkl, metrics.json, feature_importance.csv
        │   (5-fold time-series CV · 1-SE selection · isotonic calibration · SHAP)
        │
        ├── contracts.py             # shared column/feature contracts (train ↔ serve)
        ├── feature_engineering.py   # engineered features (shared train ↔ serve)
        ├── modeling.py              # picklable LightGBM frame preprocessor
        ├── ranking_metrics.py       # decile lift / capture@K business KPIs
        ▼
src/api.py (FastAPI)   ─┐            # /score, /score/batch, /score/enrich, /health, /metrics
src/app.py (Streamlit) ─┤            # Pipeline · Lead Profile · Model Performance
        ├── baseline.py              # rule-based scorer (deployed benchmark)
        └── ares.py                  # live ARES REST enrichment
```

`contracts.py` is the single feature boundary shared by training, the API, and
the dashboard, so they cannot silently drift apart.

## 6. Quick start

### Option A — Local (Python)

```bash
pip install -r requirements.txt
python src/data_generator.py                 # build data/lead_train.csv
python src/train.py                          # train → src/models/*.pkl + metrics.json
uvicorn api:app --app-dir src --port 8000    # scoring API → http://localhost:8000/docs
python -m streamlit run src/app.py           # dashboard  → http://localhost:8501
```

**Training flags (env vars):**
- `EXCLUDE_STATUS_PROXIES=1` — train the leakage-controlled, real-data-ready model (see §4).
- `RUN_OPTUNA=1` (`OPTUNA_TRIALS=50`) — hyperparameter search when LightGBM is the best single candidate.

### Option B — Docker (one command, both services)

```bash
docker compose up               # builds image, starts dashboard + scoring API
```

| Service                | URL                                  | Purpose                           |
|------------------------|--------------------------------------|-----------------------------------|
| Streamlit dashboard    | http://localhost:8501                | Business UI                       |
| Scoring API (Swagger)  | http://localhost:8000/docs           | Interactive API explorer          |
| `POST /score`          | http://localhost:8000/score          | Salesforce Apex callout target    |
| `POST /score/enrich`   | http://localhost:8000/score/enrich   | IČO-only scoring (ARES-enriched)  |
| `GET /health`          | http://localhost:8000/health         | Liveness probe                    |
| `GET /metrics`         | http://localhost:8000/metrics        | Test-set evaluation metrics       |

Trained artifacts are baked into the image at build time; rebuild with
`docker compose up --build` after retraining.

## 7. App tour

* **📋 Pipeline** — every lead with its AI score, segment, baseline score
  and expected-win %, filterable by segment / industry / region / source.
  CSV export ready for the Salesforce data loader.
* **🏢 Lead Profile** — pick any company, run **what-if** simulations on
  the behavioural features, see the top SHAP drivers, and pull live ARES data.
* **📊 Model Performance** — CV stability, AUC / PR-AUC / F1, calibration,
  confusion matrix, AI-vs-baseline comparison, and top feature importances.

## 8. Salesforce integration plan

Delivered as three custom fields on **Lead** / **Opportunity**:

| Field | Type | Source |
|---|---|---|
| `AI_Score__c` | Number(0-100) | `model.predict_proba` × 100 |
| `AI_Segment__c` | Picklist(High/Medium/Low) | derived from `AI_Score__c` |
| `AI_Top_Driver__c` | Text | top SHAP feature (`top_drivers[0]`) |

1. **Batch (recommended for MVP):** nightly job scores open Leads, exports
   CSV, Salesforce Data Loader upserts the three fields. Zero SF code changes.
2. **Real-time:** the FastAPI `/score` (or `/score/enrich`) service is called
   from a Salesforce Apex trigger via Named Credential on Lead create/update.

## 9. Recommendation for management

* **Adopt AI scoring.** Against the rules currently in production, the model
  reaches **~2× the converters in the top 20% of leads** — the same outreach
  budget, far more real opportunities. The ranking uplift survives leakage
  removal, so it is a genuine signal, not a synthetic artifact.
* **Conditions:** retrain quarterly; keep the rule-based scorer running in
  parallel as a sanity check; monitor PR-AUC (positive class ≈ 28%) and
  calibration (ECE).
* **Risks & path to production:** outcome labels are synthetic in the MVP.
  Moving to production requires real `Closed_Won` history, retraining with
  `EXCLUDE_STATUS_PROXIES=1`, a champion/challenger test against the live
  rules, and a drift monitor.

## 10. Repository layout

```
.
├── src/                            # Application source
│   ├── app.py                      # Streamlit dashboard
│   ├── api.py                      # FastAPI scoring service (Salesforce target)
│   ├── train.py                    # Training · time-series CV · 1-SE selection · calibration
│   ├── data_generator.py           # RES open data → labelled training set
│   ├── contracts.py                # Shared column/feature contracts
│   ├── feature_engineering.py      # Engineered features (shared train ↔ serve)
│   ├── modeling.py                 # Picklable LightGBM frame preprocessor
│   ├── ranking_metrics.py          # Decile lift / capture@K business KPIs
│   ├── baseline.py                 # Rule-based scorer (deployed benchmark)
│   ├── business_helpers.py         # ROI, lift, threshold stats, PSI drift
│   ├── ares.py                     # ARES REST API wrapper
│   ├── paths.py                    # Centralised filesystem locations
│   └── models/                     # Generated by train.py
│       ├── model.pkl, preprocessor.pkl, feature_names.pkl, conv_explainer.pkl
│       ├── win_model.pkl, win_preprocessor.pkl, win_explainer.pkl, win_threshold.json
│       └── metrics.json, feature_importance.csv, val_predictions.csv
├── data/
│   ├── res_open_data_sample.csv    # Seed (Czech business registry sample)
│   └── lead_train.csv              # Generated by data_generator.py
├── assets/enehano_logo.svg
├── Dockerfile · docker-compose.yml # Single image, both services
├── requirements.txt
└── README.md
```

## 11. Tech stack

Python · scikit-learn · LightGBM · pandas · Streamlit · Plotly · FastAPI ·
Docker · SHAP · joblib · requests
