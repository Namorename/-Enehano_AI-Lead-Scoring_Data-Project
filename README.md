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
rule-based scoring sheet?

## 2. Data

| Source | Role |
|---|---|
| **RES open data** (ČSÚ) | Real Czech companies — IČO, legal form, NACE, region, employee category, founding date |
| **Synthetic CRM telemetry** | Web visits, response time, emails, meetings, forms, demo, LinkedIn, etc. |
| **Synthetic outcomes** | `Converted` (Lead → Opportunity) and `Closed_Won` (Opportunity outcome), generated via a noisy logistic model so the AI has real signal to learn but cannot trivially memorise a formula |

**NACE filter.** Only sectors with realistic Salesforce-consulting demand
are kept: **C** (Manufacturing), **J** (ICT), **K** (Finance & Insurance),
**M** (Professional services). This shrinks ~90 k RES rows to ~30 k usable
companies — exactly what the brief asks for.

**ARES enrichment (bonus).** `ares.py` calls the public ARES REST API
(`ares.gov.cz`) to fetch authoritative address, registration date and NACE
for any IČO. Hooked into the Lead Profile tab with a 1-hour cache.

## 3. Models

| Model | Algorithm | Target | Test AUC | CV AUC (5-fold) | F1 |
|---|---|---|---|---|---|
| Conversion | RandomForest (200 trees, depth 12) | `Converted` | **0.892** | 0.892 ± 0.003 | 0.66 |
| Win | RandomForest (200 trees, depth 12) | `Closed_Won` (on converted only) | 0.714 | 0.740 ± 0.011 | 0.74 |
| **Baseline** (rules) | Hand-written weighted sum (`baseline.py`) | `Converted` | 0.766 | — | 0.46 |

**AI vs. baseline uplift on the conversion task: +0.126 AUC, +0.20 F1.**

The cross-validation standard deviation (~0.003) shows the model is
**stable** — performance does not depend on a lucky train/test split.

### Top conversion drivers (RandomForest feature importance)

1. `Web_Interactions` (26 %) — strongest intent signal
2. `Time_to_First_Response_h` (13 %) — speed kills or makes deals
3. `Days_Since_Last_Activity` (9 %) — pipeline freshness
4. `LeadSource = Cold Call` (6 %) — negative driver
5. `Email_Clicks` (6 %)
6. `NumberOfEmployees` / `Annual_Revenue` (~5 % each) — company-fit
7. `Meetings_Held`, `Demo_Requested` — late-funnel commitment

This matches sales intuition, which is the point: the model is
**explainable**, not a black box.

## 4. Architecture

```
res_open_data_sample.csv        # ČSÚ RES export (real Czech companies)
        │
        ▼
data_generator.py               # NACE filter + synthetic CRM + targets
        │
        ▼
lead_train.csv                  # ~30k labelled leads
        │
        ▼
train.py ──► model.pkl, win_model.pkl, preprocessor.pkl, ...
              metrics.json, feature_importance.csv
        │
        ▼
app.py (Streamlit)              # Pipeline · Profile · Performance
   ├── baseline.py              # rule-based scorer (benchmark)
   └── ares.py                  # live ARES REST enrichment
```

## 5. Quick start

### Option A — Local (Python)

```bash
pip install -r requirements.txt
python src/data_generator.py --res data/res_open_data_sample.csv             # build data/lead_train.csv
python src/train.py                      # train models → models/*.pkl + metrics.json
streamlit run src/app.py                 # dashboard  → http://localhost:8501
uvicorn api:app --app-dir src --port 8000  # scoring API → http://localhost:8000/docs
```

### Option B — Docker (one command, both services)

```bash
docker compose up               # builds image, starts dashboard + scoring API
```

After startup:

| Service                | URL                                  | Purpose                           |
|------------------------|--------------------------------------|-----------------------------------|
| Streamlit dashboard    | http://localhost:8501                | Business UI                       |
| Scoring API (Swagger)  | http://localhost:8000/docs           | Interactive API explorer          |
| `POST /score`          | http://localhost:8000/score          | Salesforce Apex callout target    |
| `GET /health`          | http://localhost:8000/health         | Liveness probe                    |
| `GET /metrics`         | http://localhost:8000/metrics        | Test-set evaluation metrics       |

Stop with `docker compose down`. Trained model artifacts (`*.pkl`, `metrics.json`)
are baked into the image at build time; rebuild with `docker compose up --build`
after retraining.

## 6. App tour

* **📋 Pipeline** — every lead with its AI score, segment, baseline score
  and expected-win %, filterable by segment / industry / region / source.
  CSV export ready for Salesforce data loader.
* **🏢 Lead Profile** — pick any company, run **what-if** simulations on
  the behavioural features (response time, web visits, meetings, …),
  see top drivers behind the score, and pull live ARES data.
* **📊 Model Performance** — CV stability, AUC / PR-AUC / F1, confusion
  matrix, AI-vs-baseline comparison, top-20 feature importances.

## 7. Salesforce integration plan

The model is delivered as three custom fields on **Lead** /
**Opportunity**:

| Field | Type | Source |
|---|---|---|
| `AI_Score__c` | Number(0-100) | `model.predict_proba` × 100 |
| `AI_Segment__c` | Picklist(High/Medium/Low) | derived from `AI_Score__c` |
| `AI_Top_Driver__c` | Text | top SHAP/importance feature |

Two delivery options:

1. **Batch (recommended for MVP):** nightly job runs `train.py` outputs
   against open Leads, exports CSV, Salesforce Data Loader upserts the
   three fields. Zero SF code changes.
2. **Real-time:** wrap `model.pkl` in a small Flask/FastAPI service,
   call from a Salesforce Apex trigger via Named Credential on Lead
   create/update.

## 8. Recommendation for management

* **Adopt AI scoring.** The +0.13 AUC uplift over the existing rule-based
  approach is large and stable across folds. Even with a conservative
  20 % redirection of rep time toward High-segment leads, the expected
  pipeline-conversion lift is in the **+15 to +25 %** range.
* **Conditions:** retrain quarterly; keep the rule-based scorer running
  in parallel as a sanity check; monitor PR-AUC (positive class is only
  ~23 % of leads).
* **Risks:** synthetic outcome labels in the MVP — moving to production
  requires 6–12 months of real `Closed_Won` history for retraining.

## 9. Repository layout

```
.
├── src/                            # Application source
│   ├── app.py                      # Streamlit dashboard (6 tabs)
│   ├── api.py                      # FastAPI scoring service (Salesforce target)
│   ├── train.py                    # Training + 5-fold CV + metrics export
│   ├── data_generator.py           # RES open data → labelled training set
│   ├── baseline.py                 # Rule-based scorer (benchmark)
│   ├── business_helpers.py         # ROI, lift, threshold stats, PSI drift
│   ├── ares.py                     # ARES REST API wrapper
│   └── paths.py                    # Centralised filesystem locations
├── data/
│   ├── res_open_data_sample.csv    # Seed (Czech business registry sample)
│   └── lead_train.csv              # Generated by data_generator.py
├── models/                         # Generated by train.py
│   ├── model.pkl, preprocessor.pkl, feature_names.pkl
│   ├── win_model.pkl, win_preprocessor.pkl
│   └── metrics.json, feature_importance.csv, val_predictions.csv
├── assets/
│   └── enehano_logo.svg
├── Dockerfile                      # Single image, both services
├── docker-compose.yml              # One-command stack: Streamlit + API
├── requirements.txt
└── README.md
```

## 10. Tech stack

Python · scikit-learn · pandas · Streamlit · Plotly · FastAPI · Docker · SHAP · joblib · requests

