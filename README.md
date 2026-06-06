# Enehano Lead Intelligence

Predictive lead scoring for Enehano Solutions, a Salesforce consulting firm.
For each lead the project predicts:

1. P(Convert) - the chance the lead becomes an Opportunity.
2. P(Win | Converted) - the chance that Opportunity is Closed Won.
3. Expected win % = P(Convert) x P(Win) - one number for ranking.

The score is exposed both as a 0-100 number and as a High/Medium/Low segment,
ready to drop into Salesforce as a custom field.

## 1. The problem

Enehano prioritises leads with simple rules and rep intuition. It works, but it
is crude and there is no clear reason why one lead beats another. The question
this project answers: can a model trained on CRM data plus Czech company data
rank leads better than a rules sheet, and can we show by how much?

Short answer: yes. Worked top-down by the score, the model reaches roughly twice
the converters of the deployed rules for the same amount of outreach (details in
section 4).

## 2. Data

| Source | Role |
|---|---|
| RES open data (ČSÚ) | Real Czech companies: IČO, legal form, NACE, region, employee band, founding date |
| Simulated CRM telemetry | Web visits, response time, emails, meetings, forms, demos, etc. |
| Simulated outcomes | `Converted` and `Closed_Won`, generated from a noisy logistic model so there is real signal to learn but no formula to memorise |

Only sectors with realistic Salesforce-consulting demand are kept: C
(Manufacturing), J (ICT), K (Finance and Insurance), M (Professional services).
That trims ~90k RES rows to ~30k usable companies.

`ares.py` optionally calls the public ARES API to fetch address, registration
date and NACE for an IČO; the `/score/enrich` endpoint uses it so a caller only
has to send an IČO plus behavioural fields.

## 3. Models

Both stages are probability-calibrated (`CalibratedClassifierCV`, isotonic) on
top of a base estimator chosen from five candidates. The candidate is picked
with a one-standard-error rule on cross-validated AUC: among models that are
statistically tied, the simplest wins (see `train.py`).

| Model | Algorithm (auto-selected) | Target | Test AUC | PR-AUC | F1 |
|---|---|---|---|---|---|
| Conversion | Calibrated HistGradientBoosting | `Converted` | 0.935 | 0.861 | 0.757 |
| Win | Calibrated LogisticRegression (converted leads only) | `Closed_Won` | 0.665 | - | 0.530 |
| Baseline | Hand-written weighted rules (`baseline.py`) | `Converted` | 0.832* | - | 0.545 |

Conversion uplift over the rule baseline: +0.103 AUC, +0.213 F1, stable across
5 time-series folds (CV std ~0.003). Isotonic calibration brings expected
calibration error from 0.084 down to 0.019, so the percentage is a real
probability, not just a rank.

\* This 0.832 baseline uses the generator's stored `Rule_Based_Score`, which
includes the excluded `Status` field. The rules the API actually serves
(`baseline.score_frame`, no `Status`) score around 0.64 - that is the honest
production comparison and the one used for the lift figures in section 4.

### A note on leakage

Two features (`Days_in_Pipeline__c`, `Days_Since_Last_Activity__c`) are derived
from the generator's hidden `Status`, which drives the label. The model partly
reconstructs `Status` from them, which inflates the 0.935 on this synthetic data
but would not carry over to real CRM data. Training with
`EXCLUDE_STATUS_PROXIES=1` drops those features; the resulting model scores ~0.88
AUC and still reaches ~60% of converters in the top 20%. Quote the 0.88 figure
when in doubt - it is the one that should generalise.

## 4. Rule vs AI, in business terms

ROC-AUC is not what a sales team feels. The metric that matters is: working only
the top slice of leads, how many real converters do you reach? On the held-out
set (n=5981, 28% convert):

| Top 20% of leads, worked by | Converters reached |
|---|---|
| Deployed rules | ~31% |
| AI model | ~64% |

So roughly twice the opportunities for the same calls. These numbers are written
into `metrics.json` (`ranking_lift_vs_live_baseline`) on every training run, via
`ranking_metrics.py`.

## 5. Quick start

```bash
pip install -r requirements.txt
python src/data_generator.py                 # build data/lead_train.csv
python src/train.py                          # train -> src/models/*.pkl + metrics.json
uvicorn api:app --app-dir src --port 8000    # scoring API -> http://localhost:8000/docs
python -m streamlit run src/app.py           # dashboard -> http://localhost:8501
```

Training flags:
- `EXCLUDE_STATUS_PROXIES=1` trains the leakage-controlled model (see section 3).
- `RUN_OPTUNA=1` (with `OPTUNA_TRIALS`) runs a hyperparameter search when LightGBM
  is the best single candidate.

### Docker

```bash
docker compose up
```

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| Scoring API (Swagger) | http://localhost:8000/docs |
| `POST /score`, `POST /score/enrich` | scoring endpoints |
| `GET /health`, `GET /metrics` | liveness and metrics |

## 6. Dashboard

- Pipeline: every lead with its AI score, segment, baseline score and expected
  win %, filterable by segment / industry / region / source, CSV export.
- Lead Profile: pick a company, run what-if changes on the behavioural fields,
  see the top SHAP drivers, pull live ARES data.
- Model Performance: CV stability, AUC / PR-AUC / F1, calibration, confusion
  matrix, rule-vs-AI comparison, feature importance.

## 7. Salesforce integration

Three custom fields on Lead / Opportunity:

| Field | Type | Source |
|---|---|---|
| `AI_Score__c` | Number(0-100) | `predict_proba` x 100 |
| `AI_Segment__c` | Picklist (High/Medium/Low) | derived from the score |
| `AI_Top_Driver__c` | Text | top SHAP feature |

Two delivery options:

1. Batch (built): `sf_sync.py` scores open leads in-process and writes the fields
   back via the Bulk API. No Salesforce code, no hosting. `create_test_leads.py`
   seeds an org with sample leads. See `docs/SALESFORCE_INTEGRATION.md`.
2. Real time: call `/score` (or `/score/enrich`) from an Apex trigger or Flow on
   lead create/update.

Everything runs on a free Salesforce Developer Edition org.

## 8. Recommendation

Adopt AI scoring, but stage it. Against the deployed rules it reaches roughly
twice the converters in the top 20%, and that uplift survives leakage removal.
Before full rollout: retrain on real history with `EXCLUDE_STATUS_PROXIES=1`, run
a champion/challenger A/B against the current rules to confirm lift on real
conversions, keep the rule scorer running in parallel, and monitor PR-AUC and
calibration. Outcome labels are synthetic in the MVP, so real-data validation is
the main precondition. Full analysis in `docs/EVALUATION_AND_DEPLOYMENT.md`.

## 9. Repository layout

```
src/
  app.py                 Streamlit dashboard
  api.py                 FastAPI scoring service
  train.py               training, time-series CV, selection, calibration
  data_generator.py      RES open data -> labelled training set
  contracts.py           shared column / feature definitions
  feature_engineering.py engineered features (training and inference)
  modeling.py            LightGBM frame preprocessor
  ranking_metrics.py     decile lift / capture@K
  baseline.py            rule-based scorer
  ares.py                ARES REST wrapper
  sf_common.py           Salesforce connection + field config
  sf_sync.py             batch scoring sync to Salesforce
  create_test_leads.py   seed an org with sample leads
  paths.py               filesystem locations
  models/                generated by train.py (model.pkl, metrics.json, ...)
data/
  res_open_data_sample.csv
  lead_train.csv         generated by data_generator.py
docs/                    integration plan, evaluation + deployment
```

## 10. Stack

Python, scikit-learn, LightGBM, pandas, Streamlit, Plotly, FastAPI, Docker,
SHAP, joblib, simple-salesforce.
