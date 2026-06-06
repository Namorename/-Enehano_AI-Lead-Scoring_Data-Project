# Evaluation, Trade-offs & Deployment Recommendations

Companion to the lead-scoring MVP. Covers three brief requirements: **user
testing**, **trade-offs**, and **deployment recommendations**. Written to be
lifted directly into the management report / presentation.

---

## 1. User testing

### 1.1 What has been tested so far (functional / technical) - done
The system has been verified to *work correctly*:
- **API**: `/score`, `/score/batch`, `/score/enrich`, `/health` return correct
 results; empty-batch and explainer-failure edge cases handled without 500s.
- **Salesforce integration**: 20 leads created, scored, and written back
 end-to-end in a live Developer Edition org (`create_test_leads.py` + `sf_sync.py`).
- **Model validation**: time-separated hold-out, AUC/PR-AUC/precision/recall,
 calibration (ECE 0.084->0.019), rule-vs-AI ranking lift.

> This is **system testing**, not end-user testing. It proves the tool is
> correct, not that salespeople find it usable and trustworthy.

### 1.2 The gap (honest) - not yet done
**No end-user (sales rep / manager) testing has been performed.** No usability
sessions, no rep feedback, no measurement of whether the scores change real
prioritization decisions. This must be disclosed, not implied.

### 1.3 Proposed user-testing plan
| Dimension | Plan |
|---|---|
| **Who** | 3-5 sales reps (primary users) + 1-2 sales managers (dashboard users) |
| **What to test** | Can a rep *find* the AI Score on a Lead? *Understand* High/Med/Low? Does the *top driver* build trust? Would it change *who they call first*? |
| **Method** | Moderated think-aloud sessions on (a) the Salesforce Lead view and (b) the Streamlit dashboard; task-based, e.g. *"pick the 5 leads you'd call first."* |
| **Metrics** | Task success rate, time-to-decision, trust rating (1-5), **SUS** (System Usability Scale), qualitative notes |
| **Acceptance criteria** | >=80% locate & correctly interpret the score; reps judge the ranking "sensible" in >=70% of cases; mean trust >=4/5 |

### 1.4 What can be run *now*, before a sales team is available
Two lightweight tests that need no production users:
1. **Expert face-validity review** - show scored leads to an Enehano consultant /
 domain expert: *"does this ranking match your intuition?"* Captures obvious
 errors cheaply.
2. **Blind A/B ranking** - present two ranked lists (AI vs rules) without labels
 and ask *"which list would you rather work?"* Measures perceived value.

The **definitive** user test is a production **champion/challenger A/B** (below)
- that measures behaviour change and real conversions, not just opinions.

---

## 2. Trade-offs

### 2.1 Rule-based vs AI scoring (the core comparison)
| | Rule-based | AI predictive |
|---|---|---|
| Accuracy / ranking lift | Lower (captures ~31% of converters in top 20%) | Higher (~60-64%) |
| Transparency | Total - anyone can read the rules | High *with SHAP*, lower than rules |
| Data needed | None | Labeled history |
| Cold start | Works day 1 | Needs data first |
| Captures interactions | No | Yes |
| Maintenance | Manual edits, drifts silently | Retrain on a cadence |
| Calibrated probabilities | No | Yes |
| **Verdict** | Simple, transparent floor | Higher value, needs data + MLOps |

-> **Recommended posture: AI as primary, rules retained as a fallback / sanity
check** (the project already runs both side-by-side).

### 2.2 Model complexity: simple vs complex
On *clean* features, **logistic regression ~ gradient boosting** (~0.88 AUC).
We adopted a **one-standard-error rule**: among statistically tied models, pick
the simplest. Trade-off accepted: forgo a fraction of a point of AUC for
interpretability, robustness, and lower maintenance. Complexity you can't justify
is a liability, not an asset.

### 2.3 Accuracy vs interpretability
Gradient boosting is less transparent than rules/logreg. Mitigated by **SHAP top
drivers** (per-lead "why") + **calibration** (the % is a real probability). Net:
near-GBM accuracy with rule-like explainability.

### 2.4 Honest performance vs headline performance (leakage)
The 0.935 AUC is partly inflated by `Status`-proxy leakage. The
**leakage-controlled** model scores ~0.88 AUC / ~60% capture. Trade-off: a
flashier number vs a number that survives contact with real data. **Recommend
quoting the honest ~0.88 / ~60%** and presenting the leakage finding as evidence
of rigor.

### 2.5 Synthetic vs real data
Synthetic RES-based data let us build and validate the *entire pipeline* now,
without waiting for CRM access. Trade-off: results are an *estimate* until
re-validated on real history. Acceptable for an MVP; mandatory to revisit before
production.

### 2.6 Batch vs real-time integration
| | Batch (built) | Real-time |
|---|---|---|
| Latency | Minutes-hours | Seconds |
| Hosting | None needed | Public HTTPS service |
| Apex | None | Apex/Flow callout |
| Effort / cost | Low | Medium |
| Best for | MVP, bulk re-scoring | "Score on save" |

-> **Start batch, add real-time once value is proven.**

### 2.7 Build (custom) vs buy (Salesforce Einstein)
A question Enehano *will* ask:
- **Einstein Lead Scoring**: out-of-the-box, native - **but** requires higher
 Salesforce editions + Einstein licensing, enough historical volume, and offers
 less model transparency/control.
- **Custom (this project)**: transparent, portable, free to prototype, tunable,
 and a **reusable consulting accelerator Enehano can resell to its clients**.
- **Honest take**: for a large org already paying for Einstein, buy may be
 simpler; for transparency, control, lower entry cost, and a productizable asset,
 build wins. They are not mutually exclusive - custom is a great way to *prove
 value* before committing to Einstein licensing.

### 2.8 Trade-off summary
The recurring theme: **we consistently chose the honest, simpler, lower-risk
option** (clean model over leaky, simplest-tied model, batch before real-time,
rules retained as fallback). That is the defensible engineering posture.

---

## 3. Deployment recommendations

### 3.1 Whether to deploy - **Yes, conditionally**
The ranking lift is real and survives leakage control (~2x converters reached in
the top 20% vs deployed rules). **Recommendation: proceed to a real-data pilot**,
but do **not** flip to full production until an A/B test confirms lift on *real*
conversions.

### 3.2 Phased rollout
1. **Backtest on real history** - retrain on real CRM data with
 `EXCLUDE_STATUS_PROXIES=1`; confirm AUC/lift hold.
2. **Shadow / champion-challenger** - score real leads but split: half worked by
 rules, half by AI; compare actual conversions after a quarter.
3. **Pilot** - one sales team, batch sync + dashboard, with user testing (§1).
4. **Full deploy** - all teams, add real-time scoring + monitoring.

### 3.3 Architecture options (pick per phase)
- **Batch sync** (built) - nightly, no hosting, zero Salesforce code. *MVP.*
- **Real-time Apex/Flow callout** - score on lead create/edit.
- **External Services (no-code)** - import the API's OpenAPI into Flow.
(Full detail in `docs/SALESFORCE_INTEGRATION.md`.)

### 3.4 Prerequisites for production
- 6-12 months of real labeled outcomes (`Converted`, ideally `Closed_Won`).
- Field mapping CRM -> model inputs; data-quality baseline.
- API auth + hosting **if** going real-time (batch needs neither).
- Stakeholder buy-in + the user testing from §1.

### 3.5 Risks & mitigations
| Risk | Mitigation |
|---|---|
| Data leakage inflates expectations | `EXCLUDE_STATUS_PROXIES` control; quote honest numbers |
| Model drift over time | Monitor PSI (`business_helpers.py`) + scheduled retrain |
| Low rep adoption | Keep rules visible, SHAP explanations, user testing, change management |
| Synthetic ≠ real performance | Real-data backtest + A/B before full rollout |
| Edition/licensing limits (real-time API) | Confirm prod Salesforce edition early; batch path avoids it |

### 3.6 Monitoring & maintenance
Track over time: **ranking lift / AUC**, **score drift (PSI)**, **calibration
(ECE)**, and **adoption** (are reps working High-segment leads first?). Retrain
quarterly or when drift exceeds a threshold; stamp `AI_Model_Version__c`.

### 3.7 Cost
- **Prototype/MVP**: €0 - free Developer Edition + open RES data + open-source stack.
- **Production**: data engineering + (optional) hosting for real-time +
 (optional) Einstein licensing if buy is chosen. Dominated by people-time, not tooling.

### 3.8 Go / no-go criteria
**Deploy fully if** the champion/challenger A/B shows a statistically significant
lift in *actual* conversions over the current rules **and** pilot reps adopt the
scores (trust >=4/5, working High leads first). Otherwise iterate.

## Summary

The tool works (system testing) and the model beats the rules on held-out data.
The open items are end-user testing and a real-data A/B test, both scoped above.
The recommendation is a phased rollout: batch pilot first, full deployment only
after measured lift on real conversions.
