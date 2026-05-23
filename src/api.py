"""
api.py
======
FastAPI scoring service — exposes the trained models over HTTP.

Explainability:
  Per-lead top_drivers are computed with shap.TreeExplainer (local SHAP
  values, positive class).  The explainer is loaded from disk once at
  startup and reused across requests — no background dataset is required
  for RandomForestClassifier (tree_path_dependent mode).  Expected latency
  is ~8-15 ms per lead on a typical server CPU.

Column contract:
  - LeadInput uses Legal_Form_Label (string), NOT Legal_Form_Code (which is
    stored as string "112"/"121" in the CSV but is not a useful API input).
  - Region field maps to the 'Region_Name' / 'Region' column used by the model.
  - All bool fields (Demo_Requested__c etc.) declared as int (0/1) — matches
    the int dtype the preprocessor was trained on.

Enrichment flow (/score/enrich):
  The "Thick Backend" endpoint accepts only an IČO and behavioral metrics.
  Firmographic data is fetched from the Czech ARES registry, cached in
  ARES_CACHE (process-level dict keyed by normalised IČO), mapped to the
  LeadInput schema, and then passed through the standard scoring pipeline.
  This keeps the frontend payload minimal and avoids direct ARES calls from
  the client, eliminating the rate-limiting risk entirely.

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import ares as ares_client
import baseline
from paths import (
    CONV_MODEL, CONV_PRE, CONV_FEAT, CONV_EXPL,
    WIN_MODEL, WIN_PRE, WIN_EXPL,
    METRICS_JSON,
)

# ── Asset loading ─────────────────────────────────────────────────────────────
ASSETS: dict = {}
METRICS: dict = {}

MODEL_PATHS = [CONV_MODEL, CONV_PRE, CONV_FEAT, CONV_EXPL, WIN_MODEL, WIN_PRE, WIN_EXPL]

# Columns excluded from training — must match train.py DROP_COLS exactly.
_API_EXCLUDE = {
    "Lead_ID", "IČO", "Company_Name",
    "Converted", "Closed_Won", "Status",
    "Conversion_Probability", "Rule_Based_Score", "Rule_Based_Segment",
    "Win_Probability__c", "IČO_Duplicate_Flag", "CreatedDate", "LastActivityDate",
}


def _load_assets() -> None:
    if not all(p.exists() for p in MODEL_PATHS):
        raise RuntimeError(
            "Trained model artifacts not found. Run `python src/train.py` first."
        )
    ASSETS["conv_model"] = joblib.load(CONV_MODEL)
    ASSETS["conv_pre"]   = joblib.load(CONV_PRE)
    ASSETS["feat_names"] = joblib.load(CONV_FEAT)
    # Pre-loaded TreeExplainer — built once at startup, reused for every request.
    # shap.TreeExplainer.shap_values() is thread-safe for read-only inference.
    ASSETS["conv_expl"]  = joblib.load(CONV_EXPL)
    ASSETS["win_model"]  = joblib.load(WIN_MODEL)
    ASSETS["win_pre"]    = joblib.load(WIN_PRE)
    ASSETS["win_expl"]   = joblib.load(WIN_EXPL)
    if METRICS_JSON.exists():
        METRICS.update(json.loads(METRICS_JSON.read_text()))


# ── ARES in-memory cache ───────────────────────────────────────────────────────
# Keyed by zero-padded 8-digit IČO string (matches ares_client.lookup normalisation).
# Lifetime is the process lifetime; restart the server to invalidate stale entries.
# For multi-worker deployments replace with Redis or a shared external store.
ARES_CACHE: dict[str, dict] = {}


# ── ARES lookup helpers ────────────────────────────────────────────────────────

# Maps common Czech legal-form codes (pravniForma) to human-readable labels.
# Unknown codes fall back gracefully to the raw code string.
_LEGAL_FORM_LABELS: dict[str, str] = {
    "101": "Fyzická osoba podnikající",
    "111": "Veřejná obchodní společnost",
    "112": "s.r.o.",
    "113": "Společnost komanditní",
    "121": "a.s.",
    "141": "Obecně prospěšná společnost",
    "145": "Ústav",
    "151": "Spolek",
    "161": "Nadace",
    "205": "Státní podnik",
    "231": "Příspěvková organizace",
    "325": "Veřejná vysoká škola",
    "421": "Odštěpný závod zahraniční osoby",
    "601": "Církev",
    "706": "Územní samosprávný celek",
    "801": "Organizační složka státu",
}

# Maps Czech region names (nazevKraje) to NUTS-3 region codes.
# Covers all 14 NUTS-3 regions; unknown names fall back to "CZ000".
_REGION_CODE_MAP: dict[str, str] = {
    "Hlavní město Praha":   "CZ010",
    "Středočeský kraj":     "CZ020",
    "Jihočeský kraj":       "CZ031",
    "Plzeňský kraj":        "CZ032",
    "Karlovarský kraj":     "CZ041",
    "Ústecký kraj":         "CZ042",
    "Liberecký kraj":       "CZ051",
    "Královéhradecký kraj": "CZ052",
    "Pardubický kraj":      "CZ053",
    "Kraj Vysočina":        "CZ063",
    "Jihomoravský kraj":    "CZ064",
    "Olomoucký kraj":       "CZ071",
    "Zlínský kraj":         "CZ072",
    "Moravskoslezský kraj": "CZ080",
}

# Maps the leading two-digit NACE division number to the NACE section letter
# and its English description.  Each tuple is (division_range, letter, label).
_NACE_SECTION_MAP: list[tuple[range, str, str]] = [
    (range(1,   4),  "A", "Agriculture, forestry and fishing"),
    (range(5,   10), "B", "Mining and quarrying"),
    (range(10,  34), "C", "Manufacturing"),
    (range(35,  36), "D", "Electricity, gas, steam and air conditioning supply"),
    (range(36,  40), "E", "Water supply; sewerage, waste management and remediation"),
    (range(41,  44), "F", "Construction"),
    (range(45,  48), "G", "Wholesale and retail trade"),
    (range(49,  54), "H", "Transportation and storage"),
    (range(55,  57), "I", "Accommodation and food service activities"),
    (range(58,  64), "J", "Information and communication"),
    (range(64,  67), "K", "Financial and insurance activities"),
    (range(68,  69), "L", "Real estate activities"),
    (range(69,  76), "M", "Professional, scientific and technical activities"),
    (range(77,  83), "N", "Administrative and support service activities"),
    (range(84,  85), "O", "Public administration and defence"),
    (range(85,  86), "P", "Education"),
    (range(86,  89), "Q", "Human health and social work activities"),
    (range(90,  94), "R", "Arts, entertainment and recreation"),
    (range(94,  97), "S", "Other service activities"),
    (range(97,  99), "T", "Activities of households as employers"),
    (range(99, 100), "U", "Activities of extraterritorial organisations"),
]

_CURRENT_YEAR: int = datetime.date.today().year


def _nace_to_section(nace_code: str | None) -> tuple[str, str]:
    """
    Convert a raw NACE code string (e.g. '62.01', '47.11') to a (letter, label)
    section pair.  Returns the 'J / Information and communication' fallback when
    the code is absent or cannot be parsed.
    """
    if not nace_code:
        return "J", "Information and communication"
    try:
        division = int(str(nace_code).split(".")[0].strip())
    except (ValueError, IndexError):
        return "J", "Information and communication"
    for div_range, letter, label in _NACE_SECTION_MAP:
        if division in div_range:
            return letter, label
    return "J", "Information and communication"


def _parse_founding_year(date_str: str | None) -> int:
    """
    Extract the 4-digit founding year from an ISO-8601 date string returned by
    ARES (e.g. '2005-03-15').  Returns 2000 as a safe default on parse failure.
    """
    if not date_str:
        return 2000
    try:
        return int(str(date_str)[:4])
    except (ValueError, TypeError):
        return 2000


def _fetch_ares(ico: str) -> dict:
    """
    Return ARES firmographic data for *ico*, consulting ARES_CACHE first.

    On a cache miss the ARES REST API is called via ares_client.lookup().
    The result (or a sentinel empty dict on failure) is stored in ARES_CACHE
    so that repeated calls for the same company never hit the network again
    within the same process lifetime.

    Raises HTTPException(502) when the ARES API is unreachable and the IČO
    is not already cached, because scoring without firmographics would produce
    meaningless results.
    """
    normalised = str(ico).strip().zfill(8)
    if normalised in ARES_CACHE:
        return ARES_CACHE[normalised]

    result = ares_client.lookup(normalised)
    if result is None:
        raise HTTPException(
            status_code=502,
            detail=(
                f"ARES lookup failed for IČO '{normalised}'. "
                "The registry may be temporarily unavailable — retry in a few seconds."
            ),
        )

    ARES_CACHE[normalised] = result
    return result


def _ares_to_firmographics(ares_data: dict) -> dict:
    """
    Map a flat ARES result dict (as returned by ares_client.lookup) to the
    firmographic fields expected by LeadInput.

    Mapping decisions
    -----------------
    legal_form  → Legal_Form_Code (raw code) + Legal_Form_Label (resolved label)
    region      → Region_Name + Region_Code (via _REGION_CODE_MAP)
    primary_nace→ CZ_NACE_Section + CZ_NACE_Section_Label (via _nace_to_section)
    city        → used as Region_Name fallback when region is absent
    registration_date → Founding_Year + Company_Age_Years (derived)

    All fields carry safe defaults so scoring is never blocked by partial data.
    """
    legal_code  = str(ares_data.get("legal_form") or "112")
    legal_label = _LEGAL_FORM_LABELS.get(legal_code, legal_code)

    region_name = ares_data.get("region") or ares_data.get("city") or "Praha"
    region_code = _REGION_CODE_MAP.get(region_name, "CZ000")

    nace_section, nace_label = _nace_to_section(ares_data.get("primary_nace"))

    founding_year    = _parse_founding_year(ares_data.get("registration_date"))
    company_age      = max(0, _CURRENT_YEAR - founding_year)

    return {
        "CZ_NACE_Section":       nace_section,
        "CZ_NACE_Section_Label": nace_label,
        "Region_Code":           region_code,
        "Region_Name":           region_name,
        "Legal_Form_Code":       legal_code,
        "Legal_Form_Label":      legal_label,
        "Founding_Year":         founding_year,
        "Company_Age_Years":     company_age,
    }


# ── Schema ────────────────────────────────────────────────────────────────────
class LeadInput(BaseModel):
    """
    Full set of Salesforce Lead fields required for model scoring.

    Field notes:
    - Bool flags are int (0/1) — matches the int64 dtype from data_generator.py.
    - Legal_Form_Code is a string category code — matches str dtype in CSV.
    - Region_Name is the actual CSV column name; 'Region' is an alias added
      at scoring time for compatibility with baseline.score_frame.
    """
    # Firmographics
    CZ_NACE_Section:        str   = "J"
    CZ_NACE_Section_Label:  str   = "Information and communication"
    Industry:               str   = "Technology"
    Region_Code:            str   = "CZ010"
    Region_Name:            str   = "Praha"
    Legal_Form_Code:        str   = "112"               # string category, not int
    Legal_Form_Label:       str   = "s.r.o."
    Founding_Year:          int   = 2010
    Company_Age_Years:      int   = 15
    Size_Band:              str   = "Small"
    NumberOfEmployees:      int   = 25
    Annual_Revenue_MCZK__c: float = 30.0
    LeadSource:             str   = "Web"
    Rating:                 str   = "Warm"
    Owner:                  str   = "Martin Novák"
    # Behavioral signals (bool flags as int — matches preprocessor)
    Time_to_First_Response_h__c: float = 24.0
    Web_Interactions__c:         int   = 0
    Email_Opens__c:              int   = 0
    Email_Clicks__c:             int   = 0
    Emails_Sent__c:              int   = 0
    Email_Open_Rate__c:          float = 0.0
    Form_Submissions__c:         int   = 0
    Demo_Requested__c:           int   = 0    # int, not bool
    Calls_Made__c:               int   = 0
    Meetings_Held__c:            int   = 0
    Proposal_Sent__c:            int   = 0    # int, not bool
    Content_Downloads__c:        int   = 0
    Chatbot_Interactions__c:     int   = 0
    LinkedIn_Viewed__c:          int   = 0    # int, not bool
    Days_Since_Last_Activity__c: int   = 30
    Days_in_Pipeline__c:         int   = 30

    @field_validator("Legal_Form_Code", mode="before")
    @classmethod
    def _coerce_legal_form_code(cls, value) -> str:
        return str(value).split(".")[0].strip()


class EnrichLeadInput(BaseModel):
    """
    Thin frontend payload for the /score/enrich endpoint.

    The caller supplies only the company IČO and CRM/behavioral metrics.
    All firmographic fields (NACE, region, legal form, founding year, …) are
    resolved server-side by querying the ARES registry, so the frontend never
    needs to know about those fields.

    Fields that are Salesforce-sourced but not available from ARES
    (Industry, Size_Band, NumberOfEmployees, Annual_Revenue_MCZK__c,
    LeadSource, Rating, Owner) are kept here as optional inputs so callers
    can override the defaults when the data is already present in the CRM.
    """
    # Company identifier — used to fetch firmographics from ARES.
    ico: str = Field(..., description="8-digit Czech company registration number (IČO).")

    # Salesforce CRM fields not covered by ARES (all optional with sensible defaults).
    Industry:               str   = "Technology"
    Size_Band:              str   = "Small"
    NumberOfEmployees:      int   = 25
    Annual_Revenue_MCZK__c: float = 30.0
    LeadSource:             str   = "Web"
    Rating:                 str   = "Warm"
    Owner:                  str   = "Unknown"

    # Behavioral / interaction signals (all optional — default to zero activity).
    Time_to_First_Response_h__c: float = 24.0
    Web_Interactions__c:         int   = 0
    Email_Opens__c:              int   = 0
    Email_Clicks__c:             int   = 0
    Emails_Sent__c:              int   = 0
    Email_Open_Rate__c:          float = 0.0
    Form_Submissions__c:         int   = 0
    Demo_Requested__c:           int   = 0
    Calls_Made__c:               int   = 0
    Meetings_Held__c:            int   = 0
    Proposal_Sent__c:            int   = 0
    Content_Downloads__c:        int   = 0
    Chatbot_Interactions__c:     int   = 0
    LinkedIn_Viewed__c:          int   = 0
    Days_Since_Last_Activity__c: int   = 30
    Days_in_Pipeline__c:         int   = 30


class ScoreOutput(BaseModel):
    ai_score:         float     = Field(..., description="0-100 conversion probability score")
    segment:          str       = Field(..., description="High / Medium / Low")
    expected_win_pct: float     = Field(..., description="Convert × Win as 0-100")
    p_win_if_converted_pct: float = Field(..., description="0-100 probability of winning after conversion")
    rule_score:       int       = Field(..., description="Baseline rule-based score")
    top_drivers:      list[str] = Field(default_factory=list, description="Top 3 features pushing the score up")


class BatchInput(BaseModel):
    leads: list[LeadInput]
    include_drivers: bool = False


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Enehano Lead Scoring API",
    version="1.2.0",
    description=(
        "AI lead scoring for Salesforce. "
        "POST /score for full-field scoring; "
        "POST /score/enrich for IČO-only enriched scoring."
    ),
)


@app.on_event("startup")
def _startup() -> None:
    _load_assets()


def _segment(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def _positive_shap_row(shap_values, row_idx: int) -> np.ndarray:
    """
    Return the positive-class SHAP vector for one row.

    Different sklearn tree explainers return different shapes:
    - RandomForestClassifier: (n_rows, n_features, n_classes)
    - GradientBoostingClassifier: (n_rows, n_features)
    - Older SHAP versions: list[class] of (n_rows, n_features)
    """
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        return np.asarray(values)[row_idx]

    values = np.asarray(shap_values)
    if values.ndim == 3:
        class_idx = 1 if values.shape[2] > 1 else 0
        return values[row_idx, :, class_idx]
    if values.ndim == 2:
        return values[row_idx]
    raise ValueError(f"Unsupported SHAP values shape: {values.shape}")


def _score_dataframe(df: pd.DataFrame, include_drivers: bool = True) -> list[ScoreOutput]:
    # Add 'Region' alias so baseline.score_frame and any region-aware logic work.
    if "Region_Name" in df.columns and "Region" not in df.columns:
        df = df.copy()
        df["Region"] = df["Region_Name"]

    p_conv = ASSETS["conv_model"].predict_proba(ASSETS["conv_pre"].transform(df))[:, 1]
    p_win  = ASSETS["win_model"].predict_proba(ASSETS["win_pre"].transform(df))[:, 1]
    rule   = baseline.score_frame(df)

    shap_vals = None
    feat_names = ASSETS["feat_names"]
    if include_drivers:
        # Local SHAP explanations are intentionally optional. They are useful
        # for a single lead profile, but too expensive for 30k-row dashboard
        # scoring where only numeric ranking fields are needed.
        X_transformed = ASSETS["conv_pre"].transform(df)
        shap_vals = ASSETS["conv_expl"].shap_values(
            X_transformed, check_additivity=False
        )

    out: list[ScoreOutput] = []
    for i in range(len(df)):
        ai = round(float(p_conv[i] * 100), 1)

        drivers: list[str] = []
        if shap_vals is not None:
            sv_row = _positive_shap_row(shap_vals, i)
            ranked = list(np.argsort(sv_row)[::-1])
            positive_idx = [j for j in ranked if sv_row[j] > 0]
            remaining = [j for j in ranked if j not in positive_idx]
            top_idx = (positive_idx + remaining)[:3]
            drivers = [feat_names[j].split("__")[0].replace("_", " ") for j in top_idx]

        out.append(ScoreOutput(
            ai_score=ai,
            segment=_segment(ai),
            expected_win_pct=round(float(p_conv[i] * p_win[i] * 100), 1),
            p_win_if_converted_pct=round(float(p_win[i] * 100), 1),
            rule_score=int(rule.iloc[i]) if hasattr(rule, "iloc") else int(rule[i]),
            top_drivers=drivers,
        ))
    return out


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": bool(ASSETS),
        "version": app.version,
        "ares_cache_size": len(ARES_CACHE),
    }


@app.get("/metrics")
def metrics_endpoint() -> dict:
    if not METRICS:
        raise HTTPException(503, "metrics.json not available — run train.py first.")
    return METRICS


@app.post("/score", response_model=ScoreOutput)
def score(lead: LeadInput) -> ScoreOutput:
    """Score a single lead using the full LeadInput schema (all fields required)."""
    df = pd.DataFrame([lead.model_dump()])
    return _score_dataframe(df)[0]

@app.post("/score/batch", response_model=list[ScoreOutput])
def score_batch(payload: BatchInput) -> list[ScoreOutput]:
    """Score up to 500 leads in one call using the full LeadInput schema."""
    if len(payload.leads) > 500:
        raise HTTPException(400, "Maximum 500 leads per batch.")
    df = pd.DataFrame([lead.model_dump() for lead in payload.leads])
    return _score_dataframe(df, include_drivers=payload.include_drivers)


@app.post("/score/enrich", response_model=ScoreOutput)
def score_enrich(payload: EnrichLeadInput) -> ScoreOutput:
    """
    Thick-backend scoring endpoint — the caller supplies only an IČO and
    behavioral metrics; firmographic data is resolved automatically.

    Flow
    ----
    1. Normalise the IČO to an 8-digit zero-padded string.
    2. Check ARES_CACHE; call ares_client.lookup() on a cache miss and store
       the result so subsequent requests for the same company are free.
    3. Map ARES fields to LeadInput firmographic fields via _ares_to_firmographics().
    4. Merge firmographics with the caller-supplied CRM/behavioral fields to
       build a complete LeadInput, preserving the strict Pydantic contract.
    5. Pass the resulting single-row DataFrame to _score_dataframe() and return
       the ScoreOutput — identical shape to the /score endpoint.

    Raises 502 if the ARES registry is unreachable and the IČO is not cached.
    """
    # Step 1 & 2: resolve firmographics, using the cache where possible.
    ares_data = _fetch_ares(payload.ico)

    # Step 3: convert raw ARES fields to LeadInput-compatible firmographic dict.
    firmographics = _ares_to_firmographics(ares_data)

    # Step 4: build a complete LeadInput by merging firmographics (from ARES)
    # with CRM/behavioural fields (from the caller).  We exclude 'ico' from the
    # payload dump because it is not a LeadInput field.
    behavioral_and_crm = payload.model_dump(exclude={"ico"})
    full_lead = LeadInput(**{**firmographics, **behavioral_and_crm})

    # Step 5: score through the standard pipeline.
    df = pd.DataFrame([full_lead.model_dump()])
    return _score_dataframe(df)[0]
