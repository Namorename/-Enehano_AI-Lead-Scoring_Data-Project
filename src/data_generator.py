"""
Enehano Solutions - B2B Lead Scoring Dataset Generator
=======================================================
Reads the CZSO RES open-data CSV (res_open_data_sample.csv) and produces a
Salesforce-ready lead_train.csv for two-stage funnel model training.

Causal model architecture - Stage 1 (Lead -> Opportunity):
  Conversion probability is a sigmoid over an additive logit composed of:
    -> Industry attractiveness (CZ-NACE section)     - base_conv per section
    -> Company size tier                              - size_logit lookup
    -> Czech region premium (Praha / Jihomoravský)    - REGIONS table
    -> Lead-source quality                            - LEAD_SOURCES quality score
    -> Pipeline status                                - STATUS_LOGIT lookup
    -> Behavioral engagement signals                  - web visits, email, demos...
    -> Company age & legal form                       - maturity / entity type
    -> Gaussian noise                                 - unobserved confounders

Causal model architecture - Stage 2 (Opportunity -> Closed Deal):
  Win probability is an additive linear model applied ONLY to converted leads:
    -> Base win rate                                  - WIN_PROBABILITY_BASE (0.15)
    -> Meetings held                                  - +0.05 per meeting
    -> Proposal sent                                  - +0.20 if True
    -> Lead rating                                    - +0.10 Hot / −0.10 Cold
    -> Time to first response                         - +0.08 if < 2 h
    -> Rule-based score                               - (score / 100) x 0.10
    -> Probability clipped to [0.01, 0.95] before Bernoulli draw

Usage:
    python data_generator.py
    python data_generator.py --res path/to/res.csv --n 10000 --out ./output
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from paths import LEAD_TRAIN, RES_SAMPLE

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

SEED        = 42
N_LEADS     = 30_000
OUTPUT_FILE = str(LEAD_TRAIN)

# Base win probability for Stage-2 causal model (Opportunity -> Closed Deal).
# All feature contributions are added on top of this intercept before clipping.
WIN_PROBABILITY_BASE: float = 0.15

np.random.seed(SEED)
random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# 1. WEIGHTS DICTIONARIES  (edit here to tune causal model)
# ──────────────────────────────────────────────────────────────────────────────

# CZ-NACE 2025 section -> Salesforce Industry + base conversion probability.
# base_conv reflects how attractive each sector is for Salesforce consulting.
NACE_SECTIONS: dict[str, dict] = {
    "A": {"label": "Agriculture",                  "sf_industry": "Agriculture",    "base_conv": 0.04},
    "B": {"label": "Mining",                       "sf_industry": "Mining",         "base_conv": 0.05},
    "C": {"label": "Manufacturing",                "sf_industry": "Manufacturing",  "base_conv": 0.12},
    "D": {"label": "Electricity & Gas",            "sf_industry": "Energy",         "base_conv": 0.10},
    "E": {"label": "Water & Waste",                "sf_industry": "Utilities",      "base_conv": 0.07},
    "F": {"label": "Construction",                 "sf_industry": "Construction",   "base_conv": 0.09},
    "G": {"label": "Wholesale & Retail",           "sf_industry": "Retail",         "base_conv": 0.14},
    "H": {"label": "Transportation",               "sf_industry": "Transportation", "base_conv": 0.10},
    "I": {"label": "Accommodation & Food",         "sf_industry": "Hospitality",    "base_conv": 0.06},
    "J": {"label": "Information & Communication",  "sf_industry": "Technology",     "base_conv": 0.28},
    "K": {"label": "Finance & Insurance",          "sf_industry": "Finance",        "base_conv": 0.22},
    "L": {"label": "Real Estate",                  "sf_industry": "Real Estate",    "base_conv": 0.11},
    "M": {"label": "Professional & Scientific",    "sf_industry": "Consulting",     "base_conv": 0.20},
    "N": {"label": "Administrative & Support",     "sf_industry": "Other",          "base_conv": 0.13},
    "P": {"label": "Education",                    "sf_industry": "Education",      "base_conv": 0.08},
    "Q": {"label": "Healthcare",                   "sf_industry": "Healthcare",     "base_conv": 0.09},
    "R": {"label": "Arts & Entertainment",         "sf_industry": "Entertainment",  "base_conv": 0.06},
    "S": {"label": "Other Services",               "sf_industry": "Other",          "base_conv": 0.07},
}

# CZ-NACE 2025 sections not present in older Rev.2 -> remap to nearest.
NACE_FALLBACK: dict[str, str] = {"T": "S", "U": "S", "V": "S"}

# NUTS-3 regions: name, sampling weight, Praha premium flag.
REGIONS: dict[str, dict] = {
    "CZ010": {"name": "Praha",             "weight": 0.28, "premium": True},
    "CZ020": {"name": "Středočeský",       "weight": 0.12, "premium": False},
    "CZ031": {"name": "Jihočeský",         "weight": 0.06, "premium": False},
    "CZ032": {"name": "Plzeňský",          "weight": 0.05, "premium": False},
    "CZ041": {"name": "Karlovarský",       "weight": 0.02, "premium": False},
    "CZ042": {"name": "Ústecký",           "weight": 0.05, "premium": False},
    "CZ051": {"name": "Liberecký",         "weight": 0.04, "premium": False},
    "CZ052": {"name": "Královéhradecký",   "weight": 0.05, "premium": False},
    "CZ053": {"name": "Pardubický",        "weight": 0.05, "premium": False},
    "CZ063": {"name": "Kraj Vysočina",     "weight": 0.04, "premium": False},
    "CZ064": {"name": "Jihomoravský",      "weight": 0.12, "premium": False},
    "CZ071": {"name": "Olomoucký",         "weight": 0.05, "premium": False},
    "CZ072": {"name": "Zlínský",           "weight": 0.04, "premium": False},
    "CZ080": {"name": "Moravskoslezský",   "weight": 0.08, "premium": False},
}

# RES size-category codes -> EU SME band label + employee range.
# Keys match the raw integer codes in "Velikostní kategorie dle počtu zaměstnanců (kód)".
RES_SIZE_MAP: dict[str, tuple[str, int, int]] = {
    # code  -> (band,        emp_lo, emp_hi)
    "110":  ("Micro",          1,     4),
    "120":  ("Micro",          5,     9),
    "210":  ("Small",         10,    19),
    "220":  ("Small",         20,    49),
    "230":  ("Medium",        50,    99),
    "240":  ("Medium",       100,   249),
    "310":  ("Large",        250,   499),
    "320":  ("Large",        500,   999),
    "330":  ("Enterprise",  1000,  2499),
    "340":  ("Enterprise",  2500,  4999),
    "350":  ("Enterprise",  5000,  9999),
}

# Size band -> additive logit contribution (larger = more likely to convert).
SIZE_LOGIT: dict[str, float] = {
    "Micro": -0.8, "Small": 0.0, "Medium": 0.5, "Large": 1.0, "Enterprise": 1.4,
}

# conv_mult used only in rule-based score (not in causal logit).
SIZE_CONV_MULT: dict[str, float] = {
    "Micro": 0.7, "Small": 1.0, "Medium": 1.3, "Large": 1.6, "Enterprise": 1.8,
}

# Legal-form name (from RES CSV) -> internal code.
LEGAL_FORM_MAP: dict[str, str] = {
    "Společnost s ručením omezeným":                                    "112",
    "Společnost s ručením omezeným v likvidaci":                        "112",
    "Akciová společnost":                                               "121",
    "Akciová společnost v likvidaci":                                   "121",
    "Evropská akciová společnost":                                      "121",
    "Veřejná obchodní společnost":                                      "105",
    "Komanditní společnost":                                            "107",
    "Fyzická osoba podnikající dle živnostenského zákona nezapsaná v OR": "101",
    "Fyzická osoba podnikající dle živnostenského zákona zapsaná v OR":   "101",
    "Fyzická osoba podnikající dle zvláštních předpisů":                  "101",
    "Samostatný zemědělec":                                             "101",
    "Fyzická osoba":                                                    "101",
    "Zapsaný ústav":                                                    "325",
    "Nadace":                                                           "325",
    "Nadační fond":                                                     "325",
    "Obec":                                                             "751",
    "Město":                                                            "751",
    "Statutární město":                                                 "751",
    "Kraj":                                                             "751",
    "Příspěvková organizace":                                           "751",
    "Organizační složka státu":                                         "751",
    "Státní podnik":                                                    "331",
    "Pobočný spolek":                                                   "811",
    "Spolek":                                                           "811",
    "Družstvo":                                                         "811",
    "Bytové družstvo":                                                  "811",
}

LEGAL_FORM_LABELS: dict[str, str] = {
    "112": "s.r.o.", "121": "a.s.", "105": "v.o.s.", "107": "k.s.",
    "101": "OSVČ",   "325": "z.ú.", "751": "Obec",   "331": "s.p.", "811": "Pobočka",
}

# Salesforce LeadSource: weight (sampling) + quality signal (0=cold, 1=hot).
LEAD_SOURCES: dict[str, dict] = {
    "Web":            {"weight": 0.30, "quality": 0.60},
    "Referral":       {"weight": 0.15, "quality": 0.90},
    "Event":          {"weight": 0.12, "quality": 0.80},
    "Cold Call":      {"weight": 0.10, "quality": 0.30},
    "Email Campaign": {"weight": 0.15, "quality": 0.50},
    "LinkedIn":       {"weight": 0.10, "quality": 0.65},
    "Partner":        {"weight": 0.05, "quality": 0.85},
    "Other":          {"weight": 0.03, "quality": 0.40},
}

# Salesforce CRM lifecycle statuses with sampling weights.
STATUS_WEIGHTS: dict[str, float] = {
    "New":         0.12,
    "Working":     0.20,
    "Contacted":   0.25,
    "Qualified":   0.15,
    "Nurturing":   0.15,
    "Unqualified": 0.08,
    "Converted":   0.05,
}

# Pipeline progress score per status (used for latent quality signal).
STATUS_PROGRESS: dict[str, float] = {
    "New": 0.1, "Working": 0.3, "Contacted": 0.4,
    "Qualified": 0.7, "Nurturing": 0.5,
    "Unqualified": 0.0, "Converted": 1.0,
}

# Status -> additive logit contribution.
STATUS_LOGIT: dict[str, float] = {
    "New": -1.5, "Working": -0.5, "Contacted": 0.0,
    "Qualified": 1.8, "Nurturing": 0.2,
    "Unqualified": -3.0, "Converted": 4.0,
}

SF_OWNERS = [
    "Martin Novák", "Jana Procházková", "Tomáš Dvořák",
    "Lucie Horáková", "Pavel Kratochvíl", "Eva Šimánek",
]

# Revenue per employee by NACE section (CZK thousands/year, CZSO 2023 estimates).
REV_PER_EMPLOYEE: dict[str, float] = {
    "J": 2800, "K": 3500, "M": 1800, "G": 4200,
    "C": 2200, "F": 1600, "H": 1900, "L": 1200,
    "D": 5000, "default": 1500,
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def sample_weighted(mapping: dict) -> str:
    """Pick one key from a dict that contains a 'weight' sub-key."""
    keys    = list(mapping.keys())
    weights = [mapping[k]["weight"] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

# ──────────────────────────────────────────────────────────────────────────────
# 3. RES CSV LOADER & ROW PARSER
# ──────────────────────────────────────────────────────────────────────────────

# Exact column names from res_open_data_sample.csv
RES_COL_ICO         = "IČO"
RES_COL_NAME        = "Obchodní jméno/název"
RES_COL_LEGAL_FORM  = "Statistická právní forma (název)"
RES_COL_SIZE        = "Velikostní kategorie dle počtu zaměstnanců (kód)"
RES_COL_REGION      = "Kraj (název)"
RES_COL_NACE        = "Hlavní ekonomická činnost (CZ NACE2025) (kód)"
RES_COL_FOUNDED     = "Datum vzniku"


def load_res(filepath: str) -> pd.DataFrame:
    """
    Read the RES CSV robustly (tries comma/semicolon delimiter, UTF-8/BOM).
    Returns only the columns needed for firmographic mapping.
    """
    needed = [RES_COL_ICO, RES_COL_NAME, RES_COL_LEGAL_FORM,
              RES_COL_SIZE, RES_COL_REGION, RES_COL_NACE, RES_COL_FOUNDED]

    for enc in ("utf-8-sig", "utf-8", "cp1250"):
        for sep in (",", ";"):
            try:
                df = pd.read_csv(filepath, encoding=enc, sep=sep,
                                 dtype=str, low_memory=False)
                if len(df.columns) >= 4:
                    present = [c for c in needed if c in df.columns]
                    df = df[present].dropna(subset=[RES_COL_ICO])
                    print(f"  [RES] Loaded {len(df):,} rows "
                          f"(enc={enc}, sep={repr(sep)})")
                    return df
            except Exception:
                continue

    raise FileNotFoundError(
        f"Cannot parse '{filepath}'. Expected RES export with headers: "
        + ", ".join(needed)
    )


def parse_nace_section(raw: str) -> str:
    """
    Derive the CZ-NACE section letter from the raw code cell.
    Handles letter-prefixed codes ("J62.01"), numeric codes ("62010"),
    and single-letter codes ("J").
    """
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return "J"   # default: ICT

    if s[0].isalpha():
        sec = s[0].upper()
    else:
        # Numeric-only: derive section from first 2-digit division
        digits = "".join(c for c in s if c.isdigit())
        if len(digits) < 2:
            return "J"
        div = int(digits[:2])
        if   div <= 3:   sec = "A"
        elif div <= 9:   sec = "B"
        elif div <= 33:  sec = "C"
        elif div == 35:  sec = "D"
        elif div <= 39:  sec = "E"
        elif div <= 43:  sec = "F"
        elif div <= 47:  sec = "G"
        elif div <= 53:  sec = "H"
        elif div <= 56:  sec = "I"
        elif div <= 63:  sec = "J"
        elif div <= 66:  sec = "K"
        elif div == 68:  sec = "L"
        elif div <= 75:  sec = "M"
        elif div <= 82:  sec = "N"
        elif div == 84:  sec = "O"
        elif div == 85:  sec = "P"
        elif div <= 88:  sec = "Q"
        elif div <= 93:  sec = "R"
        elif div <= 96:  sec = "S"
        else:            sec = "S"

    # Apply 2025 -> Rev.2 fallback, then ensure it's in our table
    sec = NACE_FALLBACK.get(sec, sec)
    return sec if sec in NACE_SECTIONS else "S"


def parse_res_row(row: pd.Series, current_year: int) -> dict:
    """
    Extract and normalise all firmographic fields from one RES CSV row.
    Returns a clean dict consumed by generate_lead().
    """
    # IČO - strip float artefact (.0), zero-pad to 8 digits
    ico = str(row.get(RES_COL_ICO, "00000000")).split(".")[0].strip().zfill(8)

    # Company name
    name = str(row.get(RES_COL_NAME, "")).strip()
    company_name = name if name and name.lower() not in ("nan", "none", "") \
                   else f"Subjekt {ico}"

    # Legal form
    lf_raw          = str(row.get(RES_COL_LEGAL_FORM, "")).strip()
    legal_form_code = LEGAL_FORM_MAP.get(lf_raw, "112")   # default s.r.o.
    legal_form_label = LEGAL_FORM_LABELS.get(legal_form_code, "s.r.o.")

    # NACE section
    nace_section = parse_nace_section(str(row.get(RES_COL_NACE, "")))

    # Employee count & size band - from RES size-category code
    size_raw = str(row.get(RES_COL_SIZE, "")).split(".")[0].strip()
    if size_raw in RES_SIZE_MAP:
        band, emp_lo, emp_hi = RES_SIZE_MAP[size_raw]
        employee_count = random.randint(emp_lo, emp_hi)
    else:
        band, emp_lo, emp_hi = "Small", 10, 49
        employee_count = random.randint(emp_lo, emp_hi)

    # Revenue estimate (CZK millions)
    avg_emp = (emp_lo + emp_hi) / 2
    rpe     = REV_PER_EMPLOYEE.get(nace_section, REV_PER_EMPLOYEE["default"])
    annual_revenue = round(avg_emp * rpe * np.random.uniform(0.6, 1.4) / 1000, 2)

    # Region - use real region name from RES if available, else sample
    region_raw = str(row.get(RES_COL_REGION, "")).strip()
    # Match RES region name to our REGIONS table (strip "kraj" suffix variations)
    region_code = None
    for code, info in REGIONS.items():
        if info["name"].lower() in region_raw.lower() or \
           region_raw.lower() in info["name"].lower():
            region_code = code
            break
    if region_code is None:
        region_keys    = list(REGIONS.keys())
        region_weights = [REGIONS[k]["weight"] for k in region_keys]
        region_code    = random.choices(region_keys, weights=region_weights, k=1)[0]

    region_name = REGIONS[region_code]["name"]

    # Founding year & company age
    origin_raw = str(row.get(RES_COL_FOUNDED, "")).strip()
    try:
        found_year = int(origin_raw[:4])
        if not (1890 <= found_year <= current_year):
            raise ValueError
    except (ValueError, IndexError):
        found_year = random.randint(2000, 2020)

    company_age = max(0, current_year - found_year)

    return {
        "ico":             ico,
        "company_name":    company_name,
        "legal_form_code": legal_form_code,
        "legal_form_label": legal_form_label,
        "nace_section":    nace_section,
        "size_band":       band,
        "employee_count":  employee_count,
        "annual_revenue":  annual_revenue,
        "region_code":     region_code,
        "region_name":     region_name,
        "found_year":      found_year,
        "company_age":     company_age,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 4. BEHAVIORAL FIELD GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_behavioral_fields(lead_source: str, status: str, quality: float) -> dict:
    """
    Produce correlated CRM engagement fields driven by a latent quality signal.

    quality [0..1]:
      0 = cold, disinterested lead
      1 = hot, highly engaged lead

    All distributions are calibrated so that the causal model (Section 5) can
    extract meaningful non-linear patterns from them.
    """
    q = quality

    # Web interactions: negative-binomial -> heavy tail for engaged leads
    web_interactions = int(clamp(
        np.random.negative_binomial(n=max(1, int(q * 5)), p=0.3),
        0, 200
    ))

    # Email engagement
    emails_sent     = max(1, int(np.random.poisson(5 + q * 10)))
    email_open_rate = clamp(np.random.beta(a=1 + q * 3, b=1 + (1 - q) * 4))
    email_opens     = int(emails_sent * email_open_rate)
    email_clicks    = int(email_opens * clamp(np.random.beta(a=1 + q * 2, b=3)))

    # Form submissions
    form_submissions = int(np.random.poisson(q * 2.5))

    # Demo requested: binary signal correlated with quality
    demo_requested = int(np.random.random() < (0.05 + q * 0.55))

    # Time to first response: faster for high-quality sources
    src_speed = LEAD_SOURCES.get(lead_source, {}).get("quality", 0.5)
    ttfr = round(clamp(
        np.random.exponential(scale=max(1, 72 * (1 - src_speed * 0.6))),
        0.25, 336
    ), 1)

    # Calls and meetings
    calls_made   = max(0, int(np.random.poisson(0.5 + q * 4)))
    meetings_held = max(0, int(np.random.poisson(q * 1.5)))

    # Proposal sent only for advanced pipeline stages
    proposal_sent = int(
        status in ("Qualified", "Converted") and np.random.random() < 0.75
    )

    # Content downloads and chat
    content_downloads   = int(np.random.poisson(q * 3))
    chatbot_interactions = int(np.random.poisson(q * 1.5))

    # LinkedIn viewed (sales-navigator signal)
    linkedin_viewed = int(np.random.random() < 0.3 + q * 0.4)

    # Days since last activity
    decay_scale = {
        "Converted": 3, "Qualified": 3, "Nurturing": 30,
        "Unqualified": 60
    }.get(status, 14)
    days_since_last_activity = int(np.random.exponential(scale=decay_scale))

    return {
        "Web_Interactions__c":         web_interactions,
        "Email_Opens__c":              email_opens,
        "Email_Clicks__c":             email_clicks,
        "Emails_Sent__c":              emails_sent,
        "Email_Open_Rate__c":          round(email_open_rate, 4),
        "Form_Submissions__c":         form_submissions,
        "Demo_Requested__c":           bool(demo_requested),
        "Time_to_First_Response_h__c": ttfr,
        "Calls_Made__c":               calls_made,
        "Meetings_Held__c":            meetings_held,
        "Proposal_Sent__c":            bool(proposal_sent),
        "Content_Downloads__c":        content_downloads,
        "Chatbot_Interactions__c":     chatbot_interactions,
        "LinkedIn_Viewed__c":          bool(linkedin_viewed),
        "Days_Since_Last_Activity__c": days_since_last_activity,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 5. CAUSAL CONVERSION MODEL  (logit -> sigmoid)
# ──────────────────────────────────────────────────────────────────────────────

def compute_conversion_probability(
    nace_section:    str,
    size_band:       str,
    region_code:     str,
    lead_source:     str,
    status:          str,
    behavioral:      dict,
    legal_form_code: str,
    company_age:     int,
) -> float:
    """
    Build the conversion logit from additive weighted contributions and pass
    through a sigmoid to produce a probability in (0, 1).

    Each block is clearly separated so weights can be tuned independently.
    """
    logit = 0.0

    # ── Industry base attractiveness ──────────────────────────────────────
    base = NACE_SECTIONS[nace_section]["base_conv"]
    logit += np.log(base / (1 - base))           # base_conv -> logit space

    # ── Company size premium ──────────────────────────────────────────────
    logit += SIZE_LOGIT.get(size_band, 0.0)

    # ── Region premium ────────────────────────────────────────────────────
    if REGIONS[region_code]["premium"]:           # Praha
        logit += 0.4
    elif REGIONS[region_code]["name"] == "Jihomoravský":
        logit += 0.2

    # ── Lead source quality ───────────────────────────────────────────────
    src_quality = LEAD_SOURCES.get(lead_source, {}).get("quality", 0.4)
    logit += (src_quality - 0.5) * 2.0           # centred around 0

    # ── Pipeline status ───────────────────────────────────────────────────
    logit += STATUS_LOGIT.get(status, 0.0)

    # ── Behavioral engagement signals ─────────────────────────────────────
    # Web interactions - log-scaled so diminishing returns apply
    logit += 0.4 * np.log1p(behavioral["Web_Interactions__c"])

    # Email open rate - S-curve centred at 0.3 (industry average)
    logit += 1.2 * (behavioral["Email_Open_Rate__c"] - 0.3)

    # Demo request - strong buying-intent signal
    if behavioral["Demo_Requested__c"]:
        logit += 1.8

    # Proposal sent - late-stage positive signal
    if behavioral["Proposal_Sent__c"]:
        logit += 1.5

    # Meetings held - capped at 5 to limit leverage
    logit += 0.4 * min(behavioral["Meetings_Held__c"], 5)

    # Time to first response - penalty for slow SDR reaction
    logit -= 0.3 * np.log1p(behavioral["Time_to_First_Response_h__c"] / 4)

    # Days since last activity - recency penalty, capped at 90 days
    logit -= 0.02 * min(behavioral["Days_Since_Last_Activity__c"], 90)

    # ── Company maturity ──────────────────────────────────────────────────
    # Established firms (up to 20 yrs) slightly more likely to invest in CRM
    logit += 0.1 * min(company_age, 20) / 20.0

    # ── Legal form ────────────────────────────────────────────────────────
    if legal_form_code in ("112", "121"):   # s.r.o. / a.s. - corporate buyers
        logit += 0.3
    elif legal_form_code == "101":          # OSVČ - sole trader, lower budget
        logit -= 0.4

    # ── Gaussian noise (unobserved confounders) ───────────────────────────
    logit += np.random.normal(0, 0.8)

    return clamp(sigmoid(logit))

# ──────────────────────────────────────────────────────────────────────────────
# 5b. CAUSAL WIN MODEL  (Stage 2: Opportunity -> Closed Deal)
# ──────────────────────────────────────────────────────────────────────────────

def compute_win_probability(
    meetings_held:          int   | float,
    proposal_sent:          bool,
    rating:                 str,
    time_to_first_response: float,
    rule_based_score:       int   | float,
) -> float:
    """
    Compute the Closed_Won probability for a single *converted* lead using an
    additive linear model grounded in B2B sales research.

    Parameters
    ----------
    meetings_held          : number of meetings held with the prospect
    proposal_sent          : whether a formal proposal was delivered
    rating                 : Salesforce lead rating - "Hot", "Warm", or "Cold"
    time_to_first_response : hours from lead creation to first SDR response
    rule_based_score       : deterministic 0-100 score from compute_rule_based_score()

    Returns
    -------
    float
        Win probability clipped to [0.01, 0.95].

    Notes
    -----
    This function intentionally avoids per-call noise so that the signal-to-noise
    ratio in the training data is high enough for a classifier to learn from.
    Stochasticity is introduced exclusively at the Bernoulli draw stage in
    build_dataset(), where np.random.binomial samples the final 0/1 outcome.
    """
    win_prob = WIN_PROBABILITY_BASE

    # Meetings held: each meeting adds a fixed probability increment.
    # No cap is applied here - the clip at the end bounds the total.
    win_prob += meetings_held * 0.05

    # Proposal sent: strongest single signal of late-stage intent.
    if proposal_sent:
        win_prob += 0.20

    # Lead rating: hot prospects are more likely to close; cold ones less so.
    if rating == "Hot":
        win_prob += 0.10
    elif rating == "Cold":
        win_prob -= 0.10

    # Time to first response: rapid follow-up correlates strongly with win rate
    # (Harvard Business Review "Lead Response Management" research).
    if time_to_first_response < 2.0:
        win_prob += 0.08

    # Rule-based score: blended firmographic + behavioural quality signal,
    # normalised to contribute at most 0.10 at a perfect score of 100.
    win_prob += (rule_based_score / 100.0) * 0.10

    # Clip to a valid probability range before the Bernoulli draw.
    return float(np.clip(win_prob, 0.01, 0.95))

# ──────────────────────────────────────────────────────────────────────────────
# 6. RULE-BASED SCORE  (baseline / explainability layer)
# ──────────────────────────────────────────────────────────────────────────────

def compute_rule_based_score(row: dict) -> int:
    """
    Deterministic 0-100 lead score inspired by Salesforce Einstein/Pardot rubrics.
    Used as a baseline to benchmark against the Random Forest model.
    """
    score = 0

    # Firmographic signals (max ~38 pts)
    score += {"Micro": 3, "Small": 8, "Medium": 15, "Large": 22, "Enterprise": 30}.get(
        row["Size_Band"], 0
    )
    high_fit = {"Technology", "Finance", "Consulting"}
    med_fit  = {"Manufacturing", "Retail", "Healthcare", "Real Estate", "Transportation"}
    if row["Industry"] in high_fit:   score += 20
    elif row["Industry"] in med_fit:  score += 10
    else:                             score += 3

    if row["Region_Name"] == "Praha":         score += 10
    elif row["Region_Name"] == "Jihomoravský": score += 5

    if row["Legal_Form_Code"] in ("112", "121"): score += 8
    elif row["Legal_Form_Code"] in ("325", "811"): score += 4

    # Behavioral signals (max ~53 pts)
    wi = row["Web_Interactions__c"]
    if wi >= 20:   score += 15
    elif wi >= 10: score += 10
    elif wi >= 5:  score += 5
    elif wi >= 1:  score += 2

    eor = row["Email_Open_Rate__c"]
    if eor >= 0.5:  score += 8
    elif eor >= 0.2: score += 4

    if row["Demo_Requested__c"]:    score += 12
    if row["Proposal_Sent__c"]:     score += 10
    if row["Meetings_Held__c"] >= 2: score += 8
    elif row["Meetings_Held__c"] == 1: score += 4

    ttr = row["Time_to_First_Response_h__c"]
    if ttr <= 1:    score += 15
    elif ttr <= 4:  score += 12
    elif ttr <= 24: score += 8
    elif ttr <= 72: score += 4

    score += {
        "Referral": 15, "Partner": 13, "Event": 10, "LinkedIn": 8,
        "Web": 6, "Email Campaign": 5, "Cold Call": 2, "Other": 1
    }.get(row["LeadSource"], 0)

    # Status modifier
    if row["Status"] == "Unqualified": score = max(0, score - 40)
    if row["Status"] == "Qualified":   score = min(100, score + 15)

    return min(100, max(0, score))

# ──────────────────────────────────────────────────────────────────────────────
# 7. SINGLE LEAD GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_lead(lead_id: int, firm: dict) -> dict:
    """
    Produce one complete Salesforce-ready lead record.

    Parameters
    ----------
    lead_id : sequential integer -> Salesforce-style Lead_ID
    firm    : dict from parse_res_row() - all real firmographic fields
    """
    # ── Salesforce standard fields ────────────────────────────────────────
    lead_source = sample_weighted(LEAD_SOURCES)
    src_quality = LEAD_SOURCES[lead_source]["quality"]

    status = random.choices(
        list(STATUS_WEIGHTS.keys()),
        weights=list(STATUS_WEIGHTS.values()),
        k=1
    )[0]

    # Rating correlated with status
    if status in ("Qualified", "Converted"):
        rating = random.choices(["Hot", "Warm", "Cold"], weights=[0.65, 0.30, 0.05])[0]
    elif status in ("Working", "Contacted"):
        rating = random.choices(["Hot", "Warm", "Cold"], weights=[0.20, 0.55, 0.25])[0]
    else:
        rating = random.choices(["Hot", "Warm", "Cold"], weights=[0.05, 0.25, 0.70])[0]

    # Latent quality signal (drives behavioral field distributions)
    quality = clamp(
        src_quality * 0.4
        + STATUS_PROGRESS[status] * 0.4
        + np.random.normal(0, 0.15)
    )

    # ── Behavioral fields ─────────────────────────────────────────────────
    behavioral = generate_behavioral_fields(lead_source, status, quality)

    # ── Temporal fields ───────────────────────────────────────────────────
    END   = datetime(2025, 3, 31)
    START = datetime(2022, 1, 1)
    created_date = random_date(START, END)
    stage_days_range = {
        "New": (0, 5), "Working": (3, 30), "Contacted": (7, 60),
        "Qualified": (14, 90), "Nurturing": (30, 180),
        "Unqualified": (1, 45), "Converted": (21, 180),
    }.get(status, (7, 90))
    days_in_pipeline = random.randint(*stage_days_range)
    last_activity    = min(created_date + timedelta(days=days_in_pipeline), END)

    # ── Conversion probability -> binary label ─────────────────────────────
    conv_prob = compute_conversion_probability(
        nace_section    = firm["nace_section"],
        size_band       = firm["size_band"],
        region_code     = firm["region_code"],
        lead_source     = lead_source,
        status          = status,
        behavioral      = behavioral,
        legal_form_code = firm["legal_form_code"],
        company_age     = firm["company_age"],
    )
    converted = int(np.random.random() < conv_prob)

    # Business rule: Converted=1 only makes sense for plausible statuses
    if converted == 1 and status not in ("Qualified", "Converted", "Contacted"):
        converted = 0

    # ── Rule-based score ──────────────────────────────────────────────────
    score_input = {
        "Size_Band":               firm["size_band"],
        "Industry":                NACE_SECTIONS[firm["nace_section"]]["sf_industry"],
        "Region_Name":             firm["region_name"],
        "Legal_Form_Code":         firm["legal_form_code"],
        "LeadSource":              lead_source,
        "Status":                  status,
        **behavioral,
    }
    rule_score = compute_rule_based_score(score_input)

    # ── Assemble record ───────────────────────────────────────────────────
    return {
        # IDs
        "Lead_ID":                     f"00Q{lead_id:07d}",
        "IČO":                         firm["ico"],
        "Company_Name":                firm["company_name"],

        # Firmographics (RES-derived)
        "CZ_NACE_Section":             firm["nace_section"],
        "CZ_NACE_Section_Label":       NACE_SECTIONS[firm["nace_section"]]["label"],
        "Industry":                    NACE_SECTIONS[firm["nace_section"]]["sf_industry"],
        "Region_Code":                 firm["region_code"],
        "Region_Name":                 firm["region_name"],
        "Legal_Form_Code":             firm["legal_form_code"],
        "Legal_Form_Label":            firm["legal_form_label"],
        "Founding_Year":               firm["found_year"],
        "Company_Age_Years":           firm["company_age"],
        "Size_Band":                   firm["size_band"],
        "NumberOfEmployees":           firm["employee_count"],
        "Annual_Revenue_MCZK__c":      firm["annual_revenue"],

        # Salesforce standard
        "LeadSource":                  lead_source,
        "Status":                      status,
        "Rating":                      rating,
        "Owner":                       random.choice(SF_OWNERS),

        # Behavioral engagement (custom fields)
        **behavioral,

        # Temporal
        "CreatedDate":                 created_date.strftime("%Y-%m-%d"),
        "LastActivityDate":            last_activity.strftime("%Y-%m-%d"),
        "Days_in_Pipeline__c":         days_in_pipeline,

        # Derived scoring
        "Rule_Based_Score":            rule_score,
        "Rule_Based_Segment":          (
            "High"   if rule_score >= 60 else
            "Medium" if rule_score >= 30 else
            "Low"
        ),
        "Conversion_Probability":      round(conv_prob, 6),

        # Win probability for Stage-2 model - computed for every row so the
        # column is always present; Closed_Won is generated in build_dataset()
        # via a single vectorised np.random.binomial call over converted rows.
        "Win_Probability__c":          round(
            compute_win_probability(
                meetings_held          = behavioral["Meetings_Held__c"],
                proposal_sent          = behavioral["Proposal_Sent__c"],
                rating                 = rating,
                time_to_first_response = behavioral["Time_to_First_Response_h__c"],
                rule_based_score       = rule_score,
            ),
            6,
        ),

        # TARGET - stage 1: Lead -> Opportunity
        "Converted":                   converted,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 8. DATASET ASSEMBLY
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(res_filepath: str, n: int = N_LEADS) -> pd.DataFrame:
    """
    Load the RES CSV, sample n rows, and generate one lead per company.
    Returns a DataFrame ready for export.
    """
    print(f"\n{'='*60}")
    print(" Enehano Solutions - Lead Dataset Generator")
    print(f"{'='*60}\n")

    df_res      = load_res(res_filepath)
    sample_size = min(n, len(df_res))
    df_sample   = df_res.sample(sample_size, random_state=SEED).reset_index(drop=True)
    print(f"  Generating {sample_size:,} leads from RES export ...\n")

    current_year = datetime.now().year
    records = []

    for i, (_, row) in enumerate(df_sample.iterrows()):
        firm   = parse_res_row(row, current_year)
        record = generate_lead(lead_id=i + 1, firm=firm)
        records.append(record)

    df = pd.DataFrame(records)

    # ── Stage-2 target: Closed_Won  ───────────────────────────────────────────
    # Business rule: a deal can only be won if the lead was first converted.
    # For non-converted rows the win probability is irrelevant - Closed_Won is
    # forced to 0.  For converted rows we draw from a Bernoulli distribution
    # whose parameter p comes from the per-row Win_Probability__c computed
    # inside generate_lead() via compute_win_probability().
    #
    # Using a vectorised np.random.binomial call (n=1 per element) over the
    # full array is both faster than a Python loop and semantically correct -
    # each row gets an independent Bernoulli draw with its own probability.
    win_probs = df["Win_Probability__c"].to_numpy()
    raw_draws = np.random.binomial(n=1, p=win_probs)           # shape (N,)
    df["Closed_Won"] = np.where(df["Converted"] == 1, raw_draws, 0).astype(int)

    # Verify the hard dependency invariant before continuing.
    assert (df.loc[df["Converted"] == 0, "Closed_Won"] == 0).all(), (
        "Invariant violated: Closed_Won=1 found where Converted=0"
    )

    # Flag duplicate IČOs originating from the RES source file
    dupes = df["IČO"].duplicated(keep="first")
    df["IČO_Duplicate_Flag"] = dupes.astype(int)
    if dupes.any():
        print(f"  [INFO] {dupes.sum():,} duplicate IČOs detected (flagged, not dropped).")

    return df


def print_quality_report(df: pd.DataFrame) -> None:
    """Print a concise quality summary to validate the generated dataset."""
    print(f"\n{'='*60}")
    print(" DATASET QUALITY REPORT")
    print(f"{'='*60}")
    total = len(df)
    conv  = df["Converted"].sum()
    won   = df["Closed_Won"].sum()

    print(f"  Rows:             {total:,}")
    print(f"  Columns:          {len(df.columns)}")
    print(f"  Conversion rate:  {conv/total:.1%}  ({conv:,} of {total:,})")
    print(f"  Win rate (overall):       {won/total:.1%}  ({won:,} of {total:,})")
    if conv > 0:
        print(f"  Win rate (of converted):  {won/conv:.1%}  ({won:,} of {conv:,})")

    print("\n  Conversions by Industry:")
    grp = df.groupby("Industry")["Converted"].agg(["sum", "count"])
    grp["rate"] = grp["sum"] / grp["count"]
    print(grp.sort_values("rate", ascending=False)
             .rename(columns={"sum": "Converted", "count": "Total"})
             .to_string())

    print("\n  Closed_Won rate by Rating (converted leads only):")
    conv_df = df[df["Converted"] == 1]
    if not conv_df.empty:
        grp2 = conv_df.groupby("Rating")["Closed_Won"].agg(["sum", "count"])
        grp2["win_rate"] = grp2["sum"] / grp2["count"]
        print(grp2.sort_values("win_rate", ascending=False)
                 .rename(columns={"sum": "Won", "count": "Converted"})
                 .to_string())

    print("\n  Win_Probability__c distribution (converted leads only):")
    if not conv_df.empty:
        print(conv_df["Win_Probability__c"].describe().to_string())

    print("\n  Rule_Based_Score distribution:")
    print(df["Rule_Based_Score"].describe().to_string())
    print(f"\n{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 9. ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enehano Solutions - B2B Lead Scoring Dataset Generator"
    )
    parser.add_argument(
        "--res", type=str, default=str(RES_SAMPLE),
        help=f"Path to CZSO RES open-data CSV (default: {RES_SAMPLE})"
    )
    parser.add_argument(
        "--n", type=int, default=N_LEADS,
        help=f"Number of leads to generate (default: {N_LEADS:,})"
    )
    parser.add_argument(
        "--out", type=str, default=OUTPUT_FILE,
        help=f"Output CSV path (default: {OUTPUT_FILE})"
    )
    args = parser.parse_args()

    df = build_dataset(res_filepath=args.res, n=args.n)
    print_quality_report(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved -> {out_path.resolve()}")
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")
