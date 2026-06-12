"""
app.py
======
Enehano Lead Intelligence - Streamlit dashboard.

A sales product, not a model showcase. Four screens:

  Today         - the call queue: five companies, ranked, with a reason to call
  All leads     - the full ranked pipeline, filters, Salesforce-ready export
  Lead profile  - one company in depth, with a live what-if simulator
  For managers  - revenue impact, model quality, stability (everything a rep
                  doesn't need is here, out of the way)

All model inference is delegated to the FastAPI backend (api.py).
Visual theming lives in .streamlit/config.toml (native dark theme);
the CSS below only adds brand identity on top, never fights the platform.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import business_helpers as bh
from contracts import API_PAYLOAD_EXCLUDE_COLS, add_region_alias, score_segment
from paths import (
    LEAD_TRAIN, METRICS_JSON, FEATURE_IMPORTANCE, VAL_PREDICTIONS, LOGO_SVG,
)

# ── Brand tokens ──────────────────────────────────────────────────────────────
# SCORING_API_URL controls the scoring backend:
#   unset or "embedded"        → in-process scoring via scorer.py (Streamlit Cloud)
#   http://host:port           → external FastAPI backend (local dev / production)
_SCORING_URL = os.getenv("SCORING_API_URL", "")
EMBEDDED     = not _SCORING_URL or _SCORING_URL.lower() == "embedded"
API_BASE     = _SCORING_URL if not EMBEDDED else "http://localhost:8000"

if EMBEDDED:
    import scorer as _scorer

LIME    = "#a6ce39"   # the one accent: scores, High priority, primary actions
AMBER   = "#d9a93c"   # Medium priority
SLATE   = "#5d6b7a"   # Low priority - cool and quiet, a lead is not an error
RED     = "#e25c5c"   # reserved for genuine problems (drift), never for leads
INK     = "#e8edf2"
MUTED   = "#94a1ae"
SURFACE = "#151c24"
BORDER  = "#26303b"

SEG_COLOR = {"High": LIME, "Medium": AMBER, "Low": SLATE}

st.set_page_config(
    page_title="Enehano Lead Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Identity layer (the theme does the heavy lifting in config.toml) ─────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}

    /* Brand stripe - a single lime line at the very top of the product */
    .stApp {{ border-top: 3px solid {LIME}; }}

    /* One orchestrated entrance: header and queue cards rise into place */
    @keyframes rise {{
        from {{ opacity: 0; transform: translateY(14px); }}
        to   {{ opacity: 1; transform: none; }}
    }}
    .brand-head {{ animation: rise .5s ease both; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ animation: rise .5s ease both; }}
    [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(1) {{ animation-delay: .05s; }}
    [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(2) {{ animation-delay: .13s; }}
    [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(3) {{ animation-delay: .21s; }}
    [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(4) {{ animation-delay: .29s; }}
    [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(5) {{ animation-delay: .37s; }}
    @media (prefers-reduced-motion: reduce) {{
        .brand-head, [data-testid="stVerticalBlockBorderWrapper"] {{ animation: none; }}
    }}

    /* Hero headline on Today */
    .hero-line {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 40px;
        font-weight: 700;
        letter-spacing: -1px;
        line-height: 1.12;
        color: {INK};
        margin: 2px 0 4px 0;
    }}
    .hero-line .hl {{ color: {LIME}; }}

    /* Slim stat strip under the hero */
    .stat-strip {{
        display: flex;
        gap: 34px;
        margin: 14px 0 6px 0;
    }}
    .stat-strip .s-val {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: {INK};
    }}
    .stat-strip .s-lab {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {MUTED};
    }}

    /* Live pulse dot */
    @keyframes pulse {{
        0%   {{ box-shadow: 0 0 0 0 rgba(166, 206, 57, 0.45); }}
        70%  {{ box-shadow: 0 0 0 7px rgba(166, 206, 57, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(166, 206, 57, 0); }}
    }}
    .live-dot {{
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: {LIME};
        animation: pulse 2.2s ease-out infinite;
        margin-right: 6px;
        vertical-align: 1px;
    }}
    @media (prefers-reduced-motion: reduce) {{ .live-dot {{ animation: none; }} }}

    /* Business case: verdict + head-to-head + rollout */
    .verdict {{
        border: 1px solid {LIME};
        border-radius: 14px;
        padding: 26px 30px;
        background: rgba(166, 206, 57, 0.06);
        margin: 6px 0 10px 0;
    }}
    .verdict .v-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 27px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: {INK};
        margin: 0 0 6px 0;
    }}
    .h2h {{ text-align: center; padding: 18px 8px; }}
    .h2h .n {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 40px; font-weight: 700; line-height: 1.05;
    }}
    .h2h .l {{
        font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: {MUTED}; margin-top: 4px;
    }}
    .step-tag {{
        display: inline-block;
        font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {LIME};
        border: 1px solid {LIME};
        border-radius: 999px;
        padding: 3px 12px;
        margin-bottom: 10px;
    }}

    /* Top navigation: a radio group dressed as a product nav */
    div[role="radiogroup"] {{
        gap: 6px;
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 4px;
        width: fit-content;
        margin: 0 auto;
    }}
    div[role="radiogroup"] label {{
        padding: 7px 20px;
        border-radius: 7px;
        font-weight: 500;
        cursor: pointer;
        transition: background .12s ease;
    }}
    div[role="radiogroup"] label:hover {{ background: #1c2530; }}
    div[role="radiogroup"] label > div:first-child {{ display: none; }}
    div[role="radiogroup"] label:has(input:checked) {{
        background: {LIME};
    }}
    div[role="radiogroup"] label:has(input:checked) p {{
        color: #10160a !important;
        font-weight: 700;
    }}

    /* Wordmark */
    .wordmark {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 26px;
        letter-spacing: -0.5px;
        color: {INK};
    }}
    .wordmark .dot {{ color: {LIME}; }}
    .tagline {{ color: {MUTED}; font-size: 13px; margin-top: -4px; }}

    .status {{ font-size: 12px; color: {MUTED}; text-align: right; }}
    .status .on  {{ color: {LIME}; }}
    .status .off {{ color: {RED}; }}

    /* Section eyebrows - quiet structure */
    .eyebrow {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {MUTED};
        margin: 6px 0 2px 0;
    }}

    /* Call-queue rank numeral - the signature element */
    .rank {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 44px;
        font-weight: 700;
        color: {LIME};
        line-height: 1.0;
        opacity: 0.92;
    }}
    .q-co   {{ font-size: 19px; font-weight: 700; color: {INK}; }}
    .q-meta {{ font-size: 13px; color: {MUTED}; }}
    .q-why  {{
        font-size: 14px;
        color: {INK};
        margin-top: 7px;
        padding-left: 10px;
        border-left: 2px solid {LIME};
    }}
    .q-score {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 38px;
        font-weight: 700;
        line-height: 1.0;
        text-align: right;
    }}

    .chip {{
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid;
    }}

    .big-score {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 76px;
        font-weight: 700;
        line-height: 1.0;
    }}

    .finding {{
        font-size: 15px;
        padding: 12px 16px;
        border-left: 2px solid {LIME};
        background: rgba(166, 206, 57, 0.07);
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }}

    /* Primary buttons: flat lime, no gradients, no motion tricks */
    .stButton > button[kind="primary"],
    .stDownloadButton > button {{
        background: {LIME};
        color: #10160a;
        border: none;
        font-weight: 700;
    }}
    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button:hover {{
        background: #b8dd4e;
        color: #10160a;
    }}

    /* ── Flex card layout — works at any viewport width ─────────────── */
    /* Cards use their own flex row so the content never collapses awkwardly */
    .q-card {{
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding-bottom: 4px;
    }}
    .q-card-rank {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 44px;
        font-weight: 700;
        color: {LIME};
        line-height: 1.0;
        opacity: 0.92;
        flex-shrink: 0;
        min-width: 52px;
    }}
    .q-card-body {{ flex: 1; min-width: 0; }}
    .q-card-right {{ flex-shrink: 0; text-align: right; padding-top: 4px; }}
    .q-card-left  {{ flex-shrink: 0; padding-top: 2px; min-width: 60px; }}

    /* ── Mobile responsive layout ────────────────────────────────────── */
    @media (max-width: 768px) {{

        /* Stack every st.columns() group into a single column */
        [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap;
        }}
        [data-testid="column"] {{
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }}

        /* Touch-friendly nav: wrap tabs, 44 px minimum tap target */
        div[role="radiogroup"] {{
            width: 100% !important;
            flex-wrap: wrap !important;
            justify-content: center !important;
        }}
        div[role="radiogroup"] label {{
            min-height: 44px !important;
            display: flex !important;
            align-items: center !important;
            padding: 8px 14px !important;
        }}

        /* Scale headlines down for narrow screens */
        .hero-line    {{ font-size: clamp(24px, 7vw, 40px); letter-spacing: -0.3px; }}
        .rank         {{ font-size: 34px; }}
        .q-score      {{ font-size: 26px; }}
        .q-card-rank  {{ font-size: 32px; min-width: 38px; }}
        .q-card       {{ gap: 10px; }}
        .big-score    {{ font-size: 54px; }}
        .h2h .n       {{ font-size: 28px; }}

        /* Stat strip: wrap gracefully instead of overflowing */
        .stat-strip {{ flex-wrap: wrap; gap: 18px; }}

        /* Logo and status pill centred on narrow screens */
        .brand-head, .status {{ text-align: center; }}

        /* Looser padding in bordered blocks */
        .verdict {{ padding: 18px 16px; }}

        /* Streamlit UI chrome: hide Deploy/Share to feel like a native app */
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        .stDeployButton {{ display: none !important; }}

        /* iOS safe-area: don't clip content behind the home indicator */
        .stApp {{ padding-bottom: env(safe-area-inset-bottom, 16px); }}

        /* Suppress the grey tap-flash on touch */
        * {{ -webkit-tap-highlight-color: transparent; }}
    }}

    /* ── Logo container — lets CSS control the width ──────────────────── */
    .logo-wrap {{ width: 190px; }}
    .logo-wrap svg {{ width: 100%; height: auto; display: block; }}

    /* ── Phone-specific tweaks (≤ 640px) — override some 768px rules ── */
    @media (max-width: 640px) {{

        /* Smaller logo on a phone */
        .logo-wrap {{ width: 128px; }}

        /* Nav: scroll horizontally instead of wrapping into a grid */
        div[role="radiogroup"] {{
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            justify-content: flex-start !important;
            scrollbar-width: none !important;
            -webkit-overflow-scrolling: touch;
        }}
        div[role="radiogroup"]::-webkit-scrollbar {{ display: none; }}
        div[role="radiogroup"] label {{
            white-space: nowrap !important;
            flex-shrink: 0 !important;
        }}

        /* Bump up the smallest text — 11px is unreadable on a phone */
        .eyebrow      {{ font-size: 13px; }}
        .q-meta       {{ font-size: 14px; }}
        .stat-strip .s-lab {{ font-size: 12px; }}
        .chip         {{ font-size: 12px; padding: 4px 12px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── API client ────────────────────────────────────────────────────────────────

class APIClient:
    """Thin wrapper around the FastAPI scoring backend."""

    BASE = API_BASE
    TIMEOUT = 60

    @classmethod
    def _get(cls, path: str) -> dict | None:
        try:
            r = requests.get(f"{cls.BASE}{path}", timeout=cls.TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return None

    @classmethod
    def _post(cls, path: str, payload: dict) -> dict | list | None:
        try:
            r = requests.post(f"{cls.BASE}{path}", json=payload, timeout=cls.TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return None

    @classmethod
    def health(cls) -> bool:
        result = cls._get("/health")
        return result is not None and result.get("status") == "ok"

    @classmethod
    def score_batch(cls, leads: list[dict], include_drivers: bool = False) -> list | None:
        return cls._post("/score/batch", {"leads": leads, "include_drivers": include_drivers})

    @classmethod
    def score_enrich(cls, ico: str, behavioral: dict) -> dict | None:
        return cls._post("/score/enrich", {"ico": ico, **behavioral})


_api_online = False if EMBEDDED else APIClient.health()


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(LEAD_TRAIN, dtype={"IČO": str, "Legal_Form_Code": str})
    df["IČO"] = df["IČO"].str.zfill(8)
    df["Legal_Form_Code"] = df["Legal_Form_Code"].str.strip()
    return add_region_alias(df, copy=False)


@st.cache_data
def load_metrics() -> dict | None:
    return json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else None


@st.cache_data
def load_importance() -> pd.DataFrame | None:
    return pd.read_csv(FEATURE_IMPORTANCE) if FEATURE_IMPORTANCE.exists() else None


@st.cache_data
def load_val_predictions() -> pd.DataFrame | None:
    return pd.read_csv(VAL_PREDICTIONS) if VAL_PREDICTIONS.exists() else None


def _apply_scores(df: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    out = df.copy()
    out["AI_Score"]           = [r["ai_score"]               for r in results]
    out["Segment"]            = [r["segment"]                for r in results]
    out["Expected_Win_%"]     = [r["expected_win_pct"]       for r in results]
    out["Rule_Score"]         = [r["rule_score"]             for r in results]
    out["top_drivers"]        = [r["top_drivers"]            for r in results]
    out["P_Win_If_Converted"] = [r["p_win_if_converted_pct"] for r in results]
    return out


@st.cache_data(ttl=300, show_spinner="Scoring the pipeline...")
def score_pipeline_via_api(df: pd.DataFrame) -> pd.DataFrame:
    """Score every lead — via embedded scorer or FastAPI backend."""
    api_cols = [c for c in df.columns if c not in API_PAYLOAD_EXCLUDE_COLS]

    if EMBEDDED:
        results = _scorer.score_dataframe(df[api_cols].fillna(0))
        if len(results) != len(df):
            st.error("Embedded scoring returned an unexpected result. Try refreshing the page.")
            st.stop()
        return _apply_scores(df, results)

    leads_payload = df[api_cols].fillna(0).to_dict(orient="records")
    results = []
    for start in range(0, len(leads_payload), 500):
        batch_result = APIClient.score_batch(leads_payload[start:start + 500])
        if not batch_result:
            st.error("Scoring is unavailable right now. The page will work again once the scoring service is reachable.")
            st.stop()
        results.extend(batch_result)

    if len(results) != len(df):
        st.error("Scoring returned an unexpected result. Try refreshing the page.")
        st.stop()

    return _apply_scores(df, results)


# ── Plain-language reason engine ─────────────────────────────────────────────
# Reads the lead's own behaviour and produces ONE action-phrased reason.
# Deterministic and local: works even when the scoring API can't return
# SHAP drivers, so the call queue never shows an empty "why".

def call_reason(row: pd.Series) -> str:
    days_quiet = int(row.get("Days_Since_Last_Activity__c", 0) or 0)
    meetings   = int(row.get("Meetings_Held__c", 0) or 0)
    web        = int(row.get("Web_Interactions__c", 0) or 0)
    opens      = int(row.get("Email_Opens__c", 0) or 0)
    sent       = int(row.get("Emails_Sent__c", 0) or 0)
    ttfr       = float(row.get("Time_to_First_Response_h__c", 99) or 99)

    if int(row.get("Proposal_Sent__c", 0) or 0):
        return "Proposal on the table - follow up before it goes stale"
    if int(row.get("Demo_Requested__c", 0) or 0):
        return "Asked for a demo - strike while interest is hot"
    if meetings >= 2:
        return f"{meetings} meetings in - momentum is building, keep it moving"
    if ttfr <= 2:
        return "Replied within two hours - that speed means intent"
    if sent >= 3 and opens >= sent * 0.8:
        return "Opens almost every email - a warm audience waiting for a call"
    if web >= 10:
        return f"{web} website visits - they are actively researching you"
    if str(row.get("Rating", "")) == "Hot" and days_quiet <= 7:
        return "Flagged hot by sales and still fresh - confirm and close"
    if 7 <= days_quiet <= 21:
        return f"Quiet for {days_quiet} days - call before this one cools off"
    return "Strong profile fit - worth a conversation this week"


def driver_phrase(driver: str) -> str:
    """Turn a raw model driver name into plain business language."""
    key = driver.replace("__c", "").strip()
    mapping = {
        "Days_Since_Last_Activity": "How recently they were active",
        "Days_in_Pipeline":         "Time spent in the pipeline",
        "Time_to_First_Response_h": "Speed of their first reply",
        "Web_Interactions":         "Website activity",
        "Email_Opens":              "Email opens",
        "Email_Clicks":             "Email engagement",
        "Email_Open_Rate":          "Email open rate",
        "Emails_Sent":              "Email touchpoints",
        "Meetings_Held":            "Meetings held",
        "Proposal_Sent":            "Proposal already sent",
        "Demo_Requested":           "Demo requested",
        "Form_Submissions":         "Forms filled on the site",
        "Content_Downloads":        "Content downloads",
        "Chatbot_Interactions":     "Chat conversations",
        "LinkedIn_Viewed":          "LinkedIn research",
        "Calls_Made":               "Calls made so far",
        "Annual_Revenue_MCZK":      "Company revenue",
        "NumberOfEmployees":        "Company size",
        "Company_Age_Years":        "Company age",
    }
    if key in mapping:
        return mapping[key]
    return key.replace("_", " ").capitalize()


def chip(seg: str) -> str:
    c = SEG_COLOR.get(seg, SLATE)
    return (
        f"<span class='chip' style='color:{c};border-color:{c};"
        f"background:{c}1f'>{seg} priority</span>"
    )


def styled(fig, height: int | None = None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=MUTED, size=13),
        title_font=dict(color=INK, size=15),
        margin=dict(l=8, r=8, t=44, b=8),
        legend=dict(font=dict(color=MUTED)),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    if height:
        fig.update_layout(height=height)
    return fig


# ── Load everything ───────────────────────────────────────────────────────────
master_df     = load_data()
metrics       = load_metrics()
importance_df = load_importance()
val_df        = load_val_predictions()

if not _api_online and not EMBEDDED:
    _logo = (f"<div style='width:210px;margin:0 auto'>{LOGO_SVG.read_text(encoding='utf-8')}</div>"
             if LOGO_SVG.exists() else "<p class='wordmark'>enehano<span class='dot'>.</span></p>")
    st.markdown(
        "<div style='text-align:center;padding:80px 0'>"
        f"{_logo}"
        "<p style='font-size:17px;margin-top:24px'>The scoring service isn't reachable right now.</p>"
        f"<p style='color:{MUTED}'>Once it's back, this page will load your ranked pipeline automatically.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

scored_df = score_pipeline_via_api(master_df)

# ── Swipe gesture callback: JS sets ?swi=<new_idx>, we read + clear it ───────
_swi_raw = st.query_params.get("swi")
if _swi_raw is not None:
    try:
        st.session_state["swipe_idx"] = int(_swi_raw)
        if st.query_params.get("swi_action") == "call":
            st.session_state.setdefault("swipe_called", set())
            _prev_idx = st.session_state["swipe_idx"] - 1
            _swipe_pool = (
                scored_df.sort_values("AI_Score", ascending=False)
                .head(20).reset_index(drop=True)
            )
            if 0 <= _prev_idx < len(_swipe_pool):
                _ico = str(_swipe_pool.iloc[_prev_idx].get("IČO", ""))
                if _ico:
                    st.session_state["swipe_called"].add(_ico)
    except (ValueError, TypeError):
        pass
    st.query_params.clear()


# ── Header ────────────────────────────────────────────────────────────────────
def _logo_html() -> str:
    """Inline the brand SVG inside .logo-wrap so CSS controls the width."""
    if LOGO_SVG.exists():
        svg = LOGO_SVG.read_text(encoding="utf-8")
        return f"<div class='logo-wrap'>{svg}</div>"
    return "<p class='wordmark'>enehano<span class='dot'>.</span></p>"


h_left, h_mid, h_right = st.columns([2, 3, 2])
with h_left:
    st.markdown(
        f"<div class='brand-head'>{_logo_html()}"
        "<p class='tagline' style='margin-top:6px'>Lead intelligence</p></div>",
        unsafe_allow_html=True,
    )
with h_right:
    st.markdown(
        "<p class='status' style='margin-top:14px'>"
        "<span class='live-dot'></span>Scoring live<br>"
        f"{len(scored_df):,} leads ranked</p>",
        unsafe_allow_html=True,
    )

# ── Swipe card HTML builder ───────────────────────────────────────────────────
def _build_swipe_html(cards: list[dict], idx: int) -> str:
    """Single-card HTML component for the Swipe screen with touch gesture support."""
    if not cards:
        return "<p style='color:#7a8898;text-align:center;padding:40px'>No leads.</p>"
    idx    = max(0, min(idx, len(cards) - 1))
    c      = cards[idx]
    total  = len(cards)
    at_end = idx >= total - 1
    nxt    = min(idx + 1, total - 1)

    drivers_html = "".join(
        f"<span class='chip'>{d}</span>" for d in c["drivers"]
    )
    call_url_js = (
        f"setParam({nxt}, 'call')"  if not at_end else "setParam(0, 'reset')"
    )
    skip_url_js = (
        f"setParam({nxt}, 'skip')" if not at_end else "setParam(0, 'reset')"
    )
    at_end_msg = " · Tap either button to restart" if at_end else ""

    return f"""<!doctype html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}}
body{{background:transparent;padding:0 4px;}}
.card{{background:#1a2230;border-radius:18px;padding:22px 18px 18px;
       border:1.5px solid #2a3340;user-select:none;touch-action:pan-y;
       position:relative;overflow:hidden;transition:transform .12s;}}
.ch{{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;}}
.co{{font-size:19px;font-weight:700;color:#f0f4f8;line-height:1.25;}}
.meta{{font-size:12px;color:#7a8898;margin-top:5px;}}
.pill{{background:{c["color"]}22;color:{c["color"]};font-size:26px;font-weight:800;
       border-radius:10px;padding:6px 12px;text-align:center;min-width:58px;flex-shrink:0;}}
.seg{{font-size:10px;text-align:center;color:{c["color"]};font-weight:600;
      letter-spacing:.05em;margin-top:4px;}}
.div{{height:1px;background:#2a3340;margin:14px 0;}}
.action{{color:#a6ce39;font-size:13px;font-weight:600;margin-bottom:10px;}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;}}
.chip{{background:#1e2a1a;color:#a6ce39;border-radius:20px;padding:3px 10px;font-size:12px;}}
.stats{{display:flex;gap:24px;margin-top:14px;}}
.sn{{font-size:17px;font-weight:700;color:#f0f4f8;}}
.sl{{font-size:10px;color:#7a8898;text-transform:uppercase;letter-spacing:.05em;}}
.prog{{text-align:center;color:#7a8898;font-size:12px;margin-top:12px;}}
.overlay{{position:absolute;inset:0;pointer-events:none;opacity:0;
          display:flex;align-items:center;justify-content:center;
          font-size:64px;border-radius:18px;transition:opacity .1s;}}
.skip-ov{{background:rgba(255,75,75,.18);}}
.call-ov{{background:rgba(166,206,57,.18);}}
</style></head><body>
<div class="card" id="card">
  <div class="overlay skip-ov" id="ov-skip">✕</div>
  <div class="overlay call-ov" id="ov-call">✓</div>
  <div class="ch">
    <div>
      <div class="co">{c["company"]}</div>
      <div class="meta">{c["industry"]} · {c["region"]} · {c["source"]}</div>
    </div>
    <div>
      <div class="pill">{c["score"]:.0f}</div>
      <div class="seg">{c["segment"]}</div>
    </div>
  </div>
  <div class="div"></div>
  <div class="action">{c["action"]}</div>
  <div class="chips">{drivers_html}</div>
  <div class="stats">
    <div><div class="sn">{c["expected"]:.0f}%</div><div class="sl">exp win</div></div>
    <div><div class="sn">{c["rule"]}</div><div class="sl">rule score</div></div>
  </div>
  <div class="prog">{idx + 1} / {total}{at_end_msg}</div>
</div>
<script>
const card   = document.getElementById('card');
const ovSkip = document.getElementById('ov-skip');
const ovCall = document.getElementById('ov-call');
let sx = 0, sy = 0, active = false;

function setParam(newIdx, action) {{
  const url = new URL(window.parent.location.href);
  url.searchParams.set('swi', newIdx);
  if (action === 'call') url.searchParams.set('swi_action', 'call');
  else url.searchParams.delete('swi_action');
  window.parent.location.href = url.toString();
}}

card.addEventListener('touchstart', e => {{
  sx = e.touches[0].clientX; sy = e.touches[0].clientY; active = true;
}}, {{passive: true}});

card.addEventListener('touchmove', e => {{
  if (!active) return;
  const dx = e.touches[0].clientX - sx;
  const dy = e.touches[0].clientY - sy;
  if (Math.abs(dx) < Math.abs(dy)) return;
  const t = Math.min(Math.abs(dx) / 120, 1);
  if (dx < 0) {{
    ovSkip.style.opacity = t * 0.9; ovCall.style.opacity = 0;
    card.style.transform = 'translateX(' + (dx * 0.25) + 'px) rotate(' + (dx * 0.02) + 'deg)';
  }} else {{
    ovCall.style.opacity = t * 0.9; ovSkip.style.opacity = 0;
    card.style.transform = 'translateX(' + (dx * 0.25) + 'px) rotate(' + (dx * 0.02) + 'deg)';
  }}
}}, {{passive: true}});

card.addEventListener('touchend', e => {{
  if (!active) return;
  active = false;
  const dx = e.changedTouches[0].clientX - sx;
  card.style.transform = '';
  ovSkip.style.opacity = 0; ovCall.style.opacity = 0;
  if (Math.abs(dx) > 80) {{
    if (dx < 0) {{ {skip_url_js}; }} else {{ {call_url_js}; }}
  }}
}});
</script></body></html>"""


# ── Navigation (session-state driven so screens can hand off to each other) ──
SCREENS = ["Today", "Swipe", "Radar", "All leads", "Lead profile", "For managers"]
if "_goto" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("_goto")
st.session_state.setdefault("nav", "Today")

nav = st.radio("Navigation", SCREENS, key="nav",
               horizontal=True, label_visibility="collapsed")
st.write("")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: Today - the call queue
# ══════════════════════════════════════════════════════════════════════════════
if nav == "Today":
    queue = scored_df.sort_values("AI_Score", ascending=False).head(5)
    queue_wins = float((queue["Expected_Win_%"] / 100).sum())
    high_count = int((scored_df["Segment"] == "High").sum())

    today = pd.Timestamp.now()
    st.markdown(f"<p class='eyebrow'>{today:%A, %d %B}</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-line'>Five calls today.<br>"
        f"<span class='hl'>{queue_wins:.1f} expected wins.</span></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color:{MUTED};font-size:15px'>Ranked by the model. "
        "Work top to bottom - each card tells you why it's worth the call.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='stat-strip'>"
        f"<div><div class='s-val'>{len(scored_df):,}</div><div class='s-lab'>Leads ranked</div></div>"
        f"<div><div class='s-val'>{high_count:,}</div><div class='s-lab'>High priority</div></div>"
        f"<div><div class='s-val'>{queue['AI_Score'].iloc[0]:.0f}</div><div class='s-lab'>Top score</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    for rank, (_, row) in enumerate(queue.iterrows(), start=1):
        seg = row["Segment"]
        col = SEG_COLOR.get(seg, SLATE)
        with st.container(border=True):
            st.markdown(
                f"<div class='q-card'>"
                f"<div class='q-card-rank'>{rank}</div>"
                f"<div class='q-card-body'>"
                f"<div class='q-co'>{row['Company_Name']}</div>"
                f"<div class='q-meta'>{row['Industry']} · {row['Region']} · via {row['LeadSource']}</div>"
                f"<div class='q-why'>{call_reason(row)}</div>"
                f"</div>"
                f"<div class='q-card-right'>"
                f"<div class='q-score' style='color:{col}'>{row['AI_Score']:.0f}</div>"
                f"<div style='margin-top:6px'>{chip(seg)}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Open profile", key=f"open_{row['IČO']}",
                         use_container_width=True):
                st.session_state["selected_ico"] = row["IČO"]
                st.session_state["_goto"] = "Lead profile"
                st.rerun()

    queue_wins = float((queue["Expected_Win_%"] / 100).sum())
    st.markdown(
        f"<p style='color:{MUTED};font-size:14px'>The queue re-ranks itself "
        "as new activity lands in the CRM.</p>",
        unsafe_allow_html=True,
    )

    st.write("")
    _, mid, _ = st.columns([2, 1.4, 2])
    with mid:
        if st.button("See all leads", type="primary", use_container_width=True):
            st.session_state["_goto"] = "All leads"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: Swipe — one card at a time with touch gesture support
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Swipe":
    from streamlit.components.v1 import html as _sv_html
    from ui_cards import _row_to_card as _to_card

    st.markdown("## Top leads")
    st.markdown(
        f"<p style='color:{MUTED};margin-top:-8px'>"
        "Swipe right (or tap <b>Call</b>) to queue a lead, "
        "swipe left (or tap <b>Skip</b>) to move on.</p>",
        unsafe_allow_html=True,
    )

    pool = (
        scored_df.sort_values("AI_Score", ascending=False)
        .head(20).reset_index(drop=True)
    )

    if pool.empty:
        st.info("No scored leads available.")
    else:
        cards = [_to_card(r) for _, r in pool.iterrows()]

        st.session_state.setdefault("swipe_idx", 0)
        st.session_state.setdefault("swipe_called", set())

        sw_idx = int(st.session_state["swipe_idx"])
        if sw_idx >= len(cards):
            sw_idx = 0
            st.session_state["swipe_idx"] = 0

        _sv_html(_build_swipe_html(cards, sw_idx), height=440, scrolling=False)

        col_skip, col_call = st.columns(2)
        with col_skip:
            if st.button("← Skip", use_container_width=True, key="swipe_skip_btn"):
                st.session_state["swipe_idx"] = min(sw_idx + 1, len(cards) - 1)
                st.rerun()
        with col_call:
            if st.button("📞 Call this lead", use_container_width=True,
                         type="primary", key="swipe_call_btn"):
                st.session_state["swipe_called"].add(cards[sw_idx]["ico"])
                st.session_state["swipe_idx"] = min(sw_idx + 1, len(cards) - 1)
                st.toast(f"Queued: {cards[sw_idx]['company']}", icon="✓")
                st.rerun()

        called_count = len(st.session_state["swipe_called"])
        if called_count:
            st.markdown(
                f"<p style='text-align:center;color:{LIME};font-size:13px;"
                f"margin-top:6px'>"
                f"{called_count} lead{'s' if called_count > 1 else ''} queued for a call"
                "</p>",
                unsafe_allow_html=True,
            )

        st.write("")
        if st.button("Open full profile", use_container_width=True, key="swipe_open_btn"):
            st.session_state["selected_ico"] = cards[sw_idx]["ico"]
            st.session_state["_goto"] = "Lead profile"
            st.rerun()

        if st.button("Reset deck", use_container_width=True, key="swipe_reset_btn"):
            st.session_state["swipe_idx"] = 0
            st.session_state["swipe_called"] = set()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: All leads
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "All leads":
    st.markdown("## Every lead, ranked")
    st.markdown(
        f"<p style='color:{MUTED};margin-top:-8px'>Highest priority first. "
        "Click any row to open its full profile.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("Filters"):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            seg_f = st.multiselect("Priority", ["High", "Medium", "Low"],
                                   default=["High", "Medium"])
        with f2:
            ind_f = st.multiselect("Industry", sorted(scored_df["Industry"].unique()))
        with f3:
            src_f = st.multiselect("Lead source", sorted(scored_df["LeadSource"].unique()))
        with f4:
            reg_f = st.multiselect("Region", sorted(scored_df["Region"].unique()))

    view = scored_df.copy()
    if seg_f: view = view[view["Segment"].isin(seg_f)]
    if ind_f: view = view[view["Industry"].isin(ind_f)]
    if src_f: view = view[view["LeadSource"].isin(src_f)]
    if reg_f: view = view[view["Region"].isin(reg_f)]
    view = view.sort_values("AI_Score", ascending=False)

    k1, k2, k3 = st.columns(3)
    k1.metric("Leads in view", f"{len(view):,}")
    k2.metric("High priority", f"{(view['Segment'] == 'High').sum():,}")
    k3.metric("Expected wins in view", f"{(view['Expected_Win_%'] / 100).sum():,.0f}")

    display_cols = ["IČO", "Company_Name", "Industry", "Region", "LeadSource",
                    "AI_Score", "Segment", "Expected_Win_%"]
    selection = st.dataframe(
        view[display_cols], use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "IČO":            st.column_config.TextColumn("Company ID"),
            "Company_Name":   st.column_config.TextColumn("Company"),
            "LeadSource":     st.column_config.TextColumn("Source"),
            "AI_Score":       st.column_config.ProgressColumn(
                                  "Score", min_value=0, max_value=100, format="%.0f"),
            "Segment":        st.column_config.TextColumn("Priority"),
            "Expected_Win_%": st.column_config.NumberColumn(
                                  "Expected win", format="%.1f%%"),
        },
    )

    if selection.selection.rows:
        st.session_state["selected_ico"] = view.iloc[selection.selection.rows[0]]["IČO"]
        st.session_state["_goto"] = "Lead profile"
        st.rerun()

    st.write("")
    _, btn_col, _ = st.columns([2, 1.6, 2])
    with btn_col:
        st.download_button(
            "Export to CSV for Salesforce",
            view[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="enehano_scored_leads.csv", mime="text/csv",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: Lead profile
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Lead profile":
    ranked = scored_df.sort_values("AI_Score", ascending=False)
    option_labels = {
        f"{r['Company_Name']}  ·  {r['AI_Score']:.0f}  ·  {r['IČO']}": r["IČO"]
        for _, r in ranked.iterrows()
    }
    labels = list(option_labels.keys())

    preselected = st.session_state.get("selected_ico")
    default_index = 0
    if preselected:
        for i, label in enumerate(labels):
            if label.endswith(preselected):
                default_index = i
                break

    picked = st.selectbox(
        "Find a company - type to search",
        labels, index=default_index,
    )
    ico = option_labels[picked]
    st.session_state["selected_ico"] = ico
    row = scored_df[scored_df["IČO"] == ico].iloc[0]

    st.markdown(f"## {row['Company_Name']}")
    st.markdown(
        f"<p style='color:{MUTED};margin-top:-8px'>"
        f"ID {row['IČO']} · {row.get('Legal_Form_Label', '')} · "
        f"{row['Industry']} · {row['Region']}</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    col_score, col_why = st.columns([1, 1.35], gap="large")

    # What-if controls live on the page, in the "why" column, under the drivers
    with col_why:
        st.markdown("<p class='eyebrow'>What if you acted?</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:{MUTED};font-size:14px;margin-top:-2px'>"
            "Move a slider - the score on the left updates live.</p>",
            unsafe_allow_html=True,
        )
        w1, w2 = st.columns(2)
        with w1:
            sim_ttfr = st.slider("First response time (hours)", 0.1, 72.0,
                                 float(row["Time_to_First_Response_h__c"]), 0.5,
                                 key=f"ttfr_{ico}")
            sim_web  = st.slider("Website visits", 0, 100,
                                 int(row["Web_Interactions__c"]), key=f"web_{ico}")
            sim_meet = st.slider("Meetings held", 0, 10,
                                 int(row["Meetings_Held__c"]), key=f"meet_{ico}")
        with w2:
            sim_forms = st.slider("Forms filled", 0, 10,
                                  int(row["Form_Submissions__c"]), key=f"forms_{ico}")
            sim_demo = st.checkbox("Demo requested",
                                   bool(row["Demo_Requested__c"]), key=f"demo_{ico}")
            sim_li   = st.checkbox("Viewed on LinkedIn",
                                   bool(row["LinkedIn_Viewed__c"]), key=f"li_{ico}")

    behavioral_override = {
        "Time_to_First_Response_h__c": sim_ttfr,
        "Web_Interactions__c":          sim_web,
        "Meetings_Held__c":             sim_meet,
        "Form_Submissions__c":          sim_forms,
        "Demo_Requested__c":            int(sim_demo),
        "LinkedIn_Viewed__c":           int(sim_li),
        "Email_Opens__c":              int(row.get("Email_Opens__c", 0)),
        "Email_Clicks__c":             int(row.get("Email_Clicks__c", 0)),
        "Emails_Sent__c":              int(row.get("Emails_Sent__c", 0)),
        "Email_Open_Rate__c":          float(row.get("Email_Open_Rate__c", 0.0)),
        "Calls_Made__c":               int(row.get("Calls_Made__c", 0)),
        "Proposal_Sent__c":            int(row.get("Proposal_Sent__c", 0)),
        "Content_Downloads__c":        int(row.get("Content_Downloads__c", 0)),
        "Chatbot_Interactions__c":     int(row.get("Chatbot_Interactions__c", 0)),
        "Days_Since_Last_Activity__c": int(row.get("Days_Since_Last_Activity__c", 30)),
        "Days_in_Pipeline__c":         int(row.get("Days_in_Pipeline__c", 30)),
        "Industry":                    str(row.get("Industry", "Technology")),
        "Size_Band":                   str(row.get("Size_Band", "Small")),
        "NumberOfEmployees":           int(row.get("NumberOfEmployees", 25)),
        "Annual_Revenue_MCZK__c":      float(row.get("Annual_Revenue_MCZK__c", 30.0)),
        "LeadSource":                  str(row.get("LeadSource", "Web")),
        "Rating":                      str(row.get("Rating", "Warm")),
        # Firmographics — API fetches these from ARES; embedded reads them from the row
        "CZ_NACE_Section":       str(row.get("CZ_NACE_Section", "J")),
        "CZ_NACE_Section_Label": str(row.get("CZ_NACE_Section_Label", "Information and communication")),
        "Region_Name":           str(row.get("Region_Name", row.get("Region", "Praha"))),
        "Region_Code":           str(row.get("Region_Code", "CZ010")),
        "Legal_Form_Code":       str(row.get("Legal_Form_Code", "112")),
        "Legal_Form_Label":      str(row.get("Legal_Form_Label", "s.r.o.")),
        "Founding_Year":         int(row.get("Founding_Year", 2010)),
        "Company_Age_Years":     int(row.get("Company_Age_Years", 15)),
    }

    original_score = float(row["AI_Score"])
    if EMBEDDED:
        sim_result = _scorer.score_lead(behavioral_override)
    else:
        sim_result = APIClient.score_enrich(str(ico), behavioral_override)

    if sim_result:
        new_score    = float(sim_result["ai_score"])
        expected_win = float(sim_result["expected_win_pct"])
        api_drivers  = sim_result.get("top_drivers", [])
    else:
        new_score    = original_score
        expected_win = float(row.get("Expected_Win_%", 0))
        api_drivers  = list(row.get("top_drivers", []) or [])

    seg = score_segment(new_score)
    seg_c = SEG_COLOR.get(seg, SLATE)
    diff = new_score - original_score

    with col_score:
        with st.container(border=True):
            st.markdown("<p class='eyebrow'>Priority score</p>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='big-score' style='color:{seg_c}'>{new_score:.0f}"
                f"<span style='font-size:26px;color:{MUTED}'> / 100</span></div>",
                unsafe_allow_html=True,
            )
            if abs(diff) >= 0.5:
                arrow = "up" if diff > 0 else "down"
                st.markdown(
                    f"<p style='color:{MUTED};font-size:14px'>Was {original_score:.0f} - "
                    f"your changes move it <b style='color:{seg_c}'>{abs(diff):.0f} points {arrow}</b></p>",
                    unsafe_allow_html=True,
                )
            st.markdown(chip(seg), unsafe_allow_html=True)
            st.markdown(
                f"<p style='margin-top:12px;font-size:15px'>{bh.explain_segment(seg)}</p>",
                unsafe_allow_html=True,
            )
            st.metric("Expected win chance", f"{expected_win:.1f}%")

    with col_why:
        st.write("")
        st.markdown("<p class='eyebrow'>Why this score</p>", unsafe_allow_html=True)
        if api_drivers:
            for d in api_drivers:
                st.markdown(
                    f"<div class='finding'>{driver_phrase(str(d))}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"<p style='color:{MUTED}'>Driver breakdown isn't available for this lead right now.</p>",
                unsafe_allow_html=True,
            )

    st.write("")
    with st.expander("Company record from the Czech business register"):
        if st.button("Look up the latest registry data"):
            with st.spinner("Checking the register..."):
                ares_result = APIClient.score_enrich(str(ico), behavioral_override)
            if not isinstance(ares_result, dict):
                st.warning("The register is temporarily unavailable. Try again in a minute.")
            else:
                hidden = {"ai_score", "segment", "expected_win_pct",
                          "p_win_if_converted_pct", "rule_score",
                          "top_drivers", "version"}
                shown = {k: v for k, v in ares_result.items() if k not in hidden}
                if shown:
                    for k, v in shown.items():
                        st.write(f"**{k.replace('_', ' ').title()}:** {v}")
                else:
                    st.info("The register returned no additional public details for this company.")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: Radar - what the old playbook misses
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Radar":
    AVG_DEAL = 500_000  # CZK, same default as the revenue calculator

    days_q = pd.to_numeric(scored_df["Days_Since_Last_Activity__c"],
                           errors="coerce").fillna(0)

    cooling = scored_df[
        (scored_df["Segment"] == "High") & days_q.between(7, 30)
    ].sort_values("Expected_Win_%", ascending=False)

    gems = scored_df[
        (scored_df["Rule_Score"] < 40) & (scored_df["AI_Score"] >= 70)
    ].sort_values("AI_Score", ascending=False)

    cooling_value = float((cooling["Expected_Win_%"] / 100).sum()) * AVG_DEAL
    gems_value    = float((gems["Expected_Win_%"] / 100).sum()) * AVG_DEAL

    st.markdown("<p class='eyebrow'>Radar</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-line'>Two lists the old<br>"
        "<span class='hl'>playbook never sees.</span></p>",
        unsafe_allow_html=True,
    )
    st.write("")

    # ── Going cold ────────────────────────────────────────────────────────
    st.markdown(
        f"<p class='eyebrow' style='color:{AMBER}'>Going cold</p>"
        f"<p style='font-size:17px;margin:0 0 2px 0'><b>{len(cooling):,} high-priority "
        f"leads are going quiet</b> - roughly "
        f"<b style='color:{AMBER}'>{cooling_value/1_000_000:,.1f}M CZK</b> slipping away "
        "while nobody calls.</p>"
        f"<p style='color:{MUTED};font-size:13px'>High scores with 7-30 days of silence. "
        f"Estimated at a {AVG_DEAL/1000:,.0f}k CZK average deal.</p>",
        unsafe_allow_html=True,
    )

    if cooling.empty:
        st.markdown(
            f"<p style='color:{MUTED}'>Nothing is going cold right now - every "
            "high-priority lead has been touched in the last week.</p>",
            unsafe_allow_html=True,
        )
    for _, row in cooling.head(4).iterrows():
        quiet = int(pd.to_numeric(row["Days_Since_Last_Activity__c"], errors="coerce") or 0)
        with st.container(border=True):
            st.markdown(
                f"<div class='q-card'>"
                f"<div class='q-card-left'>"
                f"<div style='font-family:Space Grotesk,sans-serif;font-size:28px;"
                f"font-weight:700;color:{AMBER};line-height:1.05'>{quiet}</div>"
                f"<div style='font-size:11px;color:{MUTED};font-weight:600;text-transform:uppercase;"
                f"letter-spacing:.06em'>days<br>quiet</div>"
                f"</div>"
                f"<div class='q-card-body'>"
                f"<div class='q-co'>{row['Company_Name']}</div>"
                f"<div class='q-meta'>{row['Industry']} · {row['Region']}</div>"
                f"<div class='q-why' style='border-color:{AMBER}'>Was hot — one call could bring it back.</div>"
                f"</div>"
                f"<div class='q-card-right'>"
                f"<div class='q-score' style='color:{LIME}'>{row['AI_Score']:.0f}</div>"
                f"<div style='font-size:11px;color:{MUTED};margin-top:4px'>{row['Expected_Win_%']:.0f}% win</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Open profile", key=f"cold_{row['IČO']}",
                         use_container_width=True):
                st.session_state["selected_ico"] = row["IČO"]
                st.session_state["_goto"] = "Lead profile"
                st.rerun()

    st.write("")

    # ── Hidden gems ───────────────────────────────────────────────────────
    st.markdown(
        f"<p class='eyebrow' style='color:{LIME}'>Hidden gems</p>"
        f"<p style='font-size:17px;margin:0 0 2px 0'><b>{len(gems):,} leads the old rules "
        f"would have skipped</b> - worth an estimated "
        f"<b style='color:{LIME}'>{gems_value/1_000_000:,.1f}M CZK</b> the team never "
        "would have called.</p>"
        f"<p style='color:{MUTED};font-size:13px'>Scored under 40 by the previous "
        "rule-based approach, over 70 by the model.</p>",
        unsafe_allow_html=True,
    )

    if gems.empty:
        st.markdown(
            f"<p style='color:{MUTED}'>No disagreements today - the model and the old "
            "rules currently agree on who the top leads are.</p>",
            unsafe_allow_html=True,
        )
    for _, row in gems.head(4).iterrows():
        with st.container(border=True):
            st.markdown(
                f"<div class='q-card'>"
                f"<div class='q-card-left'>"
                f"<div style='font-family:Space Grotesk,sans-serif;font-weight:700;line-height:1.1'>"
                f"<div style='font-size:22px;color:{SLATE};text-decoration:line-through'>{row['Rule_Score']:.0f}</div>"
                f"<div style='font-size:11px;color:{MUTED};margin:-2px 0 2px'>&#8595; model</div>"
                f"<div style='font-size:30px;color:{LIME}'>{row['AI_Score']:.0f}</div>"
                f"</div>"
                f"</div>"
                f"<div class='q-card-body'>"
                f"<div class='q-co'>{row['Company_Name']}</div>"
                f"<div class='q-meta'>{row['Industry']} · {row['Region']}</div>"
                f"<div class='q-why'>{call_reason(row)}</div>"
                f"</div>"
                f"<div class='q-card-right'>"
                f"<div style='font-size:11px;color:{MUTED};margin-top:8px'>{row['Expected_Win_%']:.0f}% win</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Open profile", key=f"gem_{row['IČO']}",
                         use_container_width=True):
                st.session_state["selected_ico"] = row["IČO"]
                st.session_state["_goto"] = "Lead profile"
                st.rerun()

    st.markdown(
        f"<p style='color:{MUTED};font-size:13px;margin-top:10px'>Both lists refresh "
        "with every scoring run - no one has to remember to check.</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: For managers
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "For managers":
    st.markdown("## For managers")
    st.markdown(
        f"<p style='color:{MUTED};margin-top:-8px'>Where the pipeline value sits, "
        "who holds it, and what prioritisation is worth in revenue.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    # ── Pipeline at a glance ──────────────────────────────────────────────
    st.markdown("<p class='eyebrow'>Pipeline at a glance</p>", unsafe_allow_html=True)
    total_leads   = len(scored_df)
    high          = int((scored_df["Segment"] == "High").sum())
    expected_wins = int((scored_df["Expected_Win_%"] / 100).sum())
    avg_score     = scored_df["AI_Score"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total leads", f"{total_leads:,}")
    k2.metric("High priority", f"{high:,}", delta=f"{high / total_leads:.0%} of pipeline",
              delta_color="off")
    k3.metric("Expected wins", f"{expected_wins:,}")
    k4.metric("Average score", f"{avg_score:.0f} / 100")

    if val_df is not None:
        lift10 = bh.lift_table(val_df["y_true"].values,
                               val_df["ai_proba"].values, bins=10).iloc[0]
        st.markdown(
            f"<div class='finding'>Working only the top 10% of leads, the team would catch "
            f"<b>{lift10['cum_recall']:.0%} of all conversions</b> - "
            f"<b>{lift10['lift']:.1f}x more</b> than working leads in random order.</div>",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        seg_counts = (scored_df["Segment"].value_counts()
                      .reindex(["High", "Medium", "Low"]).fillna(0))
        fig = px.pie(values=seg_counts.values, names=seg_counts.index, hole=0.62,
                     color=seg_counts.index, color_discrete_map=SEG_COLOR,
                     title="Priority mix")
        fig.update_layout(legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(styled(fig, 360), use_container_width=True)
    with c2:
        fig = px.histogram(scored_df, x="AI_Score", nbins=25, color="Segment",
                           color_discrete_map=SEG_COLOR, title="Score distribution")
        fig.update_layout(bargap=0.06, xaxis_title="Score (0-100)",
                          yaxis_title="Leads", showlegend=False)
        st.plotly_chart(styled(fig, 360), use_container_width=True)

    st.divider()

    # ── Where the money sits ──────────────────────────────────────────────
    st.markdown("<p class='eyebrow'>Where the money sits</p>", unsafe_allow_html=True)
    AVG_DEAL = 500_000
    days_q = pd.to_numeric(scored_df["Days_Since_Last_Activity__c"],
                           errors="coerce").fillna(0)
    untouched_high = scored_df[(scored_df["Segment"] == "High") & (days_q >= 7)]
    on_table = float((untouched_high["Expected_Win_%"] / 100).sum()) * AVG_DEAL

    w1, w2 = st.columns([1, 1.6])
    with w1:
        with st.container(border=True):
            st.markdown(
                f"<div class='h2h'><div class='n' style='color:{AMBER}'>"
                f"{on_table/1_000_000:,.1f}M</div>"
                "<div class='l'>CZK sitting in high-priority leads<br>"
                "untouched for a week or more</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='color:{MUTED};font-size:12px;text-align:center'>"
                f"{len(untouched_high):,} leads · estimated at a "
                f"{AVG_DEAL/1000:,.0f}k CZK average deal · the rescue list lives in Radar</p>",
                unsafe_allow_html=True,
            )
    with w2:
        if "Owner" in scored_df.columns:
            team = (
                scored_df.groupby("Owner")
                .agg(leads=("Lead_ID", "count"),
                     high=("Segment", lambda s: (s == "High").sum()),
                     expected_wins=("Expected_Win_%", lambda s: (s / 100).sum()))
                .reset_index()
                .sort_values("expected_wins", ascending=False)
                .head(10)
            )
            fig = px.bar(team.sort_values("expected_wins"),
                         x="expected_wins", y="Owner", orientation="h",
                         title="Expected wins held per rep - is the gold spread evenly?")
            fig.update_traces(marker_color=LIME)
            fig.update_layout(yaxis_title="", xaxis_title="Expected wins in book")
            st.plotly_chart(styled(fig, 340), use_container_width=True)

    st.divider()


    st.markdown("<p class='eyebrow'>Revenue impact</p>", unsafe_allow_html=True)
    if val_df is None:
        st.info("Validation data isn't available in this build, so the revenue "
                "calculator is switched off.")
    else:
        st.markdown(
            f"<p style='color:{MUTED};font-size:14px'>Plug in your own numbers. "
            "The model estimates revenue from prioritising its top picks "
            "against random order and the previous rule-based approach.</p>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            pipeline_size = st.number_input("Leads per quarter", 100, 200_000, 10_000, 500)
            avg_deal = st.number_input("Average deal value (CZK)",
                                       10_000, 50_000_000, 500_000, 10_000)
        with c2:
            win_rate   = st.slider("Win rate once converted (%)", 10, 95, 65) / 100
            ai_top_pct = st.slider("Reps work the top X% of leads", 5, 100, 20)
        with c3:
            rep_capacity = st.number_input("Leads per rep per quarter", 10, 2000, 200, 10)
            rep_cost     = st.number_input("Cost per rep per quarter (CZK)",
                                           50_000, 2_000_000, 350_000, 10_000)

        ai_lift   = bh.lift_table(val_df["y_true"].values, val_df["ai_proba"].values, bins=20)
        base_lift = bh.lift_table(val_df["y_true"].values, val_df["baseline_proba"].values, bins=20)
        base_rate = float(val_df["y_true"].mean())

        r = bh.compute_roi(
            bh.ROIInputs(pipeline_size=pipeline_size, avg_deal_value=avg_deal,
                         win_rate_if_converted=win_rate, rep_capacity=rep_capacity,
                         rep_cost=rep_cost, ai_top_pct=ai_top_pct),
            ai_lift, base_lift, base_rate,
        )

        def _m(x: float) -> str:
            return f"{x / 1_000_000:,.1f}M CZK"

        m1, m2, m3 = st.columns(3)
        m1.metric("Random order", _m(r.revenue_random))
        m2.metric("Rule-based scoring", _m(r.revenue_baseline))
        m3.metric("AI scoring", _m(r.revenue_ai),
                  delta=f"+{_m(r.uplift_vs_baseline)} vs rules")

        st.markdown(
            f"<div class='finding'>At top-{ai_top_pct}% prioritisation, AI scoring is expected "
            f"to add <b>{_m(r.uplift_vs_baseline)}</b> per quarter over the previous approach - "
            f"a <b>{(r.uplift_vs_baseline / max(1, r.revenue_baseline)) * 100:.0f}% uplift</b> "
            f"using {r.reps_needed} reps.</div>",
            unsafe_allow_html=True,
        )

        # Executive summary PDF
        try:
            import exec_pdf
            top5 = scored_df.sort_values("AI_Score", ascending=False).head(5)[
                ["Company_Name", "Industry", "Region", "AI_Score", "Expected_Win_%"]
            ]
            pdf_bytes = exec_pdf.build_pdf(
                headline_uplift=r.uplift_vs_baseline, currency="CZK",
                ai_auc=metrics["conversion_model"]["test"]["auc_roc"] if metrics else 0,
                base_auc=metrics["baseline_conversion"]["auc_roc"] if metrics else 0,
                total_leads=total_leads, high_leads=high, expected_wins=expected_wins,
                top_leads=top5,
                random_rev=r.revenue_random, baseline_rev=r.revenue_baseline,
                ai_rev=r.revenue_ai,
            )
            _, pdf_col, _ = st.columns([2, 1.8, 2])
            with pdf_col:
                st.download_button(
                    "Download executive summary (PDF)", data=pdf_bytes,
                    file_name=f"enehano_executive_summary_{pd.Timestamp.now():%Y%m%d}.pdf",
                    mime="application/pdf", use_container_width=True,
                )
        except Exception:
            pass

    st.divider()

    # ── Technical validation (off by default - managers don't need it) ────
    show_tech = st.toggle("Show technical validation (for analysts)", value=False)
    if show_tech and (val_df is None or metrics is None):
        st.info("Model diagnostics aren't available in this build.")
    elif show_tech:
        st.markdown("<p class='eyebrow'>Model quality</p>", unsafe_allow_html=True)
        y      = val_df["y_true"].values
        p_ai   = val_df["ai_proba"].values
        p_base = val_df["baseline_proba"].values

        ai_auc   = bh.auc(y, p_ai)
        base_auc = bh.auc(y, p_base)
        cv_std   = metrics["conversion_model"]["cv"]["auc_std"]

        q1, q2, q3 = st.columns(3)
        q1.metric("AI ranking quality", f"{ai_auc:.0%}",
                  delta=f"+{(ai_auc - base_auc) * 100:.0f} pts vs rules")
        q2.metric("Rule-based ranking quality", f"{base_auc:.0%}")
        q3.metric("Run-to-run consistency",
                  "Stable" if cv_std < 0.01 else "Variable",
                  delta=f"±{cv_std:.1%} across 5 retrains", delta_color="off")

        st.markdown("##### Calibrate the call threshold")
        thr = st.slider("Minimum score worth a call (0-100)", 5, 95, 50, 5)
        ts  = bh.threshold_stats(y, p_ai * 100, thr)

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Leads to call", f"{ts.leads_to_work:,}",
                  delta=f"{ts.leads_to_work_pct:.0%} of all leads", delta_color="off")
        t2.metric("Conversions caught", f"{ts.captured_conversions:,}",
                  delta=f"{ts.captured_pct:.0%} of all wins", delta_color="off")
        t3.metric("Hit rate", f"{ts.precision:.0%}")
        t4.metric("Coverage", f"{ts.recall:.0%}")

        st.markdown(
            f"<div class='finding'>At a minimum score of <b>{thr}</b>, the team works "
            f"<b>{ts.leads_to_work_pct:.0%}</b> of leads and still catches "
            f"<b>{ts.captured_pct:.0%}</b> of all conversions.</div>",
            unsafe_allow_html=True,
        )

        lift_ai_t   = bh.lift_table(y, p_ai, bins=20)
        lift_base_t = bh.lift_table(y, p_base, bins=20)
        gains = pd.concat([
            lift_ai_t.assign(Strategy="AI")[["top_pct", "cum_recall", "Strategy"]],
            lift_base_t.assign(Strategy="Rule-based")[["top_pct", "cum_recall", "Strategy"]],
            pd.DataFrame({"top_pct": [0, 100], "cum_recall": [0, 1.0],
                          "Strategy": ["Random", "Random"]}),
        ])
        fig = px.line(gains, x="top_pct", y="cum_recall", color="Strategy",
                      color_discrete_map={"AI": LIME, "Rule-based": AMBER, "Random": SLATE},
                      labels={"top_pct": "% of leads worked (best first)",
                              "cum_recall": "% of conversions captured"},
                      title="Effort against results - higher and earlier is better")
        fig.update_layout(yaxis_tickformat=".0%", xaxis_ticksuffix="%")
        st.plotly_chart(styled(fig, 420), use_container_width=True)

        if importance_df is not None:
            st.markdown("##### What signals drive the score")
            top12 = importance_df.head(12).copy()
            top12["feature"] = top12["feature"].map(driver_phrase)
            fig = px.bar(top12.sort_values("importance"), x="importance", y="feature",
                         orientation="h", title="Strongest signals in the model")
            fig.update_traces(marker_color=LIME)
            fig.update_layout(yaxis_title="", xaxis_title="Relative weight")
            st.plotly_chart(styled(fig, 420), use_container_width=True)

        st.markdown("##### Score stability")
        rng  = np.random.default_rng(42)
        idx  = rng.permutation(len(p_ai))
        half = len(idx) // 2
        ref_scores = p_ai[idx[:half]] * 100
        cur_scores = p_ai[idx[half:]] * 100
        psi = bh.population_stability_index(ref_scores, cur_scores, bins=10)

        if psi < 0.10:
            psi_label, psi_color = "Stable", LIME
        elif psi < 0.25:
            psi_label, psi_color = "Moderate shift - review", AMBER
        else:
            psi_label, psi_color = "Major drift - retrain", RED

        d1, d2, d3 = st.columns([1, 1, 2])
        d1.metric("Stability index", f"{psi:.3f}")
        d2.markdown(
            f"<div style='border:1px solid {psi_color};border-radius:8px;"
            f"padding:14px;text-align:center;font-weight:700;font-size:16px;"
            f"color:{psi_color};margin-top:4px'>{psi_label}</div>",
            unsafe_allow_html=True,
        )
        d3.markdown(
            f"<p style='color:{MUTED};font-size:13px;margin-top:8px'>"
            "Compares score distributions between past and recent leads. "
            "A value under 0.10 means the model's behaviour hasn't shifted.</p>",
            unsafe_allow_html=True,
        )

        with st.expander("Where the leads are - regional view"):
            KRAJ_COORDS = {
                "Praha": (50.075, 14.437), "Středočeský": (49.876, 14.789),
                "Jihočeský": (49.124, 14.500), "Plzeňský": (49.747, 13.378),
                "Karlovarský": (50.232, 12.871), "Ústecký": (50.661, 13.832),
                "Liberecký": (50.770, 15.058), "Královéhradecký": (50.342, 15.792),
                "Pardubický": (49.945, 16.275), "Kraj Vysočina": (49.397, 15.589),
                "Jihomoravský": (49.195, 16.608), "Olomoucký": (49.594, 17.250),
                "Zlínský": (49.224, 17.666), "Moravskoslezský": (49.823, 18.262),
            }
            region_stats = (
                scored_df.groupby("Region")
                .agg(leads=("Lead_ID", "count"),
                     avg_score=("AI_Score", "mean"),
                     high_potential=("Segment", lambda s: (s == "High").sum()),
                     expected_wins=("Expected_Win_%", lambda s: (s / 100).sum()))
                .reset_index()
            )
            region_stats["avg_score"]     = region_stats["avg_score"].round(0)
            region_stats["expected_wins"] = region_stats["expected_wins"].round(0).astype(int)

            def _coord(name: str) -> tuple:
                r = name.strip()
                if r in KRAJ_COORDS:
                    return KRAJ_COORDS[r]
                stripped = r.replace(" kraj", "").replace("Kraj ", "").strip()
                for k, v in KRAJ_COORDS.items():
                    if stripped.lower() in k.lower() or k.lower() in stripped.lower():
                        return v
                return (None, None)

            region_stats["lat"] = region_stats["Region"].map(lambda x: _coord(x)[0])
            region_stats["lon"] = region_stats["Region"].map(lambda x: _coord(x)[1])
            mapped = region_stats.dropna(subset=["lat", "lon"])

            if len(mapped):
                fig = px.scatter_geo(
                    mapped, lat="lat", lon="lon", size="leads", color="avg_score",
                    hover_name="Region",
                    hover_data={"leads": True, "avg_score": True,
                                "high_potential": True, "expected_wins": True,
                                "lat": False, "lon": False},
                    color_continuous_scale=[[0, SLATE], [0.5, AMBER], [1, LIME]],
                    size_max=52, projection="mercator",
                )
                fig.update_geos(visible=False, resolution=50,
                                showcountries=True, countrycolor=BORDER,
                                showland=True, landcolor="#10161d",
                                lataxis_range=[48.5, 51.2], lonaxis_range=[12.0, 19.0])
                fig.update_layout(coloraxis_colorbar=dict(title="Avg score"))
                st.plotly_chart(styled(fig, 480), use_container_width=True)

            st.dataframe(
                region_stats[["Region", "leads", "high_potential",
                              "avg_score", "expected_wins"]]
                .sort_values("expected_wins", ascending=False),
                use_container_width=True, hide_index=True,
                column_config={
                    "leads":          st.column_config.NumberColumn("Leads", format="%d"),
                    "high_potential": st.column_config.NumberColumn("High priority", format="%d"),
                    "avg_score":      st.column_config.NumberColumn("Avg score", format="%.0f"),
                    "expected_wins":  st.column_config.NumberColumn("Expected wins", format="%d"),
                },
            )

        with st.expander("Technical curves and raw metrics"):
            roc_a, roc_b = bh.roc_points(y, p_ai), bh.roc_points(y, p_base)
            pr_a,  pr_b  = bh.pr_points(y, p_ai),  bh.pr_points(y, p_base)
            cc1, cc2 = st.columns(2)
            with cc1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=roc_a["fpr"], y=roc_a["tpr"],
                                         name=f"AI (AUC {ai_auc:.3f})",
                                         line=dict(color=LIME, width=3)))
                fig.add_trace(go.Scatter(x=roc_b["fpr"], y=roc_b["tpr"],
                                         name=f"Rules (AUC {base_auc:.3f})",
                                         line=dict(color=AMBER, width=2)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                         line=dict(color=SLATE, dash="dash", width=1)))
                fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                                  yaxis_title="True positive rate",
                                  legend=dict(x=0.45, y=0.08))
                st.plotly_chart(styled(fig, 380), use_container_width=True)
            with cc2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pr_a["recall"], y=pr_a["precision"],
                                         name="AI", line=dict(color=LIME, width=3)))
                fig.add_trace(go.Scatter(x=pr_b["recall"], y=pr_b["precision"],
                                         name="Rules", line=dict(color=AMBER, width=2)))
                fig.update_layout(title="Precision against recall",
                                  xaxis_title="Recall", yaxis_title="Precision",
                                  legend=dict(x=0.05, y=0.08))
                st.plotly_chart(styled(fig, 380), use_container_width=True)

            conv_t = metrics["conversion_model"]["test"]
            base_c = metrics["baseline_conversion"]
            comp = pd.DataFrame({
                "Metric":     ["AUC-ROC", "PR-AUC", "Precision", "Recall", "F1"],
                "Rule-based": [base_c["auc_roc"], base_c["pr_auc"],
                               base_c["precision"], base_c["recall"], base_c["f1"]],
                "AI":         [conv_t["auc_roc"], conv_t["pr_auc"],
                               conv_t["precision"], conv_t["recall"], conv_t["f1"]],
            })
            comp["Uplift"] = comp["AI"] - comp["Rule-based"]
            st.dataframe(
                comp.style.format({"Rule-based": "{:.3f}", "AI": "{:.3f}",
                                   "Uplift": "{:+.3f}"}),
                use_container_width=True, hide_index=True,
            )
            cv = metrics["conversion_model"]["cv"]
            st.caption(
                f"Cross-validated AUC (5 folds): {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f} · "
                f"Win model test AUC: {metrics['win_model']['test']['auc_roc']:.4f} · "
                f"Dataset: {metrics['n_rows']:,} leads, {metrics['n_converted']:,} converted, "
                f"{metrics['n_won']:,} won."
            )


# ── Footer + chat ─────────────────────────────────────────────────────────────
st.write("")


st.markdown(
    f"<p style='text-align:center;color:{MUTED};font-size:12px'>"
    "Enehano Solutions · Lead Intelligence</p>",
    unsafe_allow_html=True,
)

import ui_chat
ui_chat.render_bubble(scored_df)