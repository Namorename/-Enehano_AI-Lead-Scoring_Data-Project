"""
app.py
======
Enehano Lead Intelligence — Streamlit dashboard (API-backed edition).

All model inference is delegated to the FastAPI backend (api.py).
No joblib / sklearn / shap imports — the frontend is a pure UI layer.

Column contract:
  Region        → generator writes Region_Name; load_data() adds a 'Region' alias.
  AI_Score      → populated by score_pipeline() via the /score/batch API endpoint.
  Rule_Score    → returned by the API ScoreOutput.rule_score field.

Tabs:
  Overview        · headline KPIs and plain-English findings
  Business Value  · ROI calculator (revenue uplift, payback)
  My Leads        · prioritised pipeline + filters + CSV export
  Top Lead Cards  · swipeable Tinder-style card deck
  Lead Profile    · single-lead deep-dive with what-if simulator + ARES
  Regional Map    · Czech Republic bubble map by Kraj
  Model Insights  · ROC / PR / Lift / threshold tuner
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import business_helpers as bh
import ui_splash
from contracts import API_PAYLOAD_EXCLUDE_COLS, add_region_alias, score_segment
from paths import (
    LEAD_TRAIN, METRICS_JSON, FEATURE_IMPORTANCE, VAL_PREDICTIONS, LOGO_SVG,
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE      = os.getenv("SCORING_API_URL", "http://localhost:8000")
ENEHANO_GREEN = "#a6ce39"
DANGER        = "#ff4b4b"
WARN          = "#ffc107"

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Enehano Lead Intelligence", layout="wide", page_icon="🎯")
ui_splash.maybe_show(LOGO_SVG)

# ── Glassmorphism CSS ─────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@700;800&display=swap');

    html, body, .stApp {{ font-family: 'DM Sans', sans-serif; }}

    .stApp {{
        background: radial-gradient(ellipse at 20% 0%, #0b2407 0%, #070b10 50%, #080b10 100%);
        min-height: 100vh;
        color: #f4f7fb !important;
    }}
    .stApp, .stApp * {{ color: #f4f7fb !important; }}

    /* High-contrast secondary text */
    [data-testid="stCaptionContainer"],
    [data-testid="stMarkdownContainer"] small,
    label, .stSlider label, .stMultiSelect label,
    .stSelectbox label, .stTextInput label {{
        color: #d6dde6 !important;
        opacity: 1 !important;
    }}

    /* Inputs keep dark text for readability */
    .stApp input, .stApp textarea, .stApp select,
    .stApp .stTextInput input, .stApp .stNumberInput input,
    .stApp [data-baseweb="select"] * {{ color: #111 !important; }}
    .stApp input, .stApp textarea,
    .stApp [data-baseweb="select"] > div {{
        background: #f7fafc !important;
        border-color: #d8e0ea !important;
    }}

    /* Buttons */
    .stButton > button, .stDownloadButton > button {{
        background: linear-gradient(135deg, {ENEHANO_GREEN}, #8ab52e) !important;
        color: #0a0e14 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        box-shadow: 0 4px 20px rgba(166, 206, 57, 0.25) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(166, 206, 57, 0.40) !important;
    }}
    .stButton > button *, .stDownloadButton > button * {{ color: #0a0e14 !important; }}

    /* Glassmorphism card — the main reusable surface */
    .glass-card {{
        background: rgba(12, 18, 25, 0.82);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(230, 238, 247, 0.22);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.08);
    }}

    /* Metric tiles */
    [data-testid="stMetric"] {{
        background: rgba(10, 16, 22, 0.88);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(166, 206, 57, 0.38);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }}
    [data-testid="stMetric"] * {{ color: #f4f7fb !important; }}
    [data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', sans-serif !important; font-size: 28px !important; }}
    [data-testid="stMetricDelta"] svg {{ fill: {ENEHANO_GREEN} !important; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(8, 12, 17, 0.86);
        border-radius: 14px;
        padding: 4px;
        border: 1px solid rgba(230,238,247,0.18);
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px !important;
        font-weight: 600;
        color: #d6dde6 !important;
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background: {ENEHANO_GREEN} !important;
        color: #0a0e14 !important;
        box-shadow: 0 2px 12px rgba(166,206,57,0.2);
    }}
    .stTabs [aria-selected="true"] * {{ color: #0a0e14 !important; }}

    /* Sidebar glassmorphism */
    [data-testid="stSidebar"] {{
        background: rgba(7, 10, 14, 0.96) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(230,238,247,0.18);
    }}

    /* DataFrames */
    .stDataFrame, .stDataFrame * {{ color: #f4f7fb !important; }}
    [data-testid="stDataFrame"] > div {{
        border-radius: 14px !important;
        overflow: hidden;
        border: 1px solid rgba(230,238,247,0.20) !important;
        background: rgba(8,12,17,0.90) !important;
    }}

    /* Utility classes */
    .badge {{
        display: inline-block;
        padding: 4px 13px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.06em;
        color: #111 !important;
    }}
    .hero {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 30px;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #ffffff, {ENEHANO_GREEN});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .subtle {{ font-size: 13px; opacity: 1; color: #d6dde6 !important; }}
    .finding {{
        font-size: 15px;
        padding: 12px 16px;
        border-left: 3px solid {ENEHANO_GREEN};
        background: rgba(166, 206, 57, 0.12);
        color: #f4f7fb !important;
        border-radius: 8px;
        margin: 8px 0;
        backdrop-filter: blur(8px);
    }}
    .score-ring {{
        background: rgba(8,12,17,0.86);
        backdrop-filter: blur(12px);
        border: 2px solid;
        text-align: center;
        padding: 20px;
        border-radius: 18px;
        margin-bottom: 14px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── API Client ────────────────────────────────────────────────────────────────

class APIClient:
    """Thin wrapper around the Enehano FastAPI scoring backend."""

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
    def _post(cls, path: str, payload: dict) -> dict | None:
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
    def score_single(cls, lead_dict: dict) -> dict | None:
        """POST /score — full LeadInput payload."""
        return cls._post("/score", lead_dict)

    @classmethod
    def score_batch(cls, leads: list[dict], include_drivers: bool = False) -> list[dict] | None:
        """POST /score/batch — up to 500 leads."""
        result = cls._post("/score/batch", {
            "leads": leads,
            "include_drivers": include_drivers,
        })
        return result  # list[ScoreOutput]

    @classmethod
    def score_enrich(cls, ico: str, behavioral: dict) -> dict | None:
        """POST /score/enrich — IČO + behavioral fields only."""
        return cls._post("/score/enrich", {"ico": ico, **behavioral})

    @classmethod
    def metrics(cls) -> dict | None:
        return cls._get("/metrics")


# ── Check API availability ────────────────────────────────────────────────────
_api_online = APIClient.health()
if not _api_online:
    st.error(
        "⚠️ **FastAPI backend is offline.** "
        "Start it with: `uvicorn api:app --host 0.0.0.0 --port 8000 --reload`  \n"
        "Scoring features are unavailable until the server is running."
    )


# ── Data loading (local CSV — no inference) ───────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the raw lead dataset from CSV (no model inference)."""
    df = pd.read_csv(LEAD_TRAIN, dtype={"IČO": str, "Legal_Form_Code": str})
    df["IČO"] = df["IČO"].str.zfill(8)
    df["Legal_Form_Code"] = df["Legal_Form_Code"].str.strip()
    return add_region_alias(df, copy=False)


@st.cache_data
def load_metrics() -> dict | None:
    """Load pre-computed model metrics from JSON (written by train.py)."""
    return json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else None


@st.cache_data
def load_importance() -> pd.DataFrame | None:
    """Load feature importances CSV written by train.py."""
    return pd.read_csv(FEATURE_IMPORTANCE) if FEATURE_IMPORTANCE.exists() else None


@st.cache_data
def load_val_predictions() -> pd.DataFrame | None:
    """Load held-out validation predictions written by train.py."""
    return pd.read_csv(VAL_PREDICTIONS) if VAL_PREDICTIONS.exists() else None


@st.cache_data(ttl=300, show_spinner=False)
def score_pipeline_via_api(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score the full dataset via the /score/batch endpoint.
    Results are cached for 5 minutes to avoid redundant network calls.
    Stops on API failure instead of showing fake AI values.
    """
    if not _api_online:
        st.error("FastAPI backend is offline, so AI scoring cannot be computed.")
        st.stop()

    # Build a minimal payload — only the fields the API expects as LeadInput.
    api_cols = [c for c in df.columns if c not in API_PAYLOAD_EXCLUDE_COLS]
    leads_payload = df[api_cols].fillna(0).to_dict(orient="records")

    results: list[dict] = []
    for start in range(0, len(leads_payload), 500):
        batch = leads_payload[start:start + 500]
        batch_result = APIClient.score_batch(batch, include_drivers=False)
        if not batch_result:
            st.error(
                "AI scoring failed while processing leads "
                f"{start + 1:,}-{start + len(batch):,}. "
                "Check that the FastAPI backend is running and healthy."
            )
            st.stop()
        results.extend(batch_result)
    out = df.copy()
    if results and len(results) == len(df):
        out["AI_Score"]           = [r["ai_score"]         for r in results]
        out["Segment"]            = [r["segment"]           for r in results]
        out["Expected_Win_%"]     = [r["expected_win_pct"]  for r in results]
        out["Rule_Score"]         = [r["rule_score"]        for r in results]
        out["top_drivers"]        = [r["top_drivers"]       for r in results]
        out["P_Win_If_Converted"] = [r["p_win_if_converted_pct"] for r in results]
    else:
        st.error(
            "AI scoring returned an unexpected number of results "
            f"({len(results):,} for {len(df):,} leads)."
        )
        st.stop()
    return out


# ── Load data ─────────────────────────────────────────────────────────────────
master_df     = load_data()
metrics       = load_metrics()
importance_df = load_importance()
val_df        = load_val_predictions()
scored_df     = score_pipeline_via_api(master_df)


# ── Helpers ───────────────────────────────────────────────────────────────────

def segment_label(score: float) -> tuple[str, str]:
    seg = score_segment(score)
    return seg, {"High": ENEHANO_GREEN, "Medium": WARN}.get(seg, DANGER)


# ── Header ────────────────────────────────────────────────────────────────────
hdr_l, hdr_c, hdr_r = st.columns([1, 2, 1])
with hdr_c:
    if LOGO_SVG.exists():
        st.image(str(LOGO_SVG), width="stretch")
    st.markdown(
        "<p class='hero' style='text-align:center'>Lead Intelligence</p>"
        "<p class='subtle' style='text-align:center'>Know which leads to call first — backed by data, not guesswork.</p>",
        unsafe_allow_html=True,
    )

(tab_overview, tab_value, tab_leads, tab_cards, tab_profile,
 tab_map, tab_insights) = st.tabs([
    "🏠 Overview",
    "💰 Business Value",
    "🎯 My Leads",
    "🃏 Top Lead Cards",
    "🏢 Lead Profile",
    "🗺 Regional Map",
    "🔬 Model Insights",
])


# ── TAB: Overview ─────────────────────────────────────────────────────────────
with tab_overview:
    total_leads   = len(scored_df)
    high          = (scored_df["Segment"] == "High").sum()
    expected_wins = int((scored_df["Expected_Win_%"] / 100).sum())
    avg_score     = scored_df["AI_Score"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total leads",      f"{total_leads:,}")
    k2.metric("🔥 High potential", f"{high:,}", delta=f"{high/total_leads:.0%} of all leads")
    k3.metric("Expected wins",    f"{expected_wins:,}")
    k4.metric("Avg lead quality", f"{avg_score:.1f}/100")

    st.markdown("### What the data says")

    if val_df is not None:
        lift10 = bh.lift_table(val_df["y_true"].values, val_df["ai_proba"].values, bins=10).iloc[0]
        st.markdown(
            f"<div class='finding'>📈 <b>Working only the top 10% of leads, your team would catch "
            f"{lift10['cum_recall']:.0%} of all conversions</b> — that's a "
            f"<b>{lift10['lift']:.1f}× boost</b> versus working leads in random order.</div>",
            unsafe_allow_html=True,
        )
    if metrics is not None:
        ai_auc   = metrics["conversion_model"]["test"]["auc_roc"]
        base_auc = metrics["baseline_conversion"]["auc_roc"]
        st.markdown(
            f"<div class='finding'>🧠 <b>The AI ranks leads {(ai_auc-base_auc)*100:.0f} points better "
            f"than the current rule-based scoring</b> "
            f"(AI quality {ai_auc:.0%} vs. rules {base_auc:.0%}).</div>",
            unsafe_allow_html=True,
        )
    if importance_df is not None:
        top3 = (importance_df.head(3)["feature"]
                .str.replace("__c", "").str.replace("_", " ").tolist())
        st.markdown(
            f"<div class='finding'>🔑 <b>Top 3 things that decide if a lead converts:</b> "
            f"{top3[0]}, {top3[1]}, {top3[2]}.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Lead mix")
    c1, c2 = st.columns(2)
    with c1:
        seg_counts = (scored_df["Segment"].value_counts()
                      .reindex(["High", "Medium", "Low"]).fillna(0))
        fig = px.pie(values=seg_counts.values, names=seg_counts.index, hole=0.55,
                     color=seg_counts.index,
                     color_discrete_map={"High": ENEHANO_GREEN, "Medium": WARN, "Low": DANGER},
                     title="Lead segments")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0",
                          legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(scored_df, x="AI_Score", nbins=25, color="Segment",
                           color_discrete_map={"High": ENEHANO_GREEN, "Medium": WARN, "Low": DANGER},
                           title="AI score distribution")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0e0", bargap=0.05,
                          xaxis_title="Score (0–100)", yaxis_title="Number of leads")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to read this dashboard (30-second guide)"):
        st.markdown(
            "- **Overview** (you are here): headline numbers and what they mean.\n"
            "- **Business Value**: how much money the AI can earn — input your assumptions.\n"
            "- **My Leads**: every lead, ranked. Filter, then export to CSV for Salesforce.\n"
            "- **Top Lead Cards**: flip through the hottest leads, one at a time.\n"
            "- **Lead Profile**: dive into one company. Try 'what-if' to push the score.\n"
            "- **Regional Map**: where in Czechia your best leads sit.\n"
            "- **Model Insights**: how good the AI is, and where to set the 'call' threshold.\n"
            "- **💬 button (bottom-right)**: ask a question in plain English."
        )

    # Executive PDF export
    if val_df is not None and metrics is not None:
        try:
            import exec_pdf
            ai_lift_pdf   = bh.lift_table(val_df["y_true"].values, val_df["ai_proba"].values, bins=20)
            base_lift_pdf = bh.lift_table(val_df["y_true"].values, val_df["baseline_proba"].values, bins=20)
            roi_pdf = bh.compute_roi(
                bh.ROIInputs(pipeline_size=10_000, avg_deal_value=500_000,
                             win_rate_if_converted=0.65, rep_capacity=200,
                             rep_cost=350_000, ai_top_pct=20),
                ai_lift_pdf, base_lift_pdf, float(val_df["y_true"].mean()),
            )
            top5 = scored_df.sort_values("AI_Score", ascending=False).head(5)[
                ["Company_Name", "Industry", "Region", "AI_Score", "Expected_Win_%"]
            ]
            pdf_bytes = exec_pdf.build_pdf(
                headline_uplift=roi_pdf.uplift_vs_baseline, currency="CZK",
                ai_auc=metrics["conversion_model"]["test"]["auc_roc"],
                base_auc=metrics["baseline_conversion"]["auc_roc"],
                total_leads=total_leads, high_leads=int(high), expected_wins=expected_wins,
                top_leads=top5,
                random_rev=roi_pdf.revenue_random,
                baseline_rev=roi_pdf.revenue_baseline,
                ai_rev=roi_pdf.revenue_ai,
            )
            st.download_button(
                "📄 Download executive summary (PDF)",
                data=pdf_bytes,
                file_name=f"enehano_executive_summary_{pd.Timestamp.now():%Y%m%d}.pdf",
                mime="application/pdf",
            )
        except ImportError:
            pass


# ── TAB: Business Value ───────────────────────────────────────────────────────
with tab_value:
    st.subheader("How much is AI lead scoring worth?")
    st.caption("Plug in your numbers. The dashboard estimates revenue uplift "
               "from prioritising the AI's top picks vs. random or rule-based work.")

    if val_df is None:
        st.info("Run `python src/train.py` to enable this calculator.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            pipeline_size = st.number_input("Leads per quarter",
                                            min_value=100, max_value=200_000, value=10_000, step=500)
            avg_deal = st.number_input("Average deal value (CZK)",
                                       min_value=10_000, max_value=50_000_000, value=500_000, step=10_000)
            win_rate = st.slider("Win rate once converted (%)", 10, 95, 65) / 100
        with c2:
            ai_top_pct   = st.slider("Reps work top X% of leads (AI prioritisation)", 5, 100, 20)
            rep_capacity = st.number_input("Leads each rep handles per quarter",
                                           min_value=10, max_value=2000, value=200, step=10)
            rep_cost     = st.number_input("Cost per rep per quarter (CZK)",
                                           min_value=50_000, max_value=2_000_000, value=350_000, step=10_000)
        with c3:
            st.markdown("**Currency / horizon**")
            currency   = st.selectbox("Currency label", ["CZK", "EUR", "USD"], index=0)
            horizon    = st.selectbox("Show numbers as", ["Per quarter", "Per year (×4)"], index=0)
            multiplier = 4 if horizon.startswith("Per year") else 1

        ai_lift   = bh.lift_table(val_df["y_true"].values, val_df["ai_proba"].values, bins=20)
        base_lift = bh.lift_table(val_df["y_true"].values, val_df["baseline_proba"].values, bins=20)
        base_rate = float(val_df["y_true"].mean())

        r = bh.compute_roi(
            bh.ROIInputs(pipeline_size=pipeline_size, avg_deal_value=avg_deal,
                         win_rate_if_converted=win_rate, rep_capacity=rep_capacity,
                         rep_cost=rep_cost, ai_top_pct=ai_top_pct),
            ai_lift, base_lift, base_rate,
        )

        def _fmt(x: float) -> str:
            return f"{x*multiplier/1_000_000:,.2f}M {currency}"

        st.markdown("### Expected revenue under each strategy")
        m1, m2, m3 = st.columns(3)
        m1.metric("🎲 Random order",      _fmt(r.revenue_random))
        m2.metric("📋 Rule-based scoring", _fmt(r.revenue_baseline))
        m3.metric("🧠 AI scoring",         _fmt(r.revenue_ai),
                  delta=f"+{_fmt(r.uplift_vs_baseline)} vs. rules")

        u1, u2, u3 = st.columns(3)
        u1.metric("Leads worked",    f"{r.leads_worked:,} / {pipeline_size:,}")
        u2.metric("Reps required",   f"{r.reps_needed}")
        u3.metric("Sales-team cost", _fmt(r.reps_cost))

        st.markdown(
            f"<div class='finding'>💡 At <b>top {ai_top_pct}% prioritisation</b>, AI is expected to generate "
            f"<b>{_fmt(r.uplift_vs_baseline)}</b> more revenue than the current rule-based approach"
            f" — that's a <b>{(r.uplift_vs_baseline/max(1,r.revenue_baseline))*100:.0f}% lift</b>.</div>",
            unsafe_allow_html=True,
        )

        bar_df = pd.DataFrame({
            "Strategy": ["Random", "Rule-based", "AI"],
            "Revenue":  [r.revenue_random*multiplier, r.revenue_baseline*multiplier, r.revenue_ai*multiplier],
        })
        fig = px.bar(bar_df, x="Strategy", y="Revenue", color="Strategy",
                     color_discrete_map={"Random": DANGER, "Rule-based": WARN, "AI": ENEHANO_GREEN},
                     title=f"Revenue comparison ({horizon.lower()}, {currency})")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0e0", showlegend=False, yaxis_title=currency)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Assumptions used in this calculator"):
            st.markdown(
                f"- Base conversion rate observed in data: **{base_rate:.1%}**\n"
                f"- Win-once-converted: **{win_rate:.0%}** (your input)\n"
                f"- Effective deal value (deal × win): **{avg_deal*win_rate:,.0f} {currency}**\n"
                "- 'Random' assumes reps pick leads in arbitrary order.\n"
                "- 'Rule-based' uses the current heuristic scoring (`baseline.py`).\n"
                "- 'AI' uses the trained model and the top-X% prioritisation slider above."
            )


# ── TAB: My Leads ─────────────────────────────────────────────────────────────
with tab_leads:
    st.subheader("Your prioritised leads")
    st.caption("Leads ranked by AI win likelihood. Filter, then export to CSV for Salesforce.")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        seg_f = st.multiselect("Show segments", ["High", "Medium", "Low"], default=["High", "Medium"])
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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Leads in view",    f"{len(view):,}")
    k2.metric("🔥 High potential", f"{(view['Segment']=='High').sum():,}")
    k3.metric("Avg AI score",     f"{view['AI_Score'].mean():.1f}" if len(view) else "—")
    k4.metric("Expected wins",    f"{int((view['Expected_Win_%']/100).sum()):,}")

    display_cols = ["IČO", "Company_Name", "Industry", "Region", "LeadSource",
                    "AI_Score", "Segment", "Rule_Score", "Expected_Win_%"]
    selection = st.dataframe(
        view[display_cols], use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Company_Name":   st.column_config.TextColumn("Company"),
            "LeadSource":     st.column_config.TextColumn("Source"),
            "AI_Score":       st.column_config.ProgressColumn("AI Score",   min_value=0, max_value=100, format="%.1f"),
            "Rule_Score":     st.column_config.ProgressColumn("Rule Score", min_value=0, max_value=100, format="%d"),
            "Expected_Win_%": st.column_config.NumberColumn("Expected Win %", format="%.1f%%"),
        },
    )

    if selection.selection.rows:
        st.session_state["selected_ico"] = view.iloc[selection.selection.rows[0]]["IČO"]
        st.toast("Open the 🏢 Lead Profile tab to inspect the selected company.", icon="➡")

    csv_bytes = view[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Export filtered leads (CSV for Salesforce)",
                       csv_bytes, file_name="enehano_scored_leads.csv", mime="text/csv")


# ── TAB: Top Lead Cards ───────────────────────────────────────────────────────
with tab_cards:
    import ui_cards
    ui_cards.render(scored_df, importance_df, top_n=20)


# ── TAB: Lead Profile ─────────────────────────────────────────────────────────
with tab_profile:
    raw_input = st.text_input(
        "Enter IČO (8 digits) or pick a lead in the 🎯 My Leads tab",
        value=st.session_state.get("selected_ico", ""),
        max_chars=8,
    ).strip()
    ico_input = raw_input.zfill(8) if raw_input else ""
    matches   = scored_df[scored_df["IČO"] == ico_input] if len(ico_input) == 8 else pd.DataFrame()

    if matches.empty:
        st.info("Enter a valid 8-digit IČO from the dataset (e.g. select a row in the 🎯 My Leads tab).")
    else:
        row = matches.iloc[0]
        st.title(f"🏢 {row['Company_Name']}")
        st.caption(
            f"IČO {row['IČO']} · {row.get('Legal_Form_Label', '—')} · {row['Industry']} · {row['Region']}"
        )

        # ── What-if Simulator (sidebar) ───────────────────────────────────
        with st.sidebar:
            st.markdown("### 🧪 What-if simulator")
            st.write("Drag the sliders to see how lead behaviour changes the score.")
            sim_ttfr  = st.slider("Time to first response (h)", 0.1, 72.0,
                                   float(row["Time_to_First_Response_h__c"]), 0.5)
            sim_web   = st.slider("Web interactions", 0, 100, int(row["Web_Interactions__c"]))
            sim_meet  = st.slider("Meetings held",    0, 10,  int(row["Meetings_Held__c"]))
            sim_forms = st.slider("Form submissions", 0, 10,  int(row["Form_Submissions__c"]))
            sim_demo  = st.checkbox("Demo requested",  bool(row["Demo_Requested__c"]))
            sim_li    = st.checkbox("LinkedIn viewed", bool(row["LinkedIn_Viewed__c"]))

        # Build the what-if payload — use /score/enrich so we don't need to
        # repeat all firmographic fields (ARES fills those on the backend).
        behavioral_override = {
            "Time_to_First_Response_h__c": sim_ttfr,
            "Web_Interactions__c":          sim_web,
            "Meetings_Held__c":             sim_meet,
            "Form_Submissions__c":          sim_forms,
            "Demo_Requested__c":            int(sim_demo),
            "LinkedIn_Viewed__c":           int(sim_li),
            # Preserve remaining behavioral fields from the original row
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
        }

        original_score = float(row["AI_Score"])

        if _api_online:
            sim_result = APIClient.score_enrich(str(row["IČO"]), behavioral_override)
        else:
            sim_result = None

        if sim_result:
            new_score = float(sim_result["ai_score"])
            seg, color = segment_label(new_score)
            expected_win = float(sim_result["expected_win_pct"])
            rule_score   = sim_result["rule_score"]
            api_drivers  = sim_result.get("top_drivers", [])
        else:
            # Graceful fallback when API is offline
            new_score    = original_score
            seg, color   = segment_label(new_score)
            expected_win = float(row.get("Expected_Win_%", 0))
            rule_score   = int(row.get("Rule_Score", 0))
            api_drivers  = list(row.get("top_drivers", []) or [])

        col_score, col_expl = st.columns([1, 1.4], gap="large")

        with col_score:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            diff      = new_score - original_score
            diff_html = (
                f"<span style='font-size:14px;opacity:.8'>was {original_score:.1f}% ({diff:+.1f})</span>"
            ) if abs(diff) >= 0.05 else ""
            st.markdown(
                f"<div class='score-ring' style='border-color:{color};'>"
                f"AI CONVERSION SCORE<br>"
                f"<span style='font-size:52px;font-family:Space Grotesk,sans-serif;"
                f"font-weight:800;color:{color} !important'>{new_score:.1f}%</span><br>"
                f"<span class='badge' style='background:{color};color:#111 !important;margin-top:8px'>"
                f"{seg.upper()} POTENTIAL</span><br>{diff_html}</div>",
                unsafe_allow_html=True,
            )
            st.caption(bh.explain_segment(seg))
            st.metric("Expected win % (Convert × Win)", f"{expected_win:.1f}%")
            st.metric("Rule-based score (baseline)",    f"{rule_score} / 100")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_expl:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Why this score?")

            if api_drivers:
                st.caption("Top drivers from the API (SHAP explanations):")
                for driver in api_drivers:
                    st.markdown(
                        f"<div style='background:rgba(166,206,57,0.08);"
                        f"border-left:3px solid {ENEHANO_GREEN};border-radius:6px;"
                        f"padding:8px 12px;margin:6px 0;font-size:14px'>"
                        f"✅ <b>{driver}</b></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Start the API server to see SHAP-powered driver explanations.")

            st.markdown('</div>', unsafe_allow_html=True)

        # ARES live enrichment
        with st.expander("🔎 Live data from ARES (Czech business register)"):
            if st.button("Fetch ARES data"):
                with st.spinner("Calling ARES API..."):
                    # Route ARES calls through the FastAPI backend to avoid
                    # direct external calls from the frontend process.
                    ares_result = APIClient.score_enrich(str(row["IČO"]), behavioral_override)
                if ares_result is None:
                    st.warning("ARES lookup failed or company not found.")
                else:
                    st.success("ARES data retrieved via backend.")
                    st.json(ares_result)


# ── TAB: Regional Map ─────────────────────────────────────────────────────────
with tab_map:
    st.subheader("Where are your best leads?")
    st.caption("AI score and lead volume by Czech region (Kraj).")

    KRAJ_COORDS = {
        "Praha":              (50.075, 14.437),
        "Středočeský":        (49.876, 14.789),
        "Jihočeský":          (49.124, 14.500),
        "Plzeňský":           (49.747, 13.378),
        "Karlovarský":        (50.232, 12.871),
        "Ústecký":            (50.661, 13.832),
        "Liberecký":          (50.770, 15.058),
        "Královéhradecký":    (50.342, 15.792),
        "Pardubický":         (49.945, 16.275),
        "Kraj Vysočina":      (49.397, 15.589),
        "Jihomoravský":       (49.195, 16.608),
        "Olomoucký":          (49.594, 17.250),
        "Zlínský":            (49.224, 17.666),
        "Moravskoslezský":    (49.823, 18.262),
    }

    region_stats = (
        scored_df.groupby("Region")
        .agg(
            leads=("Lead_ID", "count"),
            avg_score=("AI_Score", "mean"),
            high_potential=("Segment", lambda s: (s == "High").sum()),
            expected_wins=("Expected_Win_%", lambda s: (s / 100).sum()),
        )
        .reset_index()
    )
    region_stats["avg_score"]     = region_stats["avg_score"].round(1)
    region_stats["expected_wins"] = region_stats["expected_wins"].round(0).astype(int)

    def _match_coord(region_name: str) -> tuple:
        r = region_name.strip()
        if r in KRAJ_COORDS:
            return KRAJ_COORDS[r]
        stripped = r.replace(" kraj", "").replace("Kraj ", "").strip()
        for k, v in KRAJ_COORDS.items():
            if stripped.lower() in k.lower() or k.lower() in stripped.lower():
                return v
        return (None, None)

    region_stats["lat"] = region_stats["Region"].map(lambda r: _match_coord(r)[0])
    region_stats["lon"] = region_stats["Region"].map(lambda r: _match_coord(r)[1])
    mapped = region_stats.dropna(subset=["lat", "lon"])

    if len(mapped):
        fig = px.scatter_geo(
            mapped, lat="lat", lon="lon", size="leads", color="avg_score",
            hover_name="Region",
            hover_data={"leads": True, "avg_score": True, "high_potential": True,
                        "expected_wins": True, "lat": False, "lon": False},
            color_continuous_scale=[[0, DANGER], [0.5, WARN], [1, ENEHANO_GREEN]],
            size_max=55, projection="mercator",
        )
        fig.update_geos(visible=False, resolution=50,
                        showcountries=True, countrycolor="#444",
                        showland=True, landcolor="#1a1f25",
                        lataxis_range=[48.5, 51.2], lonaxis_range=[12.0, 19.0])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0",
                          margin=dict(l=0, r=0, t=10, b=0), height=520,
                          coloraxis_colorbar=dict(title="Avg score"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No region coordinates matched. Check Region values match KRAJ_COORDS keys.")

    c1, c2 = st.columns(2)
    with c1:
        top_regions = region_stats.sort_values("avg_score", ascending=False).head(10)
        fig = px.bar(top_regions, x="avg_score", y="Region", orientation="h",
                     color="avg_score",
                     color_continuous_scale=[[0, DANGER], [0.5, WARN], [1, ENEHANO_GREEN]],
                     title="Top regions by average AI score")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0e0", yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(region_stats.sort_values("expected_wins", ascending=False).head(10),
                     x="expected_wins", y="Region", orientation="h",
                     title="Top regions by expected wins",
                     color_discrete_sequence=[ENEHANO_GREEN])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0e0", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        region_stats[["Region", "leads", "high_potential", "avg_score", "expected_wins"]]
        .sort_values("expected_wins", ascending=False),
        use_container_width=True, hide_index=True,
        column_config={
            "leads":          st.column_config.NumberColumn("Leads",          format="%d"),
            "high_potential": st.column_config.NumberColumn("High potential", format="%d"),
            "avg_score":      st.column_config.NumberColumn("Avg AI score",   format="%.1f"),
            "expected_wins":  st.column_config.NumberColumn("Expected wins",  format="%d"),
        },
    )


# ── TAB: Model Insights ───────────────────────────────────────────────────────
with tab_insights:
    st.subheader("How good is the AI?")
    if val_df is None or metrics is None:
        st.info("Run `python src/train.py` to enable this tab.")
    else:
        y      = val_df["y_true"].values
        p_ai   = val_df["ai_proba"].values
        p_base = val_df["baseline_proba"].values

        ai_auc   = bh.auc(y, p_ai)
        base_auc = bh.auc(y, p_base)
        k1, k2, k3 = st.columns(3)
        k1.metric("AI ranking quality",        f"{ai_auc:.0%}",
                  delta=f"+{(ai_auc-base_auc)*100:.0f} pts vs. rules")
        k2.metric("Rule-based ranking quality", f"{base_auc:.0%}")
        cv_std = metrics["conversion_model"]["cv"]["auc_std"]
        stable = "✅ Stable" if cv_std < 0.01 else "⚠ Variable"
        k3.metric("Result stability (5 retrains)", stable,
                  delta=f"±{cv_std:.1%} fluctuation", delta_color="off")

        st.divider()
        st.markdown("#### 🎚 Pick the minimum score worth calling")
        thr = st.slider("Minimum AI score (0–100)", 5, 95, 50, 5)
        ts  = bh.threshold_stats(y, p_ai * 100, thr)

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Leads to call",     f"{ts.leads_to_work:,}",
                  delta=f"{ts.leads_to_work_pct:.0%} of all leads", delta_color="off")
        t2.metric("Conversions caught", f"{ts.captured_conversions:,}",
                  delta=f"{ts.captured_pct:.0%} of all wins",       delta_color="off")
        t3.metric("Hit rate",          f"{ts.precision:.0%}")
        t4.metric("Coverage",          f"{ts.recall:.0%}")

        st.markdown(
            f"<div class='finding'>👉 At a minimum score of <b>{thr}</b>, your team works "
            f"<b>{ts.leads_to_work_pct:.0%}</b> of leads and still catches "
            f"<b>{ts.captured_pct:.0%}</b> of all conversions.</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("#### 📈 Effort vs. results")
        lift_ai   = bh.lift_table(y, p_ai,   bins=20)
        lift_base = bh.lift_table(y, p_base, bins=20)
        gains = pd.concat([
            lift_ai.assign(Strategy="AI")[["top_pct", "cum_recall", "Strategy"]],
            lift_base.assign(Strategy="Rule-based")[["top_pct", "cum_recall", "Strategy"]],
            pd.DataFrame({"top_pct": [0, 100], "cum_recall": [0, 1.0], "Strategy": ["Random", "Random"]}),
        ])
        fig = px.line(gains, x="top_pct", y="cum_recall", color="Strategy", markers=True,
                      color_discrete_map={"AI": ENEHANO_GREEN, "Rule-based": WARN, "Random": DANGER},
                      labels={"top_pct": "% of leads worked (best first)",
                              "cum_recall": "% of conversions captured"},
                      title="Cumulative gain — the higher and earlier, the better")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0e0", yaxis_tickformat=".0%", xaxis_ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### 🛰 Is the AI still on target? (stability check)")
        rng  = np.random.default_rng(42)
        idx  = rng.permutation(len(p_ai))
        half = len(idx) // 2
        ref_scores = p_ai[idx[:half]] * 100
        cur_scores = p_ai[idx[half:]] * 100

        psi              = bh.population_stability_index(ref_scores, cur_scores, bins=10)
        psi_label, psi_color = bh.psi_verdict(psi)

        d1, d2, d3 = st.columns([1, 1, 2])
        d1.metric("Stability Index", f"{psi:.3f}")
        d2.markdown(
            f"<div style='background:{psi_color}22;border:2px solid {psi_color};"
            f"padding:14px;border-radius:10px;text-align:center;font-weight:700;"
            f"color:{psi_color} !important;font-size:18px;margin-top:6px'>{psi_label}</div>",
            unsafe_allow_html=True,
        )
        d3.caption(
            "**Past leads (training):** scores from the data the AI learned on.  \n"
            "**Recent leads:** scores on the newest batch of leads.  \n"
            "*(In this demo both come from the same held-out sample.)*"
        )

        drift_df = pd.concat([
            pd.DataFrame({"Score": ref_scores, "Group": "Past leads (training)"}),
            pd.DataFrame({"Score": cur_scores, "Group": "Recent leads"}),
        ])
        fig = px.histogram(drift_df, x="Score", color="Group", barmode="overlay",
                           nbins=20, opacity=0.65,
                           color_discrete_map={"Past leads (training)": ENEHANO_GREEN,
                                               "Recent leads": "#3498db"},
                           title="Score distribution: past vs. recent leads")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0e0", bargap=0.05,
                          xaxis_title="AI score (0–100)", yaxis_title="Number of leads",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 Technical curves (ROC & Precision-Recall)"):
            roc_a, roc_b = bh.roc_points(y, p_ai), bh.roc_points(y, p_base)
            pr_a,  pr_b  = bh.pr_points(y, p_ai),  bh.pr_points(y, p_base)
            cc1, cc2 = st.columns(2)
            with cc1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=roc_a["fpr"], y=roc_a["tpr"],
                                         name=f"AI (AUC={ai_auc:.3f})",
                                         line=dict(color=ENEHANO_GREEN, width=3)))
                fig.add_trace(go.Scatter(x=roc_b["fpr"], y=roc_b["tpr"],
                                         name=f"Rules (AUC={base_auc:.3f})",
                                         line=dict(color=WARN, width=2)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                         line=dict(color=DANGER, dash="dash", width=1)))
                fig.update_layout(title="ROC curve",
                                  xaxis_title="False positive rate", yaxis_title="True positive rate",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e0e0e0", legend=dict(x=0.4, y=0.1))
                st.plotly_chart(fig, use_container_width=True)
            with cc2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pr_a["recall"], y=pr_a["precision"],
                                         name="AI", line=dict(color=ENEHANO_GREEN, width=3)))
                fig.add_trace(go.Scatter(x=pr_b["recall"], y=pr_b["precision"],
                                         name="Rules", line=dict(color=WARN, width=2)))
                fig.update_layout(title="Precision-Recall curve",
                                  xaxis_title="Recall", yaxis_title="Precision",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e0e0e0", legend=dict(x=0.05, y=0.1))
                st.plotly_chart(fig, use_container_width=True)

        if importance_df is not None:
            st.markdown("#### 🔑 What drives the AI's decisions")
            top20 = importance_df.head(20).copy()
            top20["feature"] = top20["feature"].str.replace("__c", "").str.replace("_", " ")
            fig = px.bar(top20.sort_values("importance"), x="importance", y="feature",
                         orientation="h", color="importance",
                         color_continuous_scale=[[0, "#444"], [1, ENEHANO_GREEN]],
                         title="Top 20 features (Gradient Boosting importance)")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e0e0e0", coloraxis_showscale=False, height=520,
                              yaxis_title="", xaxis_title="Importance")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("⚙ For data scientists — raw metrics"):
            conv_t = metrics["conversion_model"]["test"]
            base_c = metrics["baseline_conversion"]
            comp   = pd.DataFrame({
                "Metric":     ["AUC-ROC", "PR-AUC", "Precision", "Recall", "F1"],
                "Rule-based": [base_c["auc_roc"], base_c["pr_auc"],
                               base_c["precision"], base_c["recall"], base_c["f1"]],
                "AI":         [conv_t["auc_roc"], conv_t["pr_auc"],
                               conv_t["precision"], conv_t["recall"], conv_t["f1"]],
            })
            comp["Uplift"] = comp["AI"] - comp["Rule-based"]
            st.dataframe(
                comp.style.format({"Rule-based": "{:.3f}", "AI": "{:.3f}", "Uplift": "{:+.3f}"}),
                use_container_width=True, hide_index=True,
            )
            cv = metrics["conversion_model"]["cv"]
            st.write(f"**CV AUC (5-fold):** {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}")
            st.write(f"**Win model test AUC:** {metrics['win_model']['test']['auc_roc']:.4f}")
            st.write(
                f"**Dataset:** {metrics['n_rows']:,} leads, "
                f"{metrics['n_converted']:,} converted, {metrics['n_won']:,} won."
            )


st.divider()
st.caption("Enehano Solutions · Lead Intelligence · Predictive AI MVP")

# Floating chat bubble — always visible, on every tab
import ui_chat
ui_chat.render_bubble(scored_df)
