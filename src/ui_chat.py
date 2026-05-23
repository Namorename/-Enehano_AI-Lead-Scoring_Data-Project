"""
ui_chat.py
==========
'Ask the data' floating chat bubble — API-aware edition.

Query execution remains rule-based (nl_query.py) and operates on the
locally scored DataFrame for sub-millisecond response times. The backend
API is used for:
  - /metrics endpoint to enrich the chat with live model performance figures.
  - Health-check to gracefully degrade when the API is offline.

Column contract:
  - Uses 'Region'    (aliased in load_data(), matches nl_query DIMENSIONS key)
  - Uses 'Converted' for win_rate metric (present in scored_df)
  - 'Closed_Won'     used only as fallback if 'Converted' is missing
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

import nl_query

ENEHANO_GREEN = "#a6ce39"
WARN          = "#ffc107"
DANGER        = "#ff4b4b"
_HISTORY_KEY  = "_chat_history"
_API_BASE     = os.getenv("SCORING_API_URL", "http://localhost:8000")
_API_TIMEOUT  = 4   # seconds — keeps the chat bubble snappy


# ── Backend helpers ───────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def _fetch_api_metrics() -> dict | None:
    """
    Fetch model performance metrics from the FastAPI /metrics endpoint.
    Cached for 2 minutes to avoid hammering the backend on every rerun.
    Returns None when the server is unreachable.
    """
    try:
        r = requests.get(f"{_API_BASE}/metrics", timeout=_API_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


# ── Query helpers ─────────────────────────────────────────────────────────────

def _apply_filters(df: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    """Apply key=value equality filters derived from the parsed nl_query Intent."""
    out = df
    for col, val in filters.items():
        if col in out.columns:
            out = out[out[col].astype(str) == str(val)]
    return out


def _aggregate(df: pd.DataFrame, metric: str, dimension: str | None) -> pd.DataFrame:
    """Compute the requested metric optionally grouped by `dimension`."""
    if dimension is None:
        df = df.assign(_all="All leads")
        dimension = "_all"

    if metric == "win_rate":
        col = "Converted" if "Converted" in df.columns else "Closed_Won"
        agg = df.groupby(dimension)[col].mean().mul(100).round(1)
        return agg.reset_index().rename(columns={col: "Win rate %"})

    if metric == "ai_score":
        agg = df.groupby(dimension)["AI_Score"].mean().round(1)
        return agg.reset_index().rename(columns={"AI_Score": "Avg AI score"})

    if metric == "expected_wins":
        agg = df.groupby(dimension)["Expected_Win_%"].apply(
            lambda s: int((s / 100).sum())
        )
        return agg.reset_index().rename(columns={"Expected_Win_%": "Expected wins"})

    if metric == "high_pct":
        agg = df.groupby(dimension)["Segment"].apply(
            lambda s: round((s == "High").mean() * 100, 1)
        )
        return agg.reset_index().rename(columns={"Segment": "% high potential"})

    # Default: lead count
    agg = df.groupby(dimension).size()
    return agg.reset_index(name="Leads")


# ── Full-size answer renderers ────────────────────────────────────────────────

def _render_answer(df: pd.DataFrame, intent: nl_query.Intent) -> None:
    """Render a rich answer (charts + table) inside the full-width dialog."""

    if intent.shortcut == "top_leads":
        top = df.sort_values("AI_Score", ascending=False).head(intent.top_n)
        st.markdown(f"**Here are the top {len(top)} leads by AI score:**")
        st.dataframe(
            top[["Company_Name", "Industry", "Region", "AI_Score", "Expected_Win_%"]],
            hide_index=True, use_container_width=True,
        )
        return

    if intent.shortcut in {"best_region", "best_industry", "best_source"}:
        col    = {"best_region": "Region", "best_industry": "Industry",
                  "best_source": "LeadSource"}[intent.shortcut]
        result = _aggregate(df, "win_rate", col).sort_values("Win rate %", ascending=False)
        if result.empty:
            st.warning("No data.")
            return
        winner = result.iloc[0]
        st.markdown(
            f"<div style='background:rgba(166,206,57,0.08);"
            f"border-left:3px solid {ENEHANO_GREEN};border-radius:10px;"
            f"padding:14px 18px;font-size:16px;backdrop-filter:blur(8px)'>"
            f"🏆 <b>{winner[col]}</b> leads with a "
            f"<b>{winner.iloc[1]:.1f}%</b> win rate.</div>",
            unsafe_allow_html=True,
        )
        fig = px.bar(
            result.head(10), x=col, y="Win rate %", color="Win rate %",
            color_continuous_scale=[[0, "#333"], [1, ENEHANO_GREEN]],
            title=f"Win rate by {col.lower()}",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0", coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    result = _aggregate(df, intent.metric, intent.dimension)
    if result.empty:
        st.warning("No leads match those filters.")
        return

    metric_col = result.columns[1]
    if intent.dimension is None or result.iloc[0, 0] == "All leads":
        val = result.iloc[0, 1]
        st.markdown(
            f"<div style='background:rgba(166,206,57,0.08);"
            f"border-left:3px solid {ENEHANO_GREEN};border-radius:10px;"
            f"padding:16px 20px;font-size:18px;backdrop-filter:blur(8px)'>"
            f"{metric_col}: <b style='color:{ENEHANO_GREEN}'>{val}</b></div>",
            unsafe_allow_html=True,
        )
        return

    sorted_result = result.sort_values(metric_col, ascending=False).head(15)
    fig = px.bar(
        sorted_result, x=intent.dimension, y=metric_col, color=metric_col,
        color_continuous_scale=[[0, "#333"], [1, ENEHANO_GREEN]],
        title=f"{metric_col} by {intent.dimension.lower()}",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0", coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Show data"):
        st.dataframe(sorted_result, hide_index=True, use_container_width=True)


# ── Compact answer renderers (inline bubble panel) ────────────────────────────

def _render_compact_answer(df: pd.DataFrame, intent: nl_query.Intent) -> None:
    """Render a compact one-line / mini-table answer for the chat history view."""

    if intent.shortcut == "top_leads":
        top  = df.sort_values("AI_Score", ascending=False).head(min(intent.top_n, 5))
        rows = "".join(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:3px 0;font-size:12px'>"
            f"<span style='color:#f4f7fb'>{r.Company_Name[:26]}</span>"
            f"<b style='color:{ENEHANO_GREEN}'>{float(r.AI_Score):.1f}</b></div>"
            for r in top.itertuples()
        )
        st.markdown(
            f"<div style='padding:8px 12px;font-size:12px'>"
            f"🤖 Top {len(top)} leads by AI score:{rows}</div>",
            unsafe_allow_html=True,
        )
        return

    if intent.shortcut in {"best_region", "best_industry", "best_source"}:
        col    = {"best_region": "Region", "best_industry": "Industry",
                  "best_source": "LeadSource"}[intent.shortcut]
        result = _aggregate(df, "win_rate", col).sort_values("Win rate %", ascending=False)
        if result.empty:
            st.markdown(
                "<div style='padding:6px 10px;font-size:12px'>No data.</div>",
                unsafe_allow_html=True,
            )
            return
        winner = result.iloc[0]
        st.markdown(
            f"<div style='padding:8px 12px;font-size:13px'>"
            f"🤖 🏆 <b>{winner[col]}</b> leads with "
            f"<b style='color:{ENEHANO_GREEN}'>{winner['Win rate %']:.1f}%</b> win rate.</div>",
            unsafe_allow_html=True,
        )
        return

    result = _aggregate(df, intent.metric, intent.dimension)
    if result.empty:
        st.markdown(
            "<div style='padding:8px 12px;font-size:12px'>"
            "🤖 No leads match those filters.</div>",
            unsafe_allow_html=True,
        )
        return

    metric_col = result.columns[1]
    if intent.dimension is None or result.iloc[0, 0] == "All leads":
        st.markdown(
            f"<div style='padding:8px 12px;font-size:13px'>"
            f"🤖 {metric_col}: <b style='color:{ENEHANO_GREEN}'>"
            f"{result.iloc[0, 1]}</b></div>",
            unsafe_allow_html=True,
        )
        return

    top  = result.sort_values(metric_col, ascending=False).head(5)
    rows = "".join(
        f"<div style='display:flex;justify-content:space-between;"
        f"padding:3px 0;font-size:12px'>"
        f"<span style='color:#f4f7fb'>{str(r[0])[:26]}</span>"
        f"<b style='color:{ENEHANO_GREEN}'>{r[1]}</b></div>"
        for r in top.itertuples(index=False)
    )
    st.markdown(
        f"<div style='padding:8px 12px;font-size:12px'>"
        f"🤖 {metric_col} (top 5):{rows}</div>",
        unsafe_allow_html=True,
    )


# ── Floating bubble injector ──────────────────────────────────────────────────

_BUBBLE_INJECTOR = f"""
<script>
(function() {{
    const doc = window.parent.document;

    // ── Inject glassmorphism bubble styles once ──
    if (!doc.getElementById('enehano-bubble-style')) {{
        const style = doc.createElement('style');
        style.id = 'enehano-bubble-style';
        style.textContent = `
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@700&display=swap');

            .enehano-bubble {{
                position: fixed;
                right: 28px;
                bottom: 28px;
                z-index: 99999;
                width: 64px;
                height: 64px;
                border-radius: 50%;

                /* Glassmorphism button surface */
                background: linear-gradient(135deg,
                    rgba(166, 206, 57, 0.85),
                    rgba(120, 160, 25, 0.90));
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.25);

                color: #111;
                font-size: 26px;
                font-weight: 700;
                cursor: pointer;

                box-shadow:
                    0 12px 32px rgba(0, 0, 0, 0.50),
                    0 0 0 5px rgba(166, 206, 57, 0.18),
                    inset 0 1px 0 rgba(255,255,255,0.30);

                transition: transform 180ms ease, box-shadow 180ms ease;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            .enehano-bubble:hover {{
                transform: scale(1.10) translateY(-2px);
                box-shadow:
                    0 18px 44px rgba(0, 0, 0, 0.60),
                    0 0 0 7px rgba(166, 206, 57, 0.25),
                    inset 0 1px 0 rgba(255,255,255,0.30);
            }}

            /* Hide the Streamlit trigger button visually */
            .st-key-_open_chat_btn {{
                position: absolute !important;
                width: 1px !important;
                height: 1px !important;
                overflow: hidden !important;
                clip: rect(0 0 0 0) !important;
                margin: -1px !important;
                padding: 0 !important;
                border: 0 !important;
            }}
        `;
        doc.head.appendChild(style);
    }}

    // ── Create or re-use the bubble button ──
    let btn = doc.getElementById('enehano-chat-bubble');
    if (!btn) {{
        btn = doc.createElement('button');
        btn.id        = 'enehano-chat-bubble';
        btn.className = 'enehano-bubble';
        btn.title     = 'Ask the data';
        btn.type      = 'button';
        btn.textContent = '💬';
        doc.body.appendChild(btn);
    }}

    btn.onclick = function() {{
        const wrap   = doc.querySelector('.st-key-_open_chat_btn');
        if (!wrap) {{ console.warn('chat trigger button not found'); return; }}
        const target = wrap.querySelector('button');
        if (target) target.click();
    }};
}})();
</script>
"""


# ── State helpers ─────────────────────────────────────────────────────────────

def _flush_pending() -> None:
    """Move any pending message from session state into chat history."""
    pending = st.session_state.pop("_chat_pending", None)
    if pending:
        st.session_state.setdefault(_HISTORY_KEY, []).append({"q": pending})


def _on_text_submit() -> None:
    """Callback: transfer the text input value into the pending queue."""
    text = st.session_state.get("_chat_text_input", "").strip()
    if text:
        st.session_state["_chat_pending"]     = text
        st.session_state["_chat_dialog_open"] = True
        st.session_state["_chat_text_input"]  = ""


# ── Dialog ────────────────────────────────────────────────────────────────────

@st.dialog("💬 Ask the data", width="large")
def _chat_dialog(scored_df: pd.DataFrame) -> None:
    """
    Full-screen dialog with chat history, text input, and suggestion chips.
    Model performance figures from the API /metrics endpoint are woven into
    special answers when available.
    """
    _flush_pending()

    # Optional: surface live API metric if available
    api_metrics = _fetch_api_metrics()
    if api_metrics:
        conv_auc  = api_metrics.get("conversion_model", {}).get("test", {}).get("auc_roc")
        base_auc  = api_metrics.get("baseline_conversion", {}).get("auc_roc")
        if conv_auc and base_auc:
            st.caption(
                f"🔗 API connected · Model AUC {conv_auc:.0%} "
                f"(+{(conv_auc - base_auc)*100:.0f} pts vs. rules)"
            )
    else:
        st.caption(
            "Plain English. Real numbers from your pipeline. "
            "Click a suggestion or type your own question."
        )

    history = st.session_state.get(_HISTORY_KEY, [])

    # ── Chat scroll area ──
    with st.container(border=True, height=340):
        if not history:
            st.markdown(
                "<div style='padding:24px;text-align:center;color:#d6dde6'>"
                "👋 Hi! I'm your data assistant.<br>"
                "Try one of the suggestions below to get started.</div>",
                unsafe_allow_html=True,
            )

        for entry in history[-10:]:
            # User bubble
            st.markdown(
                f"<div style='background:rgba(166,206,57,0.10);"
                f"border-left:3px solid {ENEHANO_GREEN};"
                f"border-radius:10px;padding:10px 14px;margin:8px 0;"
                f"font-size:14px;backdrop-filter:blur(8px)'>"
                f"<b>You:</b> {entry['q']}</div>",
                unsafe_allow_html=True,
            )

            # AI response
            intent = nl_query.parse(entry["q"], scored_df)
            if intent is None:
                st.markdown(
                    "<div style='background:rgba(8,12,17,0.84);"
                    "border:1px solid rgba(230,238,247,0.18);"
                    "border-radius:10px;padding:10px 14px;margin:8px 0;"
                    "font-size:14px;backdrop-filter:blur(6px)'>"
                    "🤖 Sorry, I couldn't parse that. Try a suggestion below.</div>",
                    unsafe_allow_html=True,
                )
            else:
                filtered = _apply_filters(scored_df, intent.filters)
                _render_compact_answer(filtered, intent)

    # ── Text input ──
    st.text_input(
        "Your question",
        key="_chat_text_input",
        placeholder="e.g. Win rate by industry in Praha",
        label_visibility="collapsed",
        on_change=_on_text_submit,
    )

    # ── Suggestion chips ──
    st.markdown(
        "<div style='font-size:11px;color:#d6dde6;letter-spacing:0.10em;"
        "text-transform:uppercase;margin:10px 0 6px'>Try one of these</div>",
        unsafe_allow_html=True,
    )
    sugg = nl_query.suggestions(scored_df)[:6]
    cols = st.columns(2)
    for i, s in enumerate(sugg):
        if cols[i % 2].button(s, key=f"dlg_sugg_{i}", use_container_width=True):
            st.session_state["_chat_pending"]     = s
            st.session_state["_chat_dialog_open"] = True
            st.rerun()

    # ── Footer ──
    foot_l, foot_r = st.columns([3, 1])
    with foot_l:
        st.caption(
            "Answers computed live from your pipeline data."
            + (" · API metrics live." if api_metrics else " · API offline.")
        )
    with foot_r:
        if history and st.button("🗑 Clear", key="dlg_clear", use_container_width=True):
            st.session_state[_HISTORY_KEY]        = []
            st.session_state["_chat_dialog_open"] = True
            st.rerun()


# ── Public entry point ────────────────────────────────────────────────────────

def render_bubble(scored_df: pd.DataFrame) -> None:
    """
    Render the floating glassmorphism chat bubble and its associated dialog.
    Call once at the bottom of app.py, after all tabs have been defined,
    so the bubble appears on every tab.
    """
    from streamlit.components.v1 import html as components_html

    if _HISTORY_KEY not in st.session_state:
        st.session_state[_HISTORY_KEY] = []

    # Invisible Streamlit button that the JS bubble click delegates to.
    open_clicked = st.button("__open_chat__", key="_open_chat_btn")

    # Inject the glassmorphism bubble into the parent document.
    components_html(_BUBBLE_INJECTOR, height=0, width=0)

    if open_clicked or st.session_state.get("_chat_dialog_open"):
        st.session_state["_chat_dialog_open"] = False
        _chat_dialog(scored_df)
