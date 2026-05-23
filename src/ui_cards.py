"""
ui_cards.py
===========
Tinder-style lead card stack — API-backed edition.

Key drivers are sourced directly from the API response (SHAP top_drivers
returned by /score/batch or /score/enrich), eliminating the need for local
SHAP computation in the frontend.

Column contract:
  - Uses 'Region'     (aliased from Region_Name in load_data())
  - Uses 'Rule_Score' (populated by score_pipeline_via_api() from API)
  - Uses 'top_drivers' (list[str] column added by score_pipeline_via_api())
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

ENEHANO_GREEN = "#a6ce39"
WARN          = "#ffc107"
DANGER        = "#ff4b4b"


def suggested_action(row: pd.Series) -> tuple[str, str]:
    """Return (emoji, action_text) based on lead behavioural signals."""
    if row.get("Demo_Requested__c"):
        return "📞", "Call within 24h — demo already requested"
    if row.get("Proposal_Sent__c"):
        return "✉️", "Follow up on the open proposal"
    if float(row.get("Email_Clicks__c", 0)) >= 3:
        return "📨", "Send tailored case study (high email engagement)"
    if float(row.get("Web_Interactions__c", 0)) >= 5:
        return "💬", "Open with a chat — they keep visiting the site"
    if float(row.get("Time_to_First_Response_h__c", 99)) < 4:
        return "📞", "Hot lead — call today"
    return "📅", "Schedule discovery call this week"


def _row_to_card(row: pd.Series) -> dict:
    """
    Convert a scored DataFrame row into a card dict for the HTML renderer.

    top_drivers is consumed directly from the API response stored in the
    'top_drivers' column — no local SHAP computation required.
    """
    score = round(float(row["AI_Score"]), 1)

    if score >= 70:
        color = ENEHANO_GREEN
    elif score >= 40:
        color = WARN
    else:
        color = DANGER

    # API already computed the SHAP drivers; fall back gracefully if missing.
    raw_drivers = row.get("top_drivers") or []
    # The column may be stored as a list or a string representation of a list.
    if isinstance(raw_drivers, str):
        import ast
        try:
            raw_drivers = ast.literal_eval(raw_drivers)
        except (ValueError, SyntaxError):
            raw_drivers = []
    drivers: list[str] = list(raw_drivers)[:3] or ["promising profile"]

    emoji, action = suggested_action(row)

    return {
        "ico":      str(row.get("IČO", "")),
        "company":  str(row.get("Company_Name", "—")),
        "industry": str(row.get("Industry", "—")),
        "region":   str(row.get("Region", "—")),
        "source":   str(row.get("LeadSource", "—")),
        "score":    score,
        "color":    color,
        "segment":  str(row.get("Segment", "—")).upper(),
        "drivers":  drivers,
        "expected": round(float(row.get("Expected_Win_%", 0)), 1),
        "rule":     int(row.get("Rule_Score", 0)),
        "emoji":    emoji,
        "action":   action,
    }


def render(scored_df: pd.DataFrame,
           importance_df: pd.DataFrame | None,
           top_n: int = 20) -> None:
    """
    Render the swipeable lead card deck.

    Parameters
    ----------
    scored_df:
        Full scored pipeline DataFrame (must include 'top_drivers' column).
    importance_df:
        Feature importance table from train.py (kept for API compatibility;
        no longer used for driver computation — API SHAP values take precedence).
    top_n:
        Maximum number of cards to display.
    """
    from streamlit.components.v1 import html as components_html
    import ui_cards_html

    # Prefer High-potential leads; fall back to overall top-N if pool is small.
    pool = scored_df[scored_df["Segment"] == "High"].sort_values(
        "AI_Score", ascending=False
    )
    if len(pool) < top_n:
        pool = scored_df.sort_values("AI_Score", ascending=False)
    pool = pool.head(top_n).reset_index(drop=True)

    if pool.empty:
        st.info("No leads to display yet.")
        return

    st.subheader("Swipe through your hottest leads")
    st.caption(
        f"{len(pool)} top-ranked leads, deck-style. "
        "**Drag** the card, hit **→** / **←**, or use the buttons. "
        "Swipe right = call this lead · left = skip · ↶ = undo.  \n"
        "Key drivers are computed by the backend SHAP explainer."
    )

    cards = [_row_to_card(row) for _, row in pool.iterrows()]
    components_html(ui_cards_html.build_html(cards), height=680, scrolling=False)

    st.divider()
    pick_l, pick_r = st.columns([3, 2])
    with pick_l:
        labels = [f"{i+1}. {c['company']} ({c['score']})" for i, c in enumerate(cards)]
        choice = st.selectbox(
            "Open one of these leads in Lead Profile",
            options=list(range(len(cards))),
            format_func=lambda i: labels[i],
        )
    with pick_r:
        st.write("")
        if st.button("📂 Open in Lead Profile", use_container_width=True):
            st.session_state["selected_ico"] = cards[choice]["ico"]
            st.toast(
                "Switch to the 🏢 Lead Profile tab to inspect this lead.",
                icon="➡",
            )
