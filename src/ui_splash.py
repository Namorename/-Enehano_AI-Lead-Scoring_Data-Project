"""
Splash screen shown once per session before the dashboard loads.

Renders the Enehano logo on a green gradient with a fade-in / slide-up
animation, then the dashboard mounts underneath. CSS-only - no JS, no
extra deps. Runs entirely client-side; the Python side only decides
whether to inject the markup.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

ENEHANO_GREEN = "#a6ce39"
_SESSION_KEY = "_splash_shown"


def _logo_svg(logo_path: Path) -> str:
    """Inline the SVG so the splash works even before Streamlit mounts images."""
    if not logo_path.exists():
        return f"<div style='font-size:64px;font-weight:800;color:#fff'>enehano</div>"
    try:
        return logo_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def maybe_show(logo_path: Path, duration_ms: int = 1800) -> None:
    """Inject the splash overlay on the first run of a session.

    The overlay auto-removes itself via CSS animation, so the dashboard
    underneath becomes visible without a Python rerun.
    """
    if st.session_state.get(_SESSION_KEY):
        return
    st.session_state[_SESSION_KEY] = True

    logo = _logo_svg(logo_path)
    css = f"""
    <style>
    @keyframes splashFadeOut {{
        0%   {{ opacity: 1; }}
        80%  {{ opacity: 1; }}
        100% {{ opacity: 0; visibility: hidden; }}
    }}
    @keyframes logoIn {{
        0%   {{ opacity: 0; transform: translateY(20px) scale(0.9); }}
        60%  {{ opacity: 1; transform: translateY(0) scale(1.02); }}
        100% {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}
    @keyframes tagIn {{
        0%   {{ opacity: 0; transform: translateY(10px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes appSlideUp {{
        0%   {{ opacity: 0; transform: translateY(40px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    #enehano-splash {{
        position: fixed; inset: 0; z-index: 99999;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        background: radial-gradient(circle at 30% 30%, #2d4a0a, #0e1117 70%);
        animation: splashFadeOut {duration_ms}ms ease-in-out forwards;
        pointer-events: none;
    }}
    #enehano-splash .logo {{
        animation: logoIn 900ms cubic-bezier(.2,.9,.3,1.3) both;
        max-width: 320px;
    }}
    #enehano-splash .logo svg {{ width: 320px; height: auto; }}
    #enehano-splash .tag {{
        margin-top: 18px; color: #ffffff; opacity: 0.85;
        font-size: 18px; letter-spacing: 0.08em; text-transform: uppercase;
        animation: tagIn 700ms 400ms ease-out both;
    }}
    #enehano-splash .pulse {{
        margin-top: 36px; width: 64px; height: 4px; border-radius: 2px;
        background: linear-gradient(90deg, transparent, {ENEHANO_GREEN}, transparent);
        background-size: 200% 100%;
        animation: tagIn 700ms 600ms ease-out both,
                   pulseBar 1.4s 700ms linear infinite;
    }}
    @keyframes pulseBar {{
        0%   {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
    }}
    /* Dashboard slides up underneath as the splash fades */
    .stApp > div:first-child {{
        animation: appSlideUp {duration_ms}ms ease-out both;
    }}
    </style>
    <div id="enehano-splash">
        <div class="logo">{logo}</div>
        <div class="tag">Lead Intelligence</div>
        <div class="pulse"></div>
    </div>
    """
    st.markdown(css, unsafe_allow_html=True)

