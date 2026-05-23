"""
ui_cards_html.py
================
HTML/CSS/JS template for the swipeable lead-card deck — Glassmorphism edition.

Kept separate from `ui_cards.py` so the Streamlit-side data prep stays
small and the front-end source is easy to read on its own. The Python
side serialises the cards as JSON and substitutes a single `__CARDS__`
placeholder; everything else (gestures, animation, rendering) runs in
the iframe.

Design: full Glassmorphism — backdrop-filter blur, semi-transparent
surfaces, thin luminous borders, subtle box-shadows — with the Enehano
Green (#a6ce39) preserved as the primary accent colour.
"""
from __future__ import annotations

import json
from typing import Any

ENEHANO_GREEN = "#a6ce39"

# ── Glassmorphism CSS ─────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600;700&family=Space+Grotesk:wght@700;800&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
    height: 100%;
    font-family: 'DM Sans', -apple-system, sans-serif;
    color: #f4f7fb;
    background: transparent;
    overflow: hidden;
}

/* ── Layout ── */
.deck-wrap {
    position: relative;
    width: 100%;
    height: 560px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.deck {
    position: relative;
    width: min(560px, 92vw);
    height: 490px;
}

/* ── Card surface — Glassmorphism core ── */
.card {
    position: absolute;
    inset: 0;
    padding: 26px 30px;
    border-radius: 24px;

    /* Glass surface */
    background: rgba(9, 14, 20, 0.88);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);

    /* Luminous thin border */
    border: 1px solid rgba(230, 238, 247, 0.24);
    border-top: 1px solid rgba(230, 238, 247, 0.36); /* top highlight */

    /* Depth shadow */
    box-shadow:
        0 24px 56px rgba(0, 0, 0, 0.60),
        0 4px  16px rgba(0, 0, 0, 0.30),
        inset 0 1px 0 rgba(255, 255, 255, 0.10);

    /* Interaction states */
    transition: transform 360ms cubic-bezier(.2, .9, .3, 1.2),
                opacity   260ms ease;
    will-change: transform, opacity;
    cursor: grab;
    user-select: none;
    touch-action: none;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* Subtle noise grain overlay for realism */
.card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 24px;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    opacity: 0.6;
}

.card.dragging { transition: none; cursor: grabbing; }
.card.gone-left  { transform: translate(-165%, 40px) rotate(-26deg) !important; opacity: 0; }
.card.gone-right { transform: translate( 165%, 40px) rotate( 26deg) !important; opacity: 0; }

/* ── Card content ── */
.card .top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
}

.card .meta {
    font-size: 10px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #cfd8e3;
    opacity: 1;
    margin-bottom: 5px;
}

.card .name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.5px;
}

.card .source {
    font-size: 12px;
    color: #cfd8e3;
    opacity: 1;
    margin-top: 5px;
}

/* Score block */
.card .score-box {
    text-align: center;
    min-width: 120px;
    flex-shrink: 0;
}

.card .score {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 60px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -3px;
}

.card .score-label {
    font-size: 10px;
    color: #d6dde6;
    opacity: 1;
    margin-top: -2px;
    letter-spacing: 0.06em;
}

.card .badge {
    display: inline-block;
    margin-top: 8px;
    padding: 4px 12px;
    border-radius: 999px;
    color: #111;
    font-weight: 700;
    font-size: 10px;
    letter-spacing: 0.12em;
}

/* Divider */
.card hr {
    border: none;
    border-top: 1px solid rgba(230, 238, 247, 0.18);
    margin: 14px 0;
}

/* Drivers section */
.card .section-title {
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #d6dde6;
    opacity: 1;
    margin-bottom: 8px;
}

.chip {
    display: inline-block;
    background: rgba(166, 206, 57, 0.18);
    border: 1px solid rgba(166, 206, 57, 0.55);
    border-radius: 999px;
    padding: 5px 13px;
    margin: 3px 5px 3px 0;
    font-size: 12px;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
}

/* Stats row */
.row {
    display: flex;
    gap: 20px;
    margin-top: 12px;
    font-size: 13px;
}

.row .lbl { color: #cfd8e3; opacity: 1; margin-right: 4px; }

/* Action strip */
.action {
    margin-top: auto;
    padding: 11px 15px;
    background: rgba(166, 206, 57, 0.16);
    border-left: 3px solid #a6ce39;
    border-radius: 8px;
    font-size: 13px;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

/* ── Drag overlays ── */
.stamp {
    position: absolute;
    top: 22px;
    padding: 5px 14px;
    border: 3px solid;
    border-radius: 8px;
    font-weight: 800;
    font-size: 20px;
    letter-spacing: 0.12em;
    opacity: 0;
    transition: opacity 90ms ease;
    pointer-events: none;
    backdrop-filter: blur(4px);
}

.stamp.like {
    right: 22px;
    color: #a6ce39;
    border-color: #a6ce39;
    transform: rotate(14deg);
}

.stamp.skip {
    left: 22px;
    color: #ff4b4b;
    border-color: #ff4b4b;
    transform: rotate(-14deg);
}

/* ── Controls ── */
.controls {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    display: flex;
    justify-content: center;
    gap: 14px;
    padding: 8px;
}

.btn {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    border: none;
    font-size: 20px;
    cursor: pointer;
    color: #111;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45);
    transition: transform 140ms ease, box-shadow 140ms ease;
    backdrop-filter: blur(8px);
}

.btn:hover {
    transform: scale(1.10);
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.55);
}

.btn.skip { background: linear-gradient(135deg, #ff4b4b, #cc3333); }
.btn.like { background: linear-gradient(135deg, #a6ce39, #7daa1e); }
.btn.undo { background: linear-gradient(135deg, #ffc107, #e0a800); font-size: 17px; }

/* ── Counter ── */
.counter {
    position: absolute;
    top: 8px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 11px;
    color: #cfd8e3;
    opacity: 1;
    letter-spacing: 0.12em;
}

/* ── Empty state ── */
.empty {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 14px;
    font-size: 17px;
    color: #d6dde6;
    opacity: 1;
}

/* ── Hint ── */
.hint {
    text-align: center;
    margin-top: 8px;
    font-size: 11px;
    color: #cfd8e3;
    opacity: 1;
    letter-spacing: 0.06em;
}
"""

# ── JavaScript ────────────────────────────────────────────────────────────────
# Plain string (no f-string) to avoid escaping curly braces.
_JS = r"""
const CARDS = __CARDS__;
const deck    = document.getElementById('deck');
const counter = document.getElementById('counter');
const STACK_VISIBLE = 4;
let idx = 0;
const history = [];

function render() {
    deck.innerHTML = '';
    if (idx >= CARDS.length) {
        const e = document.createElement('div');
        e.className = 'empty';
        e.innerHTML =
            '🎉<div style="font-size:15px">You reviewed every top lead.</div>' +
            '<button class="btn undo" onclick="undo()" ' +
            'style="width:auto;padding:0 18px;border-radius:10px;font-size:13px;height:40px">' +
            'Start over</button>';
        deck.appendChild(e);
        counter.textContent = `${CARDS.length} / ${CARDS.length}`;
        return;
    }
    counter.textContent = `${idx + 1} / ${CARDS.length}`;
    const slice = CARDS.slice(idx, idx + STACK_VISIBLE).reverse();
    slice.forEach((c, i) => {
        const depth = slice.length - 1 - i;   // 0 = top card
        const el    = buildCard(c);
        const scale = 1 - depth * 0.035;
        const ty    = depth * 12;
        el.style.transform = `translateY(${ty}px) scale(${scale})`;
        el.style.zIndex    = 100 - depth;
        el.style.opacity   = depth > 2 ? 0 : 1 - depth * 0.12;
        if (depth === 0) attachDrag(el);
        deck.appendChild(el);
    });
}

function buildCard(c) {
    const el = document.createElement('div');
    el.className = 'card';

    const chips = c.drivers.map(d =>
        `<span class="chip">✨ ${escapeHtml(d)}</span>`
    ).join('');

    el.innerHTML = `
        <div class="stamp like">CALL ✓</div>
        <div class="stamp skip">SKIP</div>
        <div class="top">
          <div style="flex:1;min-width:0">
            <div class="meta">${escapeHtml(c.industry)} · ${escapeHtml(c.region)}</div>
            <div class="name">${escapeHtml(c.company)}</div>
            <div class="source">via ${escapeHtml(c.source)} · IČO ${escapeHtml(c.ico)}</div>
          </div>
          <div class="score-box">
            <div class="score" style="color:${c.color}">${c.score}</div>
            <div class="score-label">AI SCORE / 100</div>
            <div class="badge" style="background:${c.color}">${c.segment} POTENTIAL</div>
          </div>
        </div>
        <hr>
        <div class="section-title">Why this lead stands out</div>
        <div>${chips}</div>
        <div class="row">
          <div>
            <span class="lbl">Expected win:</span>
            <b style="color:${c.color}"> ${c.expected}%</b>
          </div>
          <div>
            <span class="lbl">Rule-based:</span>
            <b>${c.rule}/100</b>
          </div>
        </div>
        <div class="action">
          <span style="font-size:17px;margin-right:7px">${c.emoji}</span>
          <b>Next step:</b> ${escapeHtml(c.action)}
        </div>
    `;
    return el;
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, m => ({
        '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
    }[m]));
}

function attachDrag(el) {
    let startX = 0, startY = 0, dx = 0, dy = 0, dragging = false;
    const likeStamp = el.querySelector('.stamp.like');
    const skipStamp = el.querySelector('.stamp.skip');

    const onDown = e => {
        dragging = true;
        el.classList.add('dragging');
        startX = e.clientX;
        startY = e.clientY;
        el.setPointerCapture(e.pointerId);
    };

    const onMove = e => {
        if (!dragging) return;
        dx = e.clientX - startX;
        dy = e.clientY - startY;
        const rot = dx / 16;
        el.style.transform = `translate(${dx}px, ${dy}px) rotate(${rot}deg)`;
        likeStamp.style.opacity = Math.max(0, Math.min(1,  dx / 110));
        skipStamp.style.opacity = Math.max(0, Math.min(1, -dx / 110));
    };

    const onUp = () => {
        if (!dragging) return;
        dragging = false;
        el.classList.remove('dragging');
        if      (dx >  110) commit('right');
        else if (dx < -110) commit('left');
        else {
            el.style.transform = '';
            likeStamp.style.opacity = 0;
            skipStamp.style.opacity = 0;
            dx = 0; dy = 0;
        }
    };

    el.addEventListener('pointerdown',  onDown);
    el.addEventListener('pointermove',  onMove);
    el.addEventListener('pointerup',    onUp);
    el.addEventListener('pointercancel', onUp);
}

function commit(direction) {
    const top = deck.querySelector('.card:last-child');
    if (!top) return;
    top.classList.add(direction === 'left' ? 'gone-left' : 'gone-right');
    history.push(idx);
    setTimeout(() => { idx++; render(); }, 290);
}

function undo() {
    if (history.length === 0) { idx = 0; render(); return; }
    idx = history.pop();
    render();
}

window.addEventListener('keydown', e => {
    if      (e.key === 'ArrowRight') commit('right');
    else if (e.key === 'ArrowLeft')  commit('left');
    else if (e.key === 'Backspace')  undo();
});

document.getElementById('btn-skip').onclick = () => commit('left');
document.getElementById('btn-like').onclick = () => commit('right');
document.getElementById('btn-undo').onclick = undo;

render();
"""


def build_html(cards: list[dict[str, Any]]) -> str:
    """
    Render the full self-contained deck HTML for Streamlit's components.html().

    Parameters
    ----------
    cards:
        List of card dicts produced by ui_cards._row_to_card().
        Each dict must include keys: ico, company, industry, region, source,
        score, color, segment, drivers, expected, rule, emoji, action.

    Returns
    -------
    str
        A self-contained HTML document string ready for injection via
        ``streamlit.components.v1.html()``.
    """
    # Inject card data as a JSON literal replacing the placeholder token.
    js = _JS.replace("__CARDS__", json.dumps(cards))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>{_CSS}</style>
</head>
<body>
  <div class="deck-wrap">
    <div id="counter" class="counter">1 / 1</div>
    <div id="deck"    class="deck"></div>
  </div>
  <div class="controls">
    <button id="btn-skip" class="btn skip" title="Skip (←)">✕</button>
    <button id="btn-undo" class="btn undo" title="Undo (Backspace)">↶</button>
    <button id="btn-like" class="btn like" title="Call (→)">✓</button>
  </div>
  <div class="hint">Drag the card · ← skip · → call · Backspace undo</div>
  <script>{js}</script>
</body>
</html>"""
