"""
One-page executive summary PDF, branded for Enehano.

Built with reportlab so it has zero browser / headless-Chrome dependency.
The output is a single A4 page: header band, headline number, KPI strip,
top-5 leads table, and a small bar chart comparing strategies.
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart

ENEHANO_GREEN = colors.HexColor("#a6ce39")
DARK = colors.HexColor("#1a1f25")
MUTED = colors.HexColor("#5a6470")
LIGHT_BG = colors.HexColor("#f6f8f1")


def _fmt_money(x: float, currency: str = "CZK") -> str:
    if x >= 1_000_000:
        return f"{x / 1_000_000:,.2f}M {currency}"
    if x >= 1_000:
        return f"{x / 1_000:,.0f}k {currency}"
    return f"{x:,.0f} {currency}"


def _strategy_chart(random_rev: float, baseline_rev: float, ai_rev: float) -> Drawing:
    d = Drawing(400, 130)
    bc = VerticalBarChart()
    bc.x, bc.y = 40, 18
    bc.width, bc.height = 320, 95
    bc.data = [[random_rev / 1_000_000, baseline_rev / 1_000_000, ai_rev / 1_000_000]]
    bc.categoryAxis.categoryNames = ["Random", "Rule-based", "AI"]
    bc.categoryAxis.labels.fontName = "Helvetica"
    bc.categoryAxis.labels.fontSize = 9
    bc.valueAxis.valueMin = 0
    bc.valueAxis.labels.fontSize = 8
    bc.bars[0].fillColor = ENEHANO_GREEN
    bc.bars.strokeColor = colors.white
    bc.barWidth = 30
    d.add(bc)
    return d


def build_pdf(headline_uplift: float, currency: str,
              ai_auc: float, base_auc: float,
              total_leads: int, high_leads: int, expected_wins: int,
              top_leads: pd.DataFrame,
              random_rev: float, baseline_rev: float, ai_rev: float) -> bytes:
    """Return the PDF as a bytes blob ready for st.download_button."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title="Enehano Lead Intelligence — Executive Summary")
    styles = getSampleStyleSheet()
    h_brand = ParagraphStyle("brand", parent=styles["Normal"], fontName="Helvetica-Bold",
                             fontSize=10, textColor=MUTED, spaceAfter=2)
    h_title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold",
                             fontSize=22, textColor=DARK, spaceAfter=4, leading=26)
    h_sub = ParagraphStyle("sub", parent=styles["Normal"], fontName="Helvetica",
                           fontSize=10, textColor=MUTED, spaceAfter=12)
    h_section = ParagraphStyle("section", parent=styles["Heading3"],
                               fontName="Helvetica-Bold", fontSize=12,
                               textColor=DARK, spaceBefore=10, spaceAfter=6)
    h_headline = ParagraphStyle("headline", parent=styles["Normal"],
                                fontName="Helvetica-Bold", fontSize=28,
                                textColor=ENEHANO_GREEN, alignment=1, leading=32)
    h_headline_sub = ParagraphStyle("hsub", parent=styles["Normal"],
                                    fontName="Helvetica", fontSize=10,
                                    textColor=MUTED, alignment=1, spaceAfter=8)

    story = []
    story.append(Paragraph("ENEHANO · LEAD INTELLIGENCE", h_brand))
    story.append(Paragraph("Executive summary", h_title))
    story.append(Paragraph(
        f"Generated {datetime.now():%d %B %Y, %H:%M} · "
        f"Based on {total_leads:,} scored leads.", h_sub))

    headline_box = Table(
        [[Paragraph(f"+{_fmt_money(headline_uplift, currency)} / quarter", h_headline)],
         [Paragraph("Estimated extra revenue when reps work AI's top picks first, "
                    "compared to today's rule-based scoring.", h_headline_sub)]],
        colWidths=[doc.width],
    )
    headline_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 1, ENEHANO_GREEN),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(headline_box)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Pipeline at a glance", h_section))
    kpi_data = [
        ["Total leads", "High-potential", "Expected wins", "AI quality (AUC)"],
        [f"{total_leads:,}", f"{high_leads:,}", f"{expected_wins:,}",
         f"{ai_auc:.1%}  vs  {base_auc:.1%} rules"],
    ]
    kpi = Table(kpi_data, colWidths=[doc.width / 4] * 4, rowHeights=[14, 26])
    kpi.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 14),
        ("TEXTCOLOR", (0, 1), (-1, 1), DARK),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, MUTED),
    ]))
    story.append(kpi)

    story.append(Paragraph("Top 5 leads to call this week", h_section))
    rows = [["#", "Company", "Industry", "Region", "AI score", "Win %"]]
    for i, (_, r) in enumerate(top_leads.iterrows(), 1):
        rows.append([str(i), str(r["Company_Name"])[:34], str(r["Industry"])[:18],
                     str(r["Region"])[:14], f"{int(r['AI_Score'])}",
                     f"{int(r['Expected_Win_%'])}%"])
    tbl = Table(rows, colWidths=[10 * mm, 60 * mm, 35 * mm, 30 * mm, 20 * mm, 20 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ENEHANO_GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), DARK),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.25, MUTED),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tbl)

    story.append(Paragraph("Revenue per strategy (millions)", h_section))
    story.append(KeepTogether(_strategy_chart(random_rev, baseline_rev, ai_rev)))

    doc.build(story)
    return buf.getvalue()

