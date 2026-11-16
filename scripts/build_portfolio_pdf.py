"""Build docs/StockPilot_Portfolio.pdf: project explainer, results, architecture and interview Q&A.

    python scripts/build_portfolio_pdf.py

Numbers are read from results/ and configs/ at build time, so re-run it after a retrain.
Needs reportlab (docs-only dependency): pip install reportlab
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "StockPilot_Portfolio.pdf"
REPO = "https://github.com/Abdullah-Akmal/stockpilot"
APP = "https://stockpilot-7wqs2osthj56yvsma2mmaq.streamlit.app/"

# ----------------------------------------------------------------------------- fonts & palette
FONT_DIR = Path("C:/Windows/Fonts")
try:
    pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "segoeui.ttf")))
    pdfmetrics.registerFont(TTFont("Bold", str(FONT_DIR / "segoeuib.ttf")))
    pdfmetrics.registerFont(TTFont("Light", str(FONT_DIR / "segoeuisl.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", str(FONT_DIR / "consola.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold", italic="Body", boldItalic="Bold")
except Exception:  # non-Windows fallback
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont  # noqa: F401
    for alias, real in [("Body", "Helvetica"), ("Bold", "Helvetica-Bold"), ("Light", "Helvetica"), ("Mono", "Courier")]:
        pdfmetrics.registerFontFamily(alias, normal=real)

INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#475569")
FAINT = colors.HexColor("#94a3b8")
BLUE = colors.HexColor("#2563eb")
BLUE_BG = colors.HexColor("#eff6ff")
ORANGE = colors.HexColor("#f97316")
GREEN = colors.HexColor("#15803d")
RED = colors.HexColor("#b91c1c")
LINE = colors.HexColor("#e2e8f0")
PANEL = colors.HexColor("#f8fafc")

S = {
    "title": ParagraphStyle("title", fontName="Bold", fontSize=34, leading=40, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontName="Light", fontSize=15, leading=21, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="Bold", fontSize=19, leading=24, textColor=INK, spaceBefore=4, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Bold", fontSize=12.5, leading=17, textColor=BLUE, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=14.2, textColor=INK, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="Body", fontSize=8.2, leading=11.5, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.4, leading=11.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Bold", fontSize=8.4, leading=11.2, textColor=INK),
    "head": ParagraphStyle("head", fontName="Bold", fontSize=8.4, leading=11.2, textColor=colors.white),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=9.6, leading=14, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=2.5),
    "q": ParagraphStyle("q", fontName="Bold", fontSize=10, leading=14, textColor=INK, spaceBefore=7, spaceAfter=2),
    "a": ParagraphStyle("a", fontName="Body", fontSize=9.4, leading=13.6, textColor=INK, leftIndent=10, spaceAfter=3),
    "kpi_v": ParagraphStyle("kpi_v", fontName="Bold", fontSize=20, leading=24, textColor=INK),
    "kpi_l": ParagraphStyle("kpi_l", fontName="Bold", fontSize=7.4, leading=10, textColor=MUTED),
    "kpi_s": ParagraphStyle("kpi_s", fontName="Body", fontSize=8, leading=10.5, textColor=MUTED),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.2, leading=11, textColor=INK, backColor=PANEL,
                           borderPadding=(5, 6, 5, 6), spaceBefore=4, spaceAfter=8),
    "center": ParagraphStyle("center", fontName="Body", fontSize=8.2, leading=11, textColor=MUTED, alignment=TA_CENTER),
}
W = A4[0] - 36 * mm   # content width


def short(name: str, n: int) -> str:
    """Title-case a product name and cut it at a word boundary."""
    name = name.strip().title()
    if len(name) <= n:
        return name
    return name[:n].rsplit(" ", 1)[0] + "…"


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="bullet"):
    return [Paragraph(t, S[style], bulletText="•") for t in items]


def table(rows, widths, header=True, zebra=True, align_right_from=None, font_style="cell"):
    data = []
    for i, r in enumerate(rows):
        st = "head" if (header and i == 0) else font_style
        data.append([c if not isinstance(c, str) else Paragraph(c, S[st]) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE)]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), INK)]
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                cmds.append(("BACKGROUND", (0, i), (-1, i), PANEL))
    t.setStyle(TableStyle(cmds))
    return t


def kv_table(rows, w0=38 * mm):
    return table([[f"<b>{k}</b>", v] for k, v in rows], [w0, W - w0], header=False)


def callout(text, color=BLUE, bg=BLUE_BG):
    t = Table([[Paragraph(text, S["body"])]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, color),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def img(path, width=W, max_h=None):
    im = PILImage.open(path)
    w, h = im.size
    height = width * h / w
    if max_h and height > max_h:
        height, width = max_h, max_h * w / h
    return Image(str(path), width=width, height=height)


def kpis(items):
    cells = []
    for label, value, sub in items:
        c = Table([[Paragraph(label.upper(), S["kpi_l"])], [Paragraph(value, S["kpi_v"])], [Paragraph(sub, S["kpi_s"])]],
                  colWidths=[W / len(items) - 4 * mm])
        c.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.white), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (0, 0), 7),
                               ("BOTTOMPADDING", (0, -1), (-1, -1), 7)]))
        cells.append(c)
    t = Table([cells], colWidths=[W / len(items)] * len(items))
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm)]))
    return t


# ----------------------------------------------------------------------------- diagrams
def _box(d, x, y, w, h, title, sub="", fill=colors.white, stroke=INK, tcolor=INK, fs=8.2):
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=stroke, strokeWidth=0.8))
    if sub:
        d.add(String(x + w / 2, y + h / 2 + 2.5, title, fontName="Bold", fontSize=fs, fillColor=tcolor, textAnchor="middle"))
        for i, line in enumerate(sub.split("\n")):
            d.add(String(x + w / 2, y + h / 2 - 7.5 - i * 8.5, line, fontName="Body", fontSize=6.6, fillColor=MUTED,
                         textAnchor="middle"))
    else:
        d.add(String(x + w / 2, y + h / 2 - 3, title, fontName="Bold", fontSize=fs, fillColor=tcolor, textAnchor="middle"))


def _arrow(d, x1, y1, x2, y2, color=MUTED, dash=None, label=None):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=0.9, strokeDashArray=dash))
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    s = 4.5
    d.add(Polygon([x2, y2, x2 - s * math.cos(a - 0.45), y2 - s * math.sin(a - 0.45),
                   x2 - s * math.cos(a + 0.45), y2 - s * math.sin(a + 0.45)], fillColor=color, strokeColor=color))
    if label:
        d.add(String((x1 + x2) / 2, (y1 + y2) / 2 + 3, label, fontName="Body", fontSize=6.2, fillColor=MUTED,
                     textAnchor="middle"))


def architecture() -> Drawing:
    w, h = W, 300
    d = Drawing(w, h)
    src_fill = colors.HexColor("#f1f5f9")
    core_fill = BLUE_BG
    art_fill = colors.HexColor("#fff7ed")
    cx = [0, 108, 222, 336, 418]   # column x
    bw = 96
    # column labels
    for x, t in zip([cx[0], cx[1], cx[2], cx[3] - 6, cx[4]], ["DATA SOURCES", "DATA LAYER", "LEARNING", "ARTEFACTS", "SERVING"]):
        d.add(String(x + (bw if t != "SERVING" else 92) / 2, h - 10, t, fontName="Bold", fontSize=6.8, fillColor=FAINT,
                     textAnchor="middle"))
    # sources
    _box(d, cx[0], 225, bw, 40, "Google Sheet", "LIVE mode · published CSV\nor gspread", fill=src_fill)
    _box(d, cx[0], 170, bw, 40, "UCI Online Retail II", "PUBLIC mode · 1.07M\ntransactions", fill=src_fill)
    _box(d, cx[0], 115, bw, 40, "Demo shop CSV", "synthetic test data\n(+ broken copy)", fill=src_fill)
    # data layer
    _box(d, cx[1], 190, bw, 52, "data.py", "validate schema · clean\ncancellations · daily per SKU", fill=core_fill,
         stroke=BLUE)
    _box(d, cx[1], 128, bw, 40, "Chronological split", "75% train / 25% test\n(no leakage)", fill=core_fill, stroke=BLUE)
    # learning
    _box(d, cx[2], 215, bw, 44, "baselines.py", "(s,S) · base-stock · MA+SS\ngrid search on train", fill=core_fill,
         stroke=BLUE)
    _box(d, cx[2], 150, bw, 48, "env.py + train.py", "Gymnasium env · PPO\nVecNormalize · 3 seeds", fill=core_fill,
         stroke=BLUE)
    _box(d, cx[2], 95, bw, 40, "Promotion gate", "new agent only if cheaper\non TRAIN period", fill=core_fill,
         stroke=BLUE)
    # artefacts
    _box(d, cx[3] - 6, 200, 76, 44, "models/<sku>/", "ppo.zip · vecnorm.pkl\nbaselines.json", fill=art_fill,
         stroke=ORANGE, fs=7.6)
    _box(d, cx[3] - 6, 130, 76, 44, "evaluate.py", "replay held-out\ndemand · ±30%", fill=core_fill, stroke=BLUE,
         fs=7.6)
    _box(d, cx[3] - 6, 70, 76, 40, "results/", "metrics.csv · plots", fill=art_fill, stroke=ORANGE, fs=7.6)
    # serving
    _box(d, cx[4], 150, 92, 64, "Streamlit app", "Today's Orders · Backtest\nWhat-if · About\n(inference only)",
         fill=colors.HexColor("#ecfdf5"), stroke=GREEN)
    _box(d, cx[4], 85, 92, 36, "Streamlit Cloud", "public URL", fill=colors.HexColor("#ecfdf5"), stroke=GREEN)
    # arrows
    _arrow(d, cx[0] + bw, 245, cx[1], 225)
    _arrow(d, cx[0] + bw, 190, cx[1], 210)
    _arrow(d, cx[0] + bw, 135, cx[1], 198)
    _arrow(d, cx[1] + bw / 2, 190, cx[1] + bw / 2, 168)
    _arrow(d, cx[1] + bw, 160, cx[2], 225)
    _arrow(d, cx[1] + bw, 150, cx[2], 174)
    _arrow(d, cx[2] + bw / 2, 150, cx[2] + bw / 2, 135)
    _arrow(d, cx[2] + bw, 237, cx[3] - 6, 228)
    _arrow(d, cx[2] + bw, 115, cx[3] - 6, 212)
    _arrow(d, cx[3] + 32, 200, cx[3] + 32, 174)
    _arrow(d, cx[3] + 32, 130, cx[3] + 32, 110)
    _arrow(d, cx[3] + 70, 222, cx[4], 195)
    _arrow(d, cx[3] + 70, 90, cx[4], 160)
    _arrow(d, cx[4] + 46, 150, cx[4] + 46, 121)
    _arrow(d, cx[0] + bw / 2, 265, cx[0] + bw / 2 + 0.01, 278, color=colors.white)
    d.add(Line(cx[0] + bw / 2, 265, cx[0] + bw / 2, 272, strokeColor=GREEN, strokeWidth=0.8, strokeDashArray=[2, 2]))
    d.add(Line(cx[0] + bw / 2, 272, cx[4] + 80, 272, strokeColor=GREEN, strokeWidth=0.8, strokeDashArray=[2, 2]))
    _arrow(d, cx[4] + 80, 272, cx[4] + 80, 214, color=GREEN, dash=[2, 2])
    d.add(String(260, 275, "live sheet read at request time (today's stock & demand)", fontName="Body", fontSize=6.2,
                 fillColor=GREEN, textAnchor="middle"))
    # CI loop
    _box(d, 108, 18, 300, 36, "GitHub Actions · weekly (Mon 03:00 UTC) + manual",
         "tests → pull data → tune baselines → train challengers → promote → backtest → commit",
         fill=colors.HexColor("#f5f3ff"), stroke=colors.HexColor("#7c3aed"))
    _arrow(d, 408, 36, cx[4] + 46, 85, color=colors.HexColor("#7c3aed"))
    d.add(String(cx[4] + 52, 58, "push → redeploy", fontName="Body", fontSize=6.2, fillColor=colors.HexColor("#7c3aed")))
    _arrow(d, 150, 54, cx[1] + 40, 128, color=colors.HexColor("#7c3aed"), dash=[2, 2])
    return d


def daily_cycle() -> Drawing:
    w, h = W, 78
    d = Drawing(w, h)
    steps = [("1 · Observe", "stock, pipeline,\n14-day demand, weekday"), ("2 · Order", "agent picks qty;\nfixed cost if > 0"),
             ("3 · Sell", "demand served FIFO;\nshortfall is lost"), ("4 · Age", "lots expire if a\nshelf life is set"),
             ("5 · Hold & receive", "holding cost; order\nplaced L days ago lands")]
    bw, gap = (w - 4 * 12) / 5, 12
    for i, (t, sub) in enumerate(steps):
        x = i * (bw + gap)
        _box(d, x, 14, bw, 52, t, sub, fill=BLUE_BG if i != 1 else colors.HexColor("#dbeafe"), stroke=BLUE)
        if i < 4:
            _arrow(d, x + bw + 1, 40, x + bw + gap - 1, 40, color=BLUE)
    d.add(String(w / 2, 2, "reward = −(holding + ordering + stockout penalty + waste) ÷ (margin × mean demand)",
                 fontName="Bold", fontSize=7.4, fillColor=INK, textAnchor="middle"))
    return d


# ----------------------------------------------------------------------------- page chrome
def on_page(c, doc):
    c.saveState()
    if doc.page > 1:
        c.setFont("Bold", 7.5)
        c.setFillColor(INK)
        c.drawString(18 * mm, A4[1] - 11 * mm, "Stock")
        c.setFillColor(BLUE)
        c.drawString(18 * mm + pdfmetrics.stringWidth("Stock", "Bold", 7.5), A4[1] - 11 * mm, "Pilot")
        c.setFillColor(FAINT)
        c.setFont("Body", 7.5)
        c.drawRightString(A4[0] - 18 * mm, A4[1] - 11 * mm, "Project portfolio & Q&A")
        c.setStrokeColor(LINE)
        c.line(18 * mm, A4[1] - 13 * mm, A4[0] - 18 * mm, A4[1] - 13 * mm)
        c.drawString(18 * mm, 10 * mm, REPO.replace("https://", ""))
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    c.restoreState()


# ----------------------------------------------------------------------------- content
def build() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    m = pd.read_csv(ROOT / "results" / "metrics.csv", dtype={"sku": str})
    summ = pd.read_csv(ROOT / "results" / "summary.csv", dtype={"sku": str})
    desc = json.loads((ROOT / "data" / "processed" / "sku_descriptions.json").read_text())
    daily = pd.read_csv(ROOT / "data" / "processed" / "daily_public.csv", dtype={"sku": str}, parse_dates=["date"])
    c = cfg["costs"]

    def agg(p):
        x = m[m.policy == p]
        return dict(cost=x.total_cost.sum(), hold=x.holding_cost.sum(), order=x.ordering_cost.sum(),
                    lost=x.lost_sales_cost.sum(), fill=(x.demand_units - x.lost_units).sum() / x.demand_units.sum(),
                    inv=x.avg_inventory.sum(), n=int(x.n_orders.sum()), lost_u=int(x.lost_units.sum()))
    A = {p: agg(p) for p in ["ppo", "sS", "base_stock", "ma_safety"]}
    base = summ[summ.scenario == "base"]
    wins = int(base.ppo_beats_sS.sum())
    n_sku = len(base)
    saved = A["sS"]["cost"] - A["ppo"]["cost"]
    pct = 100 * saved / A["sS"]["cost"]
    pv = m.pivot(index="sku", columns="policy", values="total_cost")
    oracle = pv[["sS", "base_stock", "ma_safety"]].min(axis=1).sum()
    wins_bs = int((pv.ppo < pv.base_stock).sum())
    stress = summ.groupby("scenario").agg(ppo=("ppo_cost", "sum"), ss=("sS_cost", "sum"), w=("ppo_beats_sS", "sum"))
    cut = sorted(daily.date.unique())[int(len(daily.date.unique()) * cfg["data"]["train_frac"])]
    test_days = int((daily[daily.date >= cut].groupby("sku").size()).iloc[0])
    losers = base[~base.ppo_beats_sS]

    story = []
    # ---------------- cover
    story += [Spacer(1, 50 * mm)]
    story.append(Paragraph('Stock<font color="#2563eb">Pilot</font>', S["title"]))
    story.append(Spacer(1, 4))
    story.append(P(cfg["project"]["tagline"], "subtitle"))
    story.append(Spacer(1, 14))
    story.append(P("A reinforcement-learning inventory replenishment system that recommends, for every product, how many "
                   "units to order today. It is trained and proven on real retail data, deployed as a public dashboard, "
                   "and retrained weekly by CI."))
    story.append(Spacer(1, 10))
    story.append(kpis([
        ("Cost saved vs (s,S)", f"{pct:.1f}%", f"£{saved:,.0f} on held-out data"),
        ("SKUs where agent wins", f"{wins} / {n_sku}", "vs tuned (s,S) policy"),
        ("Fill rate", f"{A['ppo']['fill']:.1%}", f"(s,S): {A['sS']['fill']:.1%}"),
        ("Automated tests", "36", "all passing"),
    ]))
    story.append(Spacer(1, 16))
    story.append(kv_table([
        ("Live app", f'<link href="{APP}" color="#2563eb">{APP}</link>'),
        ("Source code", f'<link href="{REPO}" color="#2563eb">{REPO}</link>'),
        ("Stack", "Python 3.11 · Gymnasium · Stable-Baselines3 (PPO) · PyTorch (CPU) · pandas · Streamlit · Plotly · "
                  "GitHub Actions · Streamlit Community Cloud"),
        ("Data", "UCI Online Retail II (real UK wholesaler, 2009-2011) · live Google Sheet · synthetic demo shop"),
        ("Document", f"Generated {date.today():%d %B %Y} from the repository's own results files"),
    ], w0=28 * mm))
    story.append(Spacer(1, 18))
    toc = ["1  The problem", "2  The data", "3  How StockPilot works, end to end", "4  The reinforcement-learning design",
           "5  Baselines and evaluation method", "6  Results", "7  The dashboard", "8  Architecture",
           "9  Tech stack", "10  MLOps: automation and retraining", "11  Testing and quality assurance",
           "12  Engineering challenges and how they were solved", "13  Limitations and future work",
           "14  Interview Q&amp;A", "15  Portfolio and CV summary"]
    half = (len(toc) + 1) // 2
    tt = Table([[P(a, "small"), P(b, "small") if b else ""] for a, b in
                zip(toc[:half], toc[half:] + [""])], colWidths=[W / 2, W / 2])
    tt.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    story += [P("<b>Contents</b>", "small"), Spacer(1, 3), tt, PageBreak()]

    # ---------------- 1 problem
    story.append(P("1  The problem", "h1"))
    story.append(P("Every day, for every product, a shop has to answer one question: <b>how many units should I order "
                   "today?</b> Each wrong answer costs money in one of four ways:"))
    story.append(table([
        ["Cost", "What it is", "How StockPilot prices it"],
        ["Holding", "Cash and space tied up in stock on the shelf", f"{c['holding_rate_annual']:.0%} per year of unit cost, charged per unit per day"],
        ["Ordering", "Fixed cost of placing a purchase order (admin, delivery)", f"£{c['order_cost']:.2f} per order"],
        ["Lost sales", "A customer wants an item that is out of stock and leaves", f"Lost margin + goodwill ({c['goodwill_frac']:.0%} of price) per lost unit"],
        ["Waste", "Perishable stock that expires before it sells", "Unit cost of each expired unit (optional shelf life)"],
    ], [24 * mm, 70 * mm, W - 94 * mm]))
    story.append(Spacer(1, 6))
    story.append(P("These pull in opposite directions. Holding more stock avoids lost sales but raises holding cost. "
                   "Ordering more often keeps stock lean but adds ordering cost. On top of that, orders take days to "
                   "arrive (the <b>lead time</b>), and demand is uncertain, lumpy and seasonal."))
    story.append(P(f"<b>Goal:</b> for each SKU, minimise <b>total cost = holding + ordering + lost sales + waste</b> "
                   f"while keeping the fill rate at or above {c['target_fill_rate']:.0%}, and <b>prove</b> it beats "
                   "classic inventory rules on real, held-out data, not on simulated demand."))
    story.append(P("Why this matters", "h2"))
    story += bullets([
        "Inventory is usually a retailer's largest current asset; small percentage improvements are large in money terms.",
        "Most small shops reorder by gut feel or a fixed rule that is never re-tuned when demand changes.",
        "Classic rules such as (s,S) are strong but static: tuned once, they cannot react to a trend, a promotion or a "
        "seasonal peak. A learned policy that observes recent demand can.",
    ])
    story.append(P("What StockPilot delivers", "h2"))
    story += bullets([
        "<b>Today's Orders:</b> a recommended order quantity per SKU, next to the classic rule's quantity, exportable as CSV.",
        "<b>Evidence:</b> a backtest on the last 25% of dates the models never saw, plus ±30% demand stress tests.",
        "<b>What-if:</b> change lead time, holding cost or stockout penalty and re-simulate instantly.",
        "<b>Automation:</b> weekly retraining from the latest data with a safety gate that never deploys a worse agent.",
    ])

    # ---------------- 2 data
    story.append(CondPageBreak(90 * mm))
    story.append(P("2  The data", "h1"))
    story.append(P("StockPilot runs on one schema, so the same code serves three data sources. One row per SKU per day:"))
    story.append(table([
        ["Column", "Meaning"],
        ["date", "Calendar day (YYYY-MM-DD)"], ["sku", "Product code"], ["units_sold", "Units sold that day"],
        ["on_hand", "Units in stock at the <b>end</b> of the day"], ["unit_cost / price", "Purchase cost and selling price per unit"],
        ["lead_time_days", "Days from placing an order to receiving it (0-14)"],
        ["on_order (optional)", "Units ordered but not yet received at end of day"],
    ], [38 * mm, W - 38 * mm]))
    story.append(P("2.1  PUBLIC mode: UCI Online Retail II (real data)", "h2"))
    story.append(P("All transactions of a UK-based online giftware wholesaler from 1 Dec 2009 to 9 Dec 2011, published by the "
                   "UC Irvine Machine Learning Repository (dataset 502, CC BY 4.0). Many customers are small shops buying in bulk, "
                   "so demand is <b>very lumpy</b>: one order can be 20 times a normal day."))
    story.append(table([
        ["Cleaning step", "Result"],
        ["Raw transactions (2 sheets)", "1,067,371 rows"],
        ["Drop cancellations (invoice starts with 'C') and negative quantities", "22,497 rows removed"],
        ["Also remove the original sales those cancellations reverse (same customer, item, quantity)", "13,176 rows removed"],
        ["Keep real products only (5-digit stock codes; drop postage, fees, manual adjustments)", "990,590 clean sale lines"],
        ["Aggregate to daily units per SKU on the store's trading-day calendar", "604 trading days"],
        ["Pick the 10 best-selling SKUs with ≥ 400 days of history and ≥ 70% days with a sale", "204 eligible → top 10"],
    ], [W - 45 * mm, 45 * mm]))
    story.append(Spacer(1, 6))
    sk = []
    tr_all = daily[daily.date < cut]
    for s in sorted(daily.sku.unique()):
        t = tr_all[tr_all.sku == s].units_sold
        sk.append([s, short(desc.get(s, ""), 36), f"{t.mean():.0f}", f"{t.std() / t.mean():.2f}",
                   f"£{daily[daily.sku == s].price.median():.2f}"])
    story.append(KeepTogether([P("The 10 SKUs (training-period statistics). CV = coefficient of variation; above 1 means "
                                 "demand swings by more than its own average.", "small"), Spacer(1, 3),
                               table([["SKU", "Product", "Mean units/day", "CV", "Median price"]] + sk,
                                     [18 * mm, 72 * mm, 28 * mm, 20 * mm, W - 138 * mm])]))
    story.append(P(f"<b>Chronological split, no leakage:</b> the first 75% of trading days (to {pd.Timestamp(cut) - pd.Timedelta(days=1):%d %b %Y}) "
                   f"are for training and tuning; the last 25% ({pd.Timestamp(cut):%d %b %Y} → 9 Dec 2011, {test_days} trading days, "
                   "including the Christmas peak) are only used for the final backtest. Baselines, PPO training and model "
                   "selection never see test data. The dataset has no stock records, so historical on-hand in the sample "
                   "files is simulated; <b>backtests use only the real demand</b>."))
    story.append(P("2.2  LIVE mode: your Google Sheet", "h2"))
    story.append(P("A shop keeps the schema above in a Google Sheet and publishes it as CSV (or shares it with a service "
                   "account). The app reads it at request time, validates it, and reports every problem at once: missing "
                   "columns, unreadable dates, negative or non-numeric values, duplicate rows, lead time above 14 days. "
                   "SKUs without a trained agent get a (s,S) recommendation tuned on their own history until the next retrain."))
    story.append(P("2.3  Demo shop: synthetic test data", "h2"))
    story.append(P("<i>Corner Coffee &amp; Goods</i> is a seeded, fictional shop generated by <font name='Mono'>scripts/make_demo_data.py</font>: "
                   "8 SKUs × one year with weekday patterns, promotions, a Christmas gift peak, lead times of 2-7 days, real "
                   "stockouts from an imperfect manual reorder rule, and one SKU launched 20 days ago. A deliberately broken "
                   "copy contains five typical sheet mistakes to test validation. It is used to test LIVE mode and the full "
                   "pipeline without touching real data."))

    # ---------------- 3 end to end
    story.append(CondPageBreak(90 * mm))
    story.append(P("3  How StockPilot works, end to end", "h1"))
    story.append(table([
        ["Step", "Module", "What happens"],
        ["1  Data", "stockpilot/data.py", "Read the sheet or UCI file, validate, clean, aggregate to daily units per SKU, split by date."],
        ["2  Baselines", "stockpilot/baselines.py", "Grid-search three classic policies on the training period only (coarse grid, then a fine grid around the best)."],
        ["3  Environment", "stockpilot/env.py", "A Gymnasium simulator of one SKU: lead-time pipeline, lost sales, FIFO expiry, costs."],
        ["4  Training", "stockpilot/train.py", "PPO per SKU on bootstrapped real demand; best checkpoint chosen by training-period cost; promoted only if it beats the current agent."],
        ["5  Backtest", "stockpilot/evaluate.py", "Replay the held-out real demand day by day for every policy from the same starting stock; ±30% stress tests; CSVs and plots."],
        ["6  Recommend", "stockpilot/recommend.py", "From each SKU's latest row (end-of-day stock, open orders, last 14 days), compute today's order for the agent and for (s,S)."],
        ["7  Serve", "app/streamlit_app.py", "Dashboard with four pages; loads saved models and only runs inference."],
        ["8  Automate", ".github/workflows/retrain.yml", "Weekly: tests → data → baselines → train → promote → backtest → commit → app redeploys."],
    ], [22 * mm, 42 * mm, W - 64 * mm]))
    story.append(P("One simulator for everything", "h2"))
    story.append(P("Training, baseline tuning, the backtest and the What-if page all run on the same <font name='Mono'>InventorySim</font> "
                   "class. A policy is just a function from the current state to an order quantity, so the agent and every "
                   "baseline are scored by identical arithmetic: the comparison cannot be skewed by two different simulators."))
    story.append(P("A day in the simulator", "h2"))
    story.append(daily_cycle())

    # ---------------- 4 RL
    story.append(CondPageBreak(90 * mm))
    story.append(P("4  The reinforcement-learning design", "h1"))
    story.append(P("Each SKU is its own Markov Decision Process (MDP), and each gets its own PPO agent."))
    story.append(kv_table([
        ("State (38 numbers)", "on-hand stock; inventory position (on-hand + on-order); the order pipeline padded to 14 slots "
                               "(units arriving in 1, 2, … 14 days); the lead time; the last 14 days of demand divided by mean "
                               "demand; day of week (one-hot). Stock values are divided by 10 × mean demand. The fixed size lets "
                               "the What-if page change the lead time without retraining."),
        ("Action", f"Discrete: {cfg['env']['n_actions']} order sizes from 0 to {cfg['env']['action_max_mult']:g}× the mean lead-time demand "
                   "(4× or 2× for some SKUs after tuning). Scaling to each SKU's demand lets one design fit items selling "
                   "5 or 500 units a day."),
        ("Transition", "The order arrives after exactly L days. Demand is served from the oldest stock first (FIFO). Unmet "
                       "demand is lost, not back-ordered. With a shelf life, lots expire and count as waste."),
        ("Reward", "−(holding + ordering + stockout penalty + waste) ÷ (margin × mean demand), so values are roughly in [−1, 0]."),
        ("Training demand", "7-day block bootstrap of the training period's real daily sales: random week-long chunks are "
                            "stitched together, keeping weekday patterns and short-term correlation."),
        ("Algorithm", f"Stable-Baselines3 PPO, 8 parallel environments, VecNormalize (observations and rewards), "
                      f"{cfg['ppo']['net_arch'][0]}×{cfg['ppo']['net_arch'][1]} MLP, γ = {cfg['ppo']['gamma']}, "
                      f"{cfg['ppo']['total_timesteps']:,}-1,500,000 steps, fixed seeds, CPU only (~2-5 min per SKU)."),
    ], w0=32 * mm))
    story.append(P("Why the reward is cost, not profit", "h2"))
    story.append(P("Daily profit = margin × units sold − costs. Units sold = demand − lost units, so profit = <b>margin × demand</b> "
                   "− total cost. Margin × demand is set by customers, not by the agent, so maximising profit and minimising "
                   "cost have <b>exactly the same optimal policy</b>. But with lumpy demand that term swings wildly from day to "
                   "day, and its noise drowned the small, slow signal of holding cost. Switching to the cost-only reward is a "
                   "pure variance reduction."))
    story.append(P("Three details that made PPO work", "h2"))
    story += bullets([
        "<b>Starting with a bias towards ordering nothing.</b> With uniform random exploration the agent ordered ~4.5× demand "
        "per day, stock grew without limit, and it never visited low-stock states. The first agent learned a constant order "
        "that ignored stock entirely. The policy head now starts with P(order 0) = 0.9; only the initial bias changes.",
        "<b>Checkpoint selection on training data.</b> Every 5% of training the policy is replayed on the training period "
        "and the cheapest checkpoint is kept, because PPO learning curves oscillate.",
        "<b>Champion/challenger promotion.</b> Several seeds are trained; the best by training cost replaces the deployed "
        "agent only if it is cheaper on the training period than the current one.",
    ])

    # ---------------- 5 baselines & evaluation
    story.append(P("5  Baselines and evaluation method", "h1"))
    story.append(table([
        ["Baseline", "Rule", "Tuned parameters"],
        ["(s,S)", "If inventory position ≤ s, order up to S; otherwise order nothing", "s and S (coarse-to-fine grid)"],
        ["Base-stock", "Every day, order up to a fixed level B", "B"],
        ["MA + safety stock", "Order up to forecast × (L+1) + z·σ·√(L+1), using a moving-average forecast", "window (7-56 days), z"],
    ], [30 * mm, 95 * mm, W - 125 * mm]))
    story.append(Spacer(1, 4))
    story += bullets([
        "Every baseline is tuned on the <b>training period only</b>, minimising cost subject to the fill-rate target. "
        "Grids were widened after the first run found optima on the grid edges, which would have under-tuned the baselines "
        "and flattered the agent.",
        "<b>Backtest:</b> each policy starts the test period with identical stock and faces the exact real daily demand. "
        "Reported: total cost (split by type), fill rate, average inventory, number of orders, waste.",
        "<b>Stress test:</b> the same test demand scaled by 0.7 and 1.3, with no re-tuning.",
        "<b>Honesty rule:</b> every model choice, including tuning variants and seeds, uses training-period cost. Test "
        "results are only ever reported, never used to choose.",
    ])

    # ---------------- 6 results
    story.append(CondPageBreak(90 * mm))
    story.append(P("6  Results", "h1"))
    story.append(P(f"Held-out test period ({pd.Timestamp(cut):%d %b %Y} → 9 Dec 2011), real demand, all {n_sku} SKUs:", "small"))
    story.append(table([
        ["Policy", "Total cost", "Fill rate", "Avg inventory", "Orders", "Lost units"],
        ["<b>StockPilot PPO</b>", f"<b>£{A['ppo']['cost']:,.0f}</b>", f"<b>{A['ppo']['fill']:.1%}</b>", f"{A['ppo']['inv']:,.0f}", f"{A['ppo']['n']}", f"{A['ppo']['lost_u']:,}"],
        ["(s,S)", f"£{A['sS']['cost']:,.0f}", f"{A['sS']['fill']:.1%}", f"{A['sS']['inv']:,.0f}", f"{A['sS']['n']}", f"{A['sS']['lost_u']:,}"],
        ["Base-stock", f"£{A['base_stock']['cost']:,.0f}", f"{A['base_stock']['fill']:.1%}", f"{A['base_stock']['inv']:,.0f}", f"{A['base_stock']['n']}", f"{A['base_stock']['lost_u']:,}"],
        ["MA + safety stock", f"£{A['ma_safety']['cost']:,.0f}", f"{A['ma_safety']['fill']:.1%}", f"{A['ma_safety']['inv']:,.0f}", f"{A['ma_safety']['n']}", f"{A['ma_safety']['lost_u']:,}"],
    ], [42 * mm] + [(W - 42 * mm) / 5] * 5))
    story.append(Spacer(1, 6))
    story.append(callout(f"<b>StockPilot beats (s,S) on {wins} of {n_sku} SKUs with {pct:.1f}% lower total cost (£{saved:,.0f} saved) "
                         f"and a higher fill rate ({A['ppo']['fill']:.1%} vs {A['sS']['fill']:.1%}). It beats base-stock on "
                         f"{wins_bs} of {n_sku}, and its total is lower than an oracle that picks the best baseline per SKU "
                         f"with hindsight (£{oracle:,.0f}).</b> The saving comes mainly from far fewer lost sales "
                         f"({A['ppo']['lost_u']:,} vs {A['sS']['lost_u']:,} units) at similar inventory."))
    story.append(Spacer(1, 6))
    rows = [["SKU", "Product", "Agent", "(s,S)", "Saving", "Fill agent", "Fill (s,S)"]]
    for _, r in base.iterrows():
        col = "#15803d" if r.ppo_beats_sS else "#b91c1c"
        rows.append([r.sku, short(desc.get(r.sku, ""), 30), f"£{r.ppo_cost:,.0f}", f"£{r.sS_cost:,.0f}",
                     f'<font color="{col}"><b>{r["saving_vs_sS_%"]:+.1f}%</b></font>', f"{r.ppo_fill:.1%}", f"{r.sS_fill:.1%}"])
    story.append(KeepTogether([P("Per-SKU results vs (s,S)", "h2"),
                               table(rows, [17 * mm, 58 * mm, 19 * mm, 19 * mm, 19 * mm, 21 * mm, W - 153 * mm])]))
    story.append(Spacer(1, 6))
    st = [["Scenario", "Agent cost", "(s,S) cost", "Saving", "Agent wins"]]
    for sc in ["demand -30%", "base", "demand +30%"]:
        r = stress.loc[sc]
        st.append([{"base": "Real demand"}.get(sc, sc.replace("demand", "Demand")), f"£{r.ppo:,.0f}", f"£{r.ss:,.0f}",
                   f"{100 * (r.ss - r.ppo) / r.ss:+.1f}%", f"{int(r.w)} / {n_sku}"])
    story.append(KeepTogether([P("Stress test (policies not re-tuned)", "h2"), table(st, [40 * mm] + [(W - 40 * mm) / 4] * 4)]))
    story.append(P("Reading the results honestly", "h2"))
    lose_txt = ", ".join(f"{r.sku} ({r['saving_vs_sS_%']:+.0f}%)" for _, r in losers.iterrows()) or "none"
    story += bullets([
        f"<b>Still losing to (s,S):</b> {lose_txt}. On steady, high-volume SKUs a tuned (s,S) is already near-optimal, "
        "and two tuning rounds did not close the gap.",
        "<b>The saving is concentrated:</b> two SKUs (22197, 84879) contribute a large share. There, (s,S) tuned on 2010 "
        "under-stocked the bigger 2011 peak, while the agent reacted to recent demand.",
        "<b>Lower demand hurts:</b> at −30% demand the agent wins only "
        f"{int(stress.loc['demand -30%'].w)} of {n_sku}; it was trained at historical levels and keeps slightly too much stock.",
        "<b>Run-to-run variance:</b> PPO results differ between seeds and even CPUs; this is why selection and promotion exist.",
        "<b>Assumed economics:</b> costs are configurable assumptions (unit cost = 60% of price, etc.), not the retailer's books.",
    ])
    story.append(CondPageBreak(90 * mm))
    story.append(P("Charts from the backtest", "h2"))
    story.append(img(ROOT / "results" / "cost_comparison.png"))
    story.append(P("Total cost per SKU on the test period, by policy (lower is better).", "center"))
    story.append(Spacer(1, 6))
    story.append(img(ROOT / "results" / "fill_rate_comparison.png"))
    story.append(P("Fill rate per SKU on the test period (higher is better).", "center"))
    story.append(CondPageBreak(90 * mm))
    story.append(img(ROOT / "results" / "stock_over_time.png", max_h=235 * mm))
    story.append(P("End-of-day stock through the test period, StockPilot (blue) vs (s,S) (orange), for each SKU.", "center"))

    # ---------------- 7 dashboard
    story.append(CondPageBreak(90 * mm))
    story.append(P("7  The dashboard", "h1"))
    story.append(P(f'Live at <link href="{APP}" color="#2563eb">{APP}</link>. A sidebar switches between PUBLIC and LIVE data; '
                   "four KPI cards summarise the backtest. The app never trains: it loads the committed models and runs inference."))
    story.append(img(ROOT / "docs" / "screenshot.png", max_h=110 * mm))
    story.append(Spacer(1, 6))
    story.append(table([
        ["Page", "What it shows"],
        ["Today's Orders", "Per SKU: StockPilot's order and the (s,S) order, on-hand, on-order, 14-day average demand, a "
                           "demand sparkline and the lead time. CSV download. New SKUs without an agent get (s,S) with a note."],
        ["Backtest", "Cost and fill-rate bar charts for all four policies, a per-SKU win/loss table, stock-over-time for any "
                     "SKU with real demand overlaid, cost breakdown by type, and the stress-test table."],
        ["What-if", "Sliders for lead time, holding cost, stockout penalty and demand scale. Re-simulates the saved agent and "
                    "(s,S) on the test period, optionally re-tuning (s,S) for the new conditions as the fair comparison."],
        ["About", "Problem statement, MDP definition, economics and how the pipeline works."],
    ], [30 * mm, W - 30 * mm]))
    story.append(Spacer(1, 4))
    story.append(P("If the live sheet is missing or invalid, the app shows a clear error listing every problem and falls back "
                   "to the public data, so it never shows a blank page."))

    # ---------------- 8 architecture
    story.append(CondPageBreak(90 * mm))
    story.append(P("8  Architecture", "h1"))
    story.append(architecture())
    story.append(Spacer(1, 6))
    story.append(P("Repository layout", "h2"))
    story.append(table([
        ["Path", "Contents"],
        ["stockpilot/", "data.py · env.py · simulate.py · baselines.py · train.py · evaluate.py · recommend.py · config.py"],
        ["app/streamlit_app.py", "The dashboard (inference only)"],
        ["tests/", "test_env.py · test_data.py · test_app.py (36 tests)"],
        ["configs/", "config.yaml (economics, env, PPO, per-SKU tuning) · demo.yaml (demo shop run)"],
        ["models/&lt;sku&gt;/", "ppo.zip · vecnormalize.pkl · meta.json · baselines.json (~0.6 MB per SKU, committed)"],
        ["results/", "metrics.csv · stress_metrics.csv · summary.csv · traces.csv · 3 PNG charts"],
        ["data/", "processed/ (cleaned UCI daily data) · sample/ (always-available fallback) · demo/ (synthetic shop)"],
        ["scripts/", "make_demo_data.py · build_portfolio_pdf.py (this document)"],
        [".github/workflows/", "retrain.yml (weekly + manual)"],
    ], [38 * mm, W - 38 * mm]))
    story.append(P("Design decisions", "h2"))
    story += bullets([
        "<b>Train offline, serve inference only.</b> Training needs minutes of CPU; Streamlit Cloud's free tier is small and "
        "restarts often. Models are small (~0.6 MB), so they are committed and loaded with caching.",
        "<b>Git as the model registry.</b> Every retrain is a commit with models and metrics, giving full history, diffs and "
        "one-click rollback without extra infrastructure.",
        "<b>Config over code.</b> Economics, environment, PPO settings and per-SKU tuning live in YAML; a second config "
        "(STOCKPILOT_CONFIG) runs the whole pipeline into separate folders.",
        "<b>One schema, three sources.</b> UCI, a Google Sheet and the demo CSV all become the same table, so nothing downstream branches on the source.",
    ])

    # ---------------- 9 tech stack
    story.append(CondPageBreak(90 * mm))
    story.append(P("9  Tech stack", "h1"))
    story.append(table([
        ["Layer", "Technology", "Used for"],
        ["Language", "Python 3.11", "Everything"],
        ["RL environment", "Gymnasium 1.4", "StockPilotEnv (standard reset/step API, Box observation, Discrete action)"],
        ["RL algorithm", "Stable-Baselines3 2.9 (PPO)", "Training, VecNormalize, DummyVecEnv, callbacks for checkpoint selection"],
        ["Deep learning", "PyTorch 2.15 (CPU wheel)", "Policy and value networks; CPU-only build keeps the deploy small"],
        ["Data", "pandas 2.3 · NumPy 2.4 · PyArrow · openpyxl", "Reading the 45 MB Excel file once, parquet cache, cleaning, aggregation"],
        ["Live data", "requests · gspread · google-auth", "Published-CSV fetch or service-account sheet access"],
        ["Config", "PyYAML", "configs/*.yaml"],
        ["Charts", "Plotly 7 (app) · Matplotlib 3.11 (reports)", "Interactive dashboard charts; PNGs for README and this PDF"],
        ["Web app", "Streamlit 1.67", "Four-page dashboard, caching, secrets, CSV download"],
        ["Testing", "pytest 9 · Streamlit AppTest", "Environment invariants, cost arithmetic, validation, headless page rendering"],
        ["CI / CD", "GitHub Actions", "Weekly retrain, tests, commit-back"],
        ["Hosting", "Streamlit Community Cloud", "Public app; redeploys on every push to main"],
        ["Docs", "Mermaid (README) · ReportLab (this PDF)", "Architecture diagram and portfolio document"],
    ], [28 * mm, 52 * mm, W - 80 * mm]))
    story.append(P("Exact versions are pinned in requirements.txt. On Linux, torch is installed from the official CPU-only wheel "
                   "URL so the cloud build does not pull CUDA libraries.", "small"))

    # ---------------- 10 MLOps
    story.append(P("10  MLOps: automation and retraining", "h1"))
    story.append(P("<font name='Mono'>.github/workflows/retrain.yml</font> runs every Monday at 03:00 UTC and on demand:"))
    story.append(P("pytest → pull latest data (live sheet if the LIVE_CSV_URL secret is set, else UCI) → tune baselines → train 3 "
                   "challenger seeds per SKU, 4 SKUs in parallel → promote only if cheaper on the training period → backtest → "
                   "commit models/ and results/ → push → Streamlit Cloud redeploys.", "code"))
    story.append(P("Champion/challenger promotion", "h2"))
    story.append(P("The first automated retrain exposed a real risk: with the same seeds on a different CPU, PPO produced agents "
                   "that beat (s,S) on only 4 of 10 SKUs (+15% cost), and the workflow blindly overwrote better ones. Now a "
                   "retrained agent replaces the deployed one only if it is cheaper on the <b>training period</b>. Applied to "
                   "the second CI run, it promoted 2 agents and kept the rest. The rule never looks at test results, so it can "
                   "occasionally keep an agent that turns out worse on test; that is the price of an honest evaluation."))
    story.append(P("Failure handling", "h2"))
    story.append(P("Each SKU trains in its own process; the workflow records every process ID and fails with the SKU name and the "
                   "last 30 log lines if any training fails, so a partial retrain cannot pass silently."))

    # ---------------- 11 testing
    story.append(CondPageBreak(90 * mm))
    story.append(P("11  Testing and quality assurance", "h1"))
    story.append(table([
        ["Test file", "Tests", "What is checked"],
        ["tests/test_env.py", "23", "Stock never negative (lead times 0/1/3/7, shelf life on/off, random policies, many seeds); "
                                    "orders arrive after exactly L days for L = 0, 1, 2, 3, 7, 14; L = 0 serves the same day; "
                                    "cost arithmetic against a hand-worked 3-day example; profit identity; FIFO expiry and waste "
                                    "cost; observation shape constant across lead times; action scaling; reward scale; exact replay."],
        ["tests/test_data.py", "5", "Valid schema passes; missing columns, bad dates, negative and non-numeric values, duplicates "
                                    "are all reported; chronological split has no overlap."],
        ["tests/test_app.py", "8", "Every dashboard page renders headlessly in PUBLIC and LIVE mode with no exception; an "
                                   "unreachable sheet shows a clear error and falls back."],
    ], [32 * mm, 13 * mm, W - 45 * mm]))
    story.append(P("Beyond unit tests", "h2"))
    story += bullets([
        "Every pipeline step was run on real data and its output inspected before moving on.",
        "Diagnostic scripts compared agent and (s,S) behaviour (inventory, order sizes, response to stock) to find root causes, "
        "not just symptoms.",
        "The deployed app was clicked through page by page after each deploy; its recommendations matched the local run exactly "
        "(7 SKUs, 6,952 units), confirming identical inference on Linux.",
        "A synthetic shop with a deliberately broken copy tests LIVE mode end to end, including all validation messages.",
        "README numbers were recomputed from the results files after every change; one wrong hand-written figure was caught this way.",
    ])

    # ---------------- 12 challenges
    story.append(P("12  Engineering challenges and how they were solved", "h1"))
    ch = [
        ["Problem", "Root cause", "Fix"],
        ["PPO 10-50× worse than (s,S)", "Random exploration flooded stock; agent learned a constant order that ignored stock",
         "Start with P(order 0) = 0.9; add inventory position as a feature"],
        ["Agent overstocked ~100 days of demand", "Profit reward noise (margin × lumpy demand) hid the holding-cost signal",
         "Cost-only reward: same optimum, far lower variance"],
        ["Unstable learning curves", "PPO oscillates between updates", "Keep the checkpoint with the lowest training-period cost"],
        ["Baselines looked weaker than they were", "Grid optima on the grid edge", "Wider coarse grid, then fine search around the best"],
        ["Original economics made RL lose by design", "At £15/order the optimal cycle was ~36 days, beyond the action range",
         "Re-based to £2/order and 100%/yr carrying cost; all configurable and stated"],
        ["CI retrain deployed worse agents", "Run-to-run variance + blind overwrite", "Champion/challenger promotion, 3 seeds"],
        ["Half the CI SKUs silently not retrained", "Parallel processes shared and deleted one staging folder; `wait` hid failures",
         "Per-SKU staging; workflow checks every exit code"],
        ["pandas DLL blocked on Windows", "Smart App Control blocked pandas 3.1.1", "Pinned pandas 2.3.3"],
        ["Charts showed '20k, 40k' for SKUs", "Plotly read numeric-looking SKU strings as numbers", "Force a category axis"],
        ["App not found after 'deploy'", "Default main-file path; app lives in app/", "Set app/streamlit_app.py"],
    ]
    story.append(table(ch, [42 * mm, 62 * mm, W - 104 * mm]))

    # ---------------- 13 limitations
    story.append(CondPageBreak(90 * mm))
    story.append(P("13  Limitations and future work", "h1"))
    story.append(table([
        ["Limitation", "Possible next step"],
        ["Sales are treated as demand; stockout days under-count true demand (censoring)", "Demand un-censoring, e.g. impute demand on stockout days"],
        ["One agent per SKU; no sharing across products", "One shared policy with SKU features, enabling cold start for new SKUs"],
        ["Fixed lead time per SKU", "Stochastic lead times and supplier reliability in the environment"],
        ["Assumed cost parameters", "Calibrate with a retailer's real costs; sensitivity analysis is already possible on the What-if page"],
        ["Weaker when demand drops sharply", "Train on scaled demand (domain randomisation) so the agent sees ±30% regimes"],
        ["No supplier constraints", "Minimum order quantities, pack sizes, shared budget or warehouse capacity across SKUs"],
        ["Steady SKUs still lose to (s,S)", "Hybrid: let the agent adjust (s,S) parameters instead of raw quantities"],
        ["Single test period", "Rolling-origin backtests over several cut-off dates for confidence intervals"],
    ], [W / 2, W / 2]))

    # ---------------- 14 Q&A
    story.append(P("14  Interview Q&amp;A", "h1"))
    qa = [
        ("Overview", [
            ("In one sentence, what is StockPilot?",
             "A reinforcement-learning system that decides how many units of each product to order today to minimise holding, "
             "ordering, lost-sales and waste cost, proven against classic inventory rules on held-out real retail data and "
             "deployed as a live dashboard with weekly retraining."),
            ("Who would use it?",
             "A small or mid-sized retailer or wholesaler that tracks daily sales and stock in a spreadsheet and reorders by "
             "rule of thumb. They keep their sheet; StockPilot reads it and suggests today's orders."),
            ("What is the headline result?",
             f"On the last 25% of dates, which the models never saw, the agent beat a tuned (s,S) policy on {wins} of {n_sku} "
             f"SKUs with {pct:.1f}% lower total cost and a higher fill rate ({A['ppo']['fill']:.1%} vs {A['sS']['fill']:.1%})."),
        ]),
        ("Data", [
            ("Why the UCI Online Retail II dataset?",
             "It is real, public, daily-granular retail transaction data over two years with thousands of products, so results "
             "are reproducible by anyone. Its lumpy wholesale demand makes it a hard, realistic test."),
            ("How did you clean it?",
             "Removed cancellations and negative quantities, and also the original sales they reversed; kept real product codes "
             "only; aggregated to daily units on the store's trading-day calendar; chose the top 10 SKUs with enough history."),
            ("How do you prevent data leakage?",
             "A single chronological cut: 75% of dates for training and tuning, 25% for testing. Demand statistics, baseline "
             "parameters, agent training and every model-selection decision use the training slice only. The test-period "
             "history window starts from the last training days."),
            ("The dataset has no stock levels. Is that a problem?",
             "Not for the evaluation: the backtest simulates stock from the same starting point for every policy and uses only "
             "real demand. Simulated on-hand only appears in the sample data used for the Today's Orders demo."),
        ]),
        ("Reinforcement learning", [
            ("Why RL instead of a forecast plus a rule?",
             "A forecast-plus-rule system optimises forecast error, not cost. RL optimises total cost directly, handles lead "
             "times and lost sales in one objective, and learns how much to react to recent demand. The baselines include a "
             "forecast plus safety stock rule, which the agent beat clearly."),
            ("Why PPO?",
             "It is stable, works well with discrete actions and vectorised environments, needs little tuning and has a mature, "
             "well-tested implementation in Stable-Baselines3."),
            ("Why a discrete action space?",
             "Order quantities are naturally discrete; a small set of demand-scaled sizes makes exploration easier than a "
             "continuous action and lets one design fit any SKU volume."),
            ("Why is the reward cost and not profit?",
             "Profit = margin × demand − cost, and margin × demand does not depend on the agent, so both rewards have the same "
             "optimal policy. The cost-only version removes a huge noise term, which made learning work."),
            ("What was the hardest RL problem?",
             "Exploration. Random actions flooded the warehouse, so the agent never saw the states that matter and learned to "
             "ignore stock. Biasing the initial policy towards ordering nothing fixed it."),
            ("How do you train on only about 450 days of history?",
             "With a 7-day block bootstrap: random week-long chunks of real history are stitched into many 180-day episodes, "
             "preserving weekday patterns and short-term correlation. The agent sees hundreds of thousands of plausible days."),
            ("Why VecNormalize?",
             "Observations and rewards differ by orders of magnitude between SKUs; normalising them keeps network inputs and "
             "advantage estimates well-scaled. The saved statistics are reused at inference."),
        ]),
        ("Evaluation and honesty", [
            ("How do you know the agent is really better and not overfitted?",
             "Every choice uses training-period cost; the test period is used once, for reporting. Baselines get the same "
             "treatment and a wide search. Results are also shown against the best baseline per SKU with hindsight and under "
             "±30% demand."),
            ("Where does it fail?",
             f"It still loses on {lose_txt}, it wins only {int(stress.loc['demand -30%'].w)} of {n_sku} SKUs when demand "
             "drops 30%, and much of the saving comes from two SKUs where (s,S) under-stocked a seasonal peak."),
            ("Why not pick the agent that scored best on the test set?",
             "That would leak test information into model selection and overstate real performance. On one SKU this rule "
             "kept an agent that did worse on test; that is the cost of an honest number."),
        ]),
        ("Engineering and deployment", [
            ("Why commit model files to Git?",
             "They are small (~0.6 MB per SKU). Git gives versioning, diffs, rollback and a single source of truth for the "
             "app and CI without a separate model registry."),
            ("How does retraining avoid making things worse?",
             "Champion/challenger: 3 seeds per SKU, the best by training cost is promoted only if it beats the current agent "
             "on the training period. The workflow also fails if any SKU's training fails."),
            ("How is the app kept fast on a free tier?",
             "No training in the app; models and data are cached with Streamlit caching; CPU-only torch keeps the build small."),
            ("How did you test it?",
             "36 pytest tests (environment invariants, exact cost arithmetic, schema validation, every dashboard page in both "
             "modes), plus end-to-end runs on real and synthetic data and manual checks of the deployed app."),
            ("How would you scale it to 10,000 SKUs?",
             "Train one shared policy conditioned on SKU features instead of one per SKU, run training as a batch job on a "
             "larger runner or cloud instance, store models in object storage, and serve recommendations as a nightly batch "
             "rather than per page load."),
        ]),
    ]
    for section, items in qa:
        story.append(CondPageBreak(40 * mm))
        story.append(P(section, "h2"))
        for q, a in items:
            story.append(KeepTogether([P(f"Q. {q}", "q"), P(a, "a")]))

    # ---------------- 15 portfolio
    story.append(CondPageBreak(90 * mm))
    story.append(P("15  Portfolio and CV summary", "h1"))
    story.append(P("Project blurb (portfolio / LinkedIn)", "h2"))
    story.append(callout(f"<b>StockPilot: autopilot for inventory.</b> I built and deployed a reinforcement-learning system that "
                         f"recommends daily order quantities per product. Trained with PPO on real retail data (UCI Online Retail II, "
                         f"1M+ transactions), it beat a tuned (s,S) policy on {wins} of {n_sku} products in a held-out backtest, "
                         f"cutting total inventory cost by {pct:.0f}% while raising the fill rate to {A['ppo']['fill']:.1%}. "
                         "It runs as a live Streamlit dashboard reading a Google Sheet, and retrains weekly through GitHub "
                         "Actions with a champion/challenger gate."))
    story.append(P("CV bullet points", "h2"))
    story += bullets([
        f"Built an end-to-end RL inventory system (Gymnasium, Stable-Baselines3 PPO, PyTorch) that cut held-out inventory cost by "
        f"{pct:.0f}% versus a tuned (s,S) policy on real retail data, winning on {wins}/{n_sku} SKUs.",
        "Engineered a leakage-free evaluation: chronological split, training-only tuning and model selection, ±30% demand "
        "stress tests, comparison against a hindsight oracle of classical baselines.",
        "Diagnosed and fixed PPO failure modes (exploration collapse, reward variance, unstable checkpoints) through targeted "
        "experiments, turning a 10-50× cost gap into a win.",
        "Shipped a 4-page Streamlit dashboard with live Google Sheets integration and schema validation, deployed on "
        "Streamlit Community Cloud.",
        "Automated weekly retraining with GitHub Actions and a champion/challenger promotion gate after detecting that a "
        "naive retrain silently degraded models.",
        "Wrote 36 pytest tests covering simulator invariants, cost arithmetic, data validation and headless UI rendering.",
    ])
    story.append(P("Skills demonstrated", "h2"))
    story.append(table([
        ["Area", "Evidence in this project"],
        ["Reinforcement learning", "MDP design, reward shaping, exploration fix, PPO tuning, VecNormalize, seed variance"],
        ["Operations research", "(s,S), base-stock and safety-stock policies; lost-sales inventory with lead times"],
        ["Data engineering", "Cleaning 1M+ transactions, schema validation, live sheet ingestion, synthetic data generation"],
        ["ML evaluation", "Leakage-free backtesting, stress tests, honest reporting of losses"],
        ["Software engineering", "Modular package, config-driven design, 36 tests, pinned dependencies"],
        ["MLOps / deployment", "GitHub Actions retraining, promotion gate, Git-based model registry, Streamlit Cloud"],
        ["Communication", "README, Mermaid architecture, dashboard UX, this document"],
    ], [38 * mm, W - 38 * mm]))
    story.append(Spacer(1, 10))
    story.append(P(f'Live app: <link href="{APP}" color="#2563eb">{APP}</link><br/>'
                   f'Code: <link href="{REPO}" color="#2563eb">{REPO}</link>', "small"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm,
                            bottomMargin=16 * mm, title="StockPilot: project portfolio & Q&A",
                            author="Abdullah-Akmal", subject=cfg["project"]["tagline"])
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(build())
