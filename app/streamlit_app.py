"""StockPilot dashboard. Loads trained agents from models/ and results from results/; never trains."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from stockpilot.baselines import BASELINES, make_policy, tune  # noqa: E402
from stockpilot.config import load_config, resolve  # noqa: E402
from stockpilot.data import DataError, load_dataset, sku_contexts  # noqa: E402
from stockpilot.recommend import recommend  # noqa: E402
from stockpilot.simulate import simulate  # noqa: E402
from stockpilot.train import load_agent  # noqa: E402

st.set_page_config(page_title="StockPilot", page_icon="📦", layout="wide")
CFG = load_config()
LABELS = {"ppo": "StockPilot PPO", "sS": "(s,S)", "base_stock": "Base-stock", "ma_safety": "MA + safety stock"}
COLORS = {"ppo": "#2563eb", "sS": "#f97316", "base_stock": "#64748b", "ma_safety": "#a3a3a3"}

st.markdown("""
<style>
.block-container {padding-top: 3.2rem; max-width: 1400px;}
.sp-header {display:flex; align-items:baseline; gap:.9rem; flex-wrap:wrap; margin-bottom:.2rem;}
.sp-title {font-size:2.3rem !important; line-height:1.1; font-weight:800; letter-spacing:-.02em; color:#0f172a; margin:0;}
.sp-title span {color:#2563eb;}
.sp-tag {color:#475569; font-size:1.05rem;}
.sp-kpi {background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:1rem 1.2rem; height:100%;
         box-shadow:0 1px 2px rgba(15,23,42,.04);}
.sp-kpi .lbl {color:#64748b; font-size:.74rem; text-transform:uppercase; letter-spacing:.05em; font-weight:600;}
.sp-kpi .val {font-size:clamp(1.2rem, 2.1vw, 1.9rem); white-space:nowrap; font-weight:750; color:#0f172a; margin-top:.15rem; font-variant-numeric: tabular-nums;}
.sp-kpi .sub {color:#64748b; font-size:.85rem;}
.sp-kpi .good {color:#15803d; font-weight:600;} .sp-kpi .bad {color:#b91c1c; font-weight:600;}
.sp-pill {display:inline-block; padding:.12rem .6rem; border-radius:999px; font-size:.78rem; font-weight:600;
          background:#e0e7ff; color:#3730a3; margin-left:.3rem;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner=False)
def cached_agent(path: str):
    return load_agent(Path(path))


@st.cache_data(show_spinner=False, ttl=600)
def cached_live(url, sa_json, sheet_key, worksheet):
    from stockpilot.data import load_live
    return load_live(url, json.loads(sa_json) if sa_json else None, sheet_key, worksheet)


@st.cache_data(show_spinner=False)
def cached_public():
    return load_dataset("public", CFG)


@st.cache_data(show_spinner=False)
def read_results():
    r = resolve(CFG["paths"]["results_dir"])
    out = {}
    for name in ["metrics", "stress_metrics", "summary", "traces"]:
        p = r / f"{name}.csv"
        out[name] = pd.read_csv(p, dtype={"sku": str}) if p.exists() else None
    return out


def descriptions() -> dict:
    p = resolve("data/processed/sku_descriptions.json")
    return json.loads(p.read_text()) if p.exists() else {}


def secret(section: str, key: str | None = None):
    try:
        v = st.secrets[section]
        return v[key] if key else v
    except Exception:
        return None


# --------------------------------------------------------------------------- sidebar
st.sidebar.markdown("### 📦 StockPilot")
mode = st.sidebar.radio("Data source", ["PUBLIC", "LIVE"],
                        index=0 if CFG["data"]["mode"] == "public" else 1,
                        help="PUBLIC: UCI Online Retail II (real UK retailer). LIVE: your Google Sheet.")
page = st.sidebar.radio("Page", ["Today's Orders", "Backtest", "What-if", "About"])

df, source, banner = None, None, None
if mode == "LIVE":
    url = secret("live", "csv_url")
    sa = secret("gcp_service_account")
    sheet_key, ws = secret("live", "sheet_key"), secret("live", "worksheet")
    try:
        df = cached_live(url, json.dumps(dict(sa)) if sa else None, sheet_key, ws)
        source = "live"
    except DataError as e:
        st.sidebar.error("LIVE data unavailable")
        banner = ("error", f"**LIVE mode could not load your sheet.**\n\n{e}\n\n"
                           "Showing PUBLIC data instead. Add `live.csv_url` in the app's Secrets "
                           "(see README > Deploy).")
    except Exception as e:  # network, auth, parsing
        banner = ("error", f"**LIVE mode failed:** {type(e).__name__}: {e}. Showing PUBLIC data instead.")
if df is None:
    ds = cached_public()
    df, source = ds.df, ds.source
    if ds.message:
        banner = ("warning", ds.message)

sku_all = sorted(df["sku"].unique())
desc = descriptions()
st.sidebar.caption(f"Source: **{source}** · {len(sku_all)} SKUs · "
                   f"{df['date'].min().date()} → {df['date'].max().date()}")
if st.sidebar.button("Reload data"):
    st.cache_data.clear()
    st.rerun()

# --------------------------------------------------------------------------- header + KPIs
st.markdown(f"""<div class="sp-header"><div class="sp-title">Stock<span>Pilot</span></div>
<span class="sp-tag">{CFG['project']['tagline']}</span></div>""", unsafe_allow_html=True)
if banner:
    getattr(st, banner[0])(banner[1])

res = read_results()


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="sp-kpi"><div class="lbl">{label}</div><div class="val">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


m = res["metrics"]
if m is not None:
    ppo, ss = m[m.policy == "ppo"], m[m.policy == "sS"]
    saved = ss.total_cost.sum() - ppo.total_cost.sum()
    pct = 100 * saved / ss.total_cost.sum()
    fill_ppo = ppo.demand_units.sub(ppo.lost_units).sum() / ppo.demand_units.sum()
    fill_ss = ss.demand_units.sub(ss.lost_units).sum() / ss.demand_units.sum()
    wins = int((res["summary"].query("scenario == 'base'").ppo_beats_sS).sum())
    c1, c2, c3, c4 = st.columns(4)
    cls = "good" if saved >= 0 else "bad"
    kpi(c1, "Cost saved vs (s,S)", f"£{saved:,.0f}",
        f'<span class="{cls}">{pct:+.1f}%</span> on the held-out test period')
    kpi(c2, "Fill rate (StockPilot)", f"{fill_ppo:.1%}", f"(s,S): {fill_ss:.1%} · target {CFG['costs']['target_fill_rate']:.0%}")
    kpi(c3, "Avg inventory (StockPilot)", f"{ppo.avg_inventory.sum():,.0f}",
        f"(s,S): {ss.avg_inventory.sum():,.0f} units across SKUs")
    kpi(c4, "SKUs where agent wins", f"{wins} / {len(ppo)}", "lower total cost than (s,S)")
else:
    st.info("No backtest results yet. Run `python -m stockpilot.evaluate`.")
st.write("")


# --------------------------------------------------------------------------- pages
@st.cache_data(show_spinner=False, ttl=600)
def cached_recommend(_df, source_key: str):
    return recommend(_df, CFG, cached_agent, desc)


def page_today():
    st.subheader("Today's Orders")
    st.caption("Recommended order quantity for each SKU, decided from the latest row of data "
               "(end-of-day stock and open orders). The classic (s,S) policy's order is shown next to it.")
    with st.spinner("Loading agents and computing today's orders..."):
        rec = cached_recommend(df, source)
    if rec.empty:
        st.warning("No SKUs to show.")
        return
    as_of = rec["as_of"].dropna().max() if "as_of" in rec else ""
    st.markdown(f"Data as of **{as_of}** · orders below are for the next trading day.")
    view = rec[["sku", "description", "stockpilot_order", "sS_order", "on_hand", "on_order", "avg_demand_14d",
                "recent_demand", "lead_time", "note"]]
    st.dataframe(
        view, hide_index=True, use_container_width=True,
        column_config={
            "sku": "SKU", "description": st.column_config.TextColumn("Product", width="medium"), "on_hand": st.column_config.NumberColumn("On hand", format="%d"),
            "on_order": st.column_config.NumberColumn("On order", format="%d"),
            "position": st.column_config.NumberColumn("Position", format="%d", help="On hand + on order"),
            "avg_demand_14d": st.column_config.NumberColumn("Avg demand (14d)", format="%.1f"),
            "lead_time": st.column_config.NumberColumn("Lead time (d)", format="%d"),
            "recent_demand": st.column_config.LineChartColumn("Last 14 days demand"),
            "stockpilot_order": st.column_config.NumberColumn("🚀 StockPilot order", format="%d"),
            "sS_order": st.column_config.NumberColumn("(s,S) order", format="%d"),
            "note": "Note",
        })
    n_order = int((rec.stockpilot_order.fillna(0) > 0).sum())
    st.caption(f"StockPilot places orders for {n_order} of {len(rec)} SKUs today "
               f"({int(rec.stockpilot_order.fillna(0).sum()):,} units); (s,S) for "
               f"{int((rec.sS_order.fillna(0) > 0).sum())} ({int(rec.sS_order.fillna(0).sum()):,} units).")
    csv = rec.drop(columns=["recent_demand"]).to_csv(index=False).encode()
    st.download_button("⬇ Download orders (CSV)", csv, file_name=f"stockpilot_orders_{as_of}.csv", mime="text/csv")


def page_backtest():
    st.subheader("Backtest on held-out data")
    if m is None:
        st.info("Run `python -m stockpilot.evaluate` to produce results.")
        return
    split = f"Policies tuned/trained on the first {int(CFG['data']['train_frac'] * 100)}% of dates; " \
            f"scored here on the last {100 - int(CFG['data']['train_frac'] * 100)}% with real demand."
    st.caption(split + f" Results source: **{m.data_source.iloc[0]}** data.")
    s = res["summary"].query("scenario == 'base'")
    pols = [p for p in LABELS if p in m.policy.unique()]

    c1, c2 = st.columns(2)
    for col, metric, title in [(c1, "total_cost", "Total cost (GBP, lower is better)"),
                               (c2, "fill_rate", "Fill rate (higher is better)")]:
        fig = go.Figure()
        for p in pols:
            d = m[m.policy == p]
            fig.add_bar(name=LABELS[p], x=d.sku, y=d[metric], marker_color=COLORS[p])
        fig.update_layout(barmode="group", title=title, height=380, margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=-0.15), plot_bgcolor="white")
        fig.update_xaxes(type="category", title="SKU")   # SKU codes look numeric; keep them as labels
        if metric == "fill_rate":
            fig.update_yaxes(range=[max(0, m.fill_rate.min() - 0.03), 1.0], tickformat=".0%")
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("**Per-SKU result vs (s,S)**")
    tbl = s[["sku", "ppo_cost", "sS_cost", "saving_vs_sS_%", "ppo_fill", "sS_fill", "ppo_beats_sS"]].copy()
    tbl.insert(1, "product", tbl.sku.map(desc).fillna(""))
    st.dataframe(tbl, hide_index=True, use_container_width=True, column_config={
        "ppo_cost": st.column_config.NumberColumn("StockPilot cost", format="£%.0f"),
        "sS_cost": st.column_config.NumberColumn("(s,S) cost", format="£%.0f"),
        "saving_vs_sS_%": st.column_config.NumberColumn("Saving", format="%.1f%%"),
        "ppo_fill": st.column_config.NumberColumn("StockPilot fill", format="%.3f"),
        "sS_fill": st.column_config.NumberColumn("(s,S) fill", format="%.3f"),
        "ppo_beats_sS": st.column_config.CheckboxColumn("Agent wins")})

    st.markdown("**Stock over time: StockPilot vs (s,S)**")
    tr = res["traces"]
    sku = st.selectbox("SKU", sorted(tr.sku.unique()), format_func=lambda k: f"{k} · {desc.get(k, '')}")
    fig = go.Figure()
    for p in ["sS", "ppo"]:
        t = tr[(tr.sku == sku) & (tr.policy == p)]
        fig.add_scatter(x=t.date, y=t.on_hand_end, name=f"{LABELS[p]} stock", line=dict(color=COLORS[p], width=2))
    t = tr[(tr.sku == sku) & (tr.policy == "ppo")]
    fig.add_bar(x=t.date, y=t.demand, name="Real demand", marker_color="rgba(15,23,42,.18)")
    lost = tr[(tr.sku == sku)].groupby("policy").lost.sum()
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.08), yaxis_title="Units")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Lost units on this SKU: StockPilot {int(lost.get('ppo', 0)):,} · (s,S) {int(lost.get('sS', 0)):,}")

    st.markdown("**Cost breakdown (all SKUs)**")
    br = m.groupby("policy")[["holding_cost", "ordering_cost", "lost_sales_cost", "waste_cost"]].sum().loc[pols]
    fig = go.Figure()
    for c, colr in zip(br.columns, ["#93c5fd", "#fdba74", "#fca5a5", "#d4d4d8"]):
        fig.add_bar(name=c.replace("_cost", "").replace("_", " "), x=[LABELS[p] for p in br.index], y=br[c],
                    marker_color=colr)
    fig.update_layout(barmode="stack", height=340, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                      yaxis_title="GBP")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Stress test (policies not re-tuned)**")
    sm = res["summary"]
    agg = sm.groupby("scenario").agg(ppo_cost=("ppo_cost", "sum"), sS_cost=("sS_cost", "sum"),
                                     wins=("ppo_beats_sS", "sum"), skus=("sku", "count")).reindex(
        ["demand -30%", "base", "demand +30%"]).reset_index()
    agg["saving_%"] = (100 * (agg.sS_cost - agg.ppo_cost) / agg.sS_cost).round(1)
    agg["agent wins"] = agg.wins.astype(int).astype(str) + " / " + agg.skus.astype(str)
    st.dataframe(agg[["scenario", "ppo_cost", "sS_cost", "saving_%", "agent wins"]], hide_index=True,
                 use_container_width=True, column_config={
                     "ppo_cost": st.column_config.NumberColumn("StockPilot cost", format="£%.0f"),
                     "sS_cost": st.column_config.NumberColumn("(s,S) cost", format="£%.0f"),
                     "saving_%": st.column_config.NumberColumn("Saving", format="%.1f%%")})


@st.cache_data(show_spinner=False)
def whatif_run(sku: str, lead_time: int, holding_rate: float, penalty: float, demand_scale: float, retune: bool):
    ds = cached_public() if source != "live" else None
    data = df if ds is None else ds.df
    ctx = next(c for c in sku_contexts(data, CFG) if c.sku == sku)
    over = {"holding_cost": holding_rate / 365 * ctx.params["unit_cost"], "stockout_penalty": penalty}
    rep = ctx.test_replay(int(CFG["env"]["history_len"]))
    d = resolve(CFG["paths"]["models_dir"]) / sku
    bl = json.loads((d / "baselines.json").read_text())
    pols = {}
    agent = cached_agent(str(d))
    if agent:
        pols["ppo"] = agent[0]
    pols["sS"] = make_policy("sS", bl["sS"]["params"])
    if retune:
        prm, _ = tune("sS", ctx.params, ctx.train_replay(), CFG["costs"]["target_fill_rate"], over, lead_time)
        pols["sS_retuned"] = make_policy("sS", prm)
    out, traces = [], {}
    for name, pol in pols.items():
        mm, tr = simulate(pol, ctx.params, rep["demand"], rep["dows"], rep["history"], lead_time=lead_time,
                          demand_scale=demand_scale, overrides=over, trace=True)
        tr["date"] = rep["dates"]
        out.append({"policy": name, **mm})
        traces[name] = tr
    return pd.DataFrame(out), traces, ctx.params


def page_whatif():
    st.subheader("What-if")
    st.caption("Change the operating conditions and re-run the saved policies on the test period's real demand. "
               "The agent is not retrained; (s,S) can optionally be re-tuned on the training period for the new "
               "conditions, which is the fairest comparison.")
    if m is None:
        st.info("Train and evaluate first.")
        return
    skus = sorted(m.sku.unique())
    c0, c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1, 1])
    sku = c0.selectbox("SKU", skus, format_func=lambda k: f"{k} · {desc.get(k, '')[:24]}")
    meta_p = cached_agent(str(resolve(CFG["paths"]["models_dir"]) / sku))
    base_p = meta_p[1]["params"] if meta_p else None
    L0 = int(base_p["lead_time"]) if base_p else int(CFG["costs"]["lead_time_days"])
    lt = c1.slider("Lead time (days)", 0, 14, L0)
    hr = c2.slider("Holding cost (%/yr of unit cost)", 10, 300, int(CFG["costs"]["holding_rate_annual"] * 100), 10)
    pen0 = float(base_p["stockout_penalty"]) if base_p else 1.0
    pen = c3.slider("Stockout penalty (£/lost unit)", 0.0, round(max(pen0 * 4, 1.0), 2), round(pen0, 2), 0.01)
    scale = c4.slider("Demand scale", 0.5, 1.5, 1.0, 0.05)
    retune = st.toggle("Also re-tune (s,S) for these conditions (train period only)", value=True)
    if lt != L0:
        st.info(f"The agent was trained with a {L0}-day lead time. It observes the lead time and its pipeline, "
                "but other lead times are outside its training distribution.")
    with st.spinner("Simulating..."):
        out, traces, _ = whatif_run(sku, lt, hr / 100, pen, scale, retune)
    lab = {**LABELS, "sS_retuned": "(s,S) re-tuned"}
    out["Policy"] = out.policy.map(lab)
    cols = st.columns(len(out))
    for col, (_, r) in zip(cols, out.iterrows()):
        kpi(col, r.Policy, f"£{r.total_cost:,.0f}",
            f"fill {r.fill_rate:.1%} · avg inv {r.avg_inventory:,.0f} · {int(r.n_orders)} orders")
    fig = go.Figure()
    pal = {"ppo": COLORS["ppo"], "sS": COLORS["sS"], "sS_retuned": "#a855f7"}
    for name, tr in traces.items():
        fig.add_scatter(x=tr.date, y=tr.on_hand_end, name=lab[name], line=dict(color=pal[name], width=2))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.08), yaxis_title="End-of-day stock")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(out[["Policy", "total_cost", "holding_cost", "ordering_cost", "lost_sales_cost", "fill_rate",
                      "avg_inventory", "n_orders", "lost_units"]].round(3), hide_index=True, use_container_width=True)


def page_about():
    st.subheader("About StockPilot")
    st.markdown(f"""
**Problem.** Every day, for every product, a store must decide how much to order. Order too little and customers
leave empty-handed (lost sales); order too much and cash sits on shelves (holding cost) or expires (waste); order
too often and fixed ordering costs add up. StockPilot learns, per SKU, an ordering policy that minimises
**holding + ordering + lost-sales + waste cost** while keeping the fill rate near the
{CFG['costs']['target_fill_rate']:.0%} target, and proves it against classic rules on held-out real data.

**MDP (one per SKU).**

| | |
|---|---|
| **State** | on-hand stock, the order pipeline (up to 14 days, padded), lead time, inventory position, last 14 days of demand (÷ mean), day of week |
| **Action** | discrete order quantity: {CFG['env']['n_actions']} levels from 0 to {CFG['env']['action_max_mult']}× mean lead-time demand (4× for SKUs tuned that way in `sku_overrides`) |
| **Transition** | order arrives after exactly *L* days · demand served FIFO from stock · unmet demand is lost · optional shelf-life expiry |
| **Reward** | −(holding + fixed ordering + stockout penalty + waste) ÷ (margin × mean demand). This equals daily profit minus margin × demand, a term the agent cannot affect, so it has the same optimum as profit with far lower variance. |
| **Demand** | training: 7-day block bootstrap of the training period's real sales; evaluation: exact replay of real held-out sales |

**Economics (configurable in `configs/config.yaml`).** Unit cost = {CFG['costs']['unit_cost_frac']:.0%} of price;
holding = {CFG['costs']['holding_rate_annual']:.0%}/yr of unit cost; £{CFG['costs']['order_cost']:.2f} per order;
stockout penalty = lost margin + {CFG['costs']['goodwill_frac']:.0%} of price (goodwill); lead time {CFG['costs']['lead_time_days']} days.

**How it works.**
1. `stockpilot/data.py` reads the live Google Sheet (or UCI Online Retail II), validates and cleans it.
2. Baselines, (s,S), base-stock and moving-average + safety stock, are grid-searched on the training period only.
3. `stockpilot/train.py` trains PPO (Stable-Baselines3, VecNormalize, fixed seeds, CPU) per SKU; checkpoints are selected by training-period cost only.
4. `stockpilot/evaluate.py` replays the last 25% of dates with real demand (plus ±30% stress tests).
5. This app loads the saved agents and only runs inference. A weekly GitHub Action retrains and commits new models.

**Data.** PUBLIC mode uses the [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
dataset (a UK online giftware wholesaler, 2009-12 to 2011-12; CC BY 4.0). Cancellations and the sales they reverse
are removed, and daily units are aggregated on the store's trading-day calendar. The dataset has no inventory records,
so historical on-hand in the sample is simulated; the backtests use only the real demand.
""")


{"Today's Orders": page_today, "Backtest": page_backtest, "What-if": page_whatif, "About": page_about}[page]()
st.markdown("<br><div style='color:#94a3b8;font-size:.8rem'>StockPilot · PPO agents trained offline · "
            "never trains inside the app</div>", unsafe_allow_html=True)
