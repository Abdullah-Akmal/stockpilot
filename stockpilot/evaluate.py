"""Backtest every policy on the held-out TEST period using real demand (plus +/-30% stress tests)."""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .baselines import BASELINES, make_policy  # noqa: E402
from .config import load_config, resolve  # noqa: E402
from .data import load_dataset, sku_contexts  # noqa: E402
from .simulate import simulate  # noqa: E402
from .train import load_agent  # noqa: E402

SCENARIOS = {"base": 1.0, "demand -30%": 0.7, "demand +30%": 1.3}
LABELS = {"ppo": "StockPilot PPO", "sS": "(s,S)", "base_stock": "Base-stock", "ma_safety": "MA + safety stock"}
COLORS = {"ppo": "#2563eb", "sS": "#f97316", "base_stock": "#64748b", "ma_safety": "#a3a3a3"}


def policies_for(sku: str, cfg: dict) -> tuple[dict, dict | None]:
    d = resolve(cfg["paths"]["models_dir"]) / sku
    pols = {}
    agent = load_agent(d)
    if agent is not None:
        pols["ppo"] = agent[0]
    bl = json.loads((d / "baselines.json").read_text())
    for name in BASELINES:
        pols[name] = make_policy(name, bl[name]["params"])
    return pols, (agent[1] if agent else None)


def backtest(cfg: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, traces = [], []
    H = int(cfg["env"]["history_len"])
    for ctx in sku_contexts(df, cfg):
        pols, _ = policies_for(ctx.sku, cfg)
        rep = ctx.test_replay(H)
        for scen, scale in SCENARIOS.items():
            for name, pol in pols.items():
                m, tr = simulate(pol, ctx.params, rep["demand"], rep["dows"], rep["history"],
                                 demand_scale=scale, trace=(scale == 1.0))
                rows.append({"sku": ctx.sku, "scenario": scen, "policy": name, **m})
                if tr is not None:
                    tr["date"] = rep["dates"]
                    tr["sku"], tr["policy"] = ctx.sku, name
                    traces.append(tr)
    return pd.DataFrame(rows), pd.concat(traces, ignore_index=True)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    piv = res.pivot_table(index=["scenario", "sku"], columns="policy", values="total_cost").reset_index()
    fill = res.pivot_table(index=["scenario", "sku"], columns="policy", values="fill_rate").reset_index()
    out = piv[["scenario", "sku"]].copy()
    out["ppo_cost"], out["sS_cost"] = piv["ppo"].round(2), piv["sS"].round(2)
    out["best_baseline"] = piv[BASELINES].idxmin(axis=1)
    out["best_baseline_cost"] = piv[BASELINES].min(axis=1).round(2)
    out["saving_vs_sS_%"] = (100 * (piv["sS"] - piv["ppo"]) / piv["sS"]).round(1)
    out["ppo_fill"], out["sS_fill"] = fill["ppo"].round(4), fill["sS"].round(4)
    out["ppo_beats_sS"] = piv["ppo"] < piv["sS"]
    return out


def plots(res: pd.DataFrame, traces: pd.DataFrame, out_dir) -> None:
    base = res[res.scenario == "base"]
    skus = sorted(base.sku.unique())
    pols = [p for p in LABELS if p in base.policy.unique()]
    x = np.arange(len(skus))
    w = 0.8 / len(pols)
    for metric, fname, ylabel in [("total_cost", "cost_comparison.png", "Total cost on test period (GBP)"),
                                  ("fill_rate", "fill_rate_comparison.png", "Fill rate on test period")]:
        fig, ax = plt.subplots(figsize=(12, 4.8))
        for i, p in enumerate(pols):
            v = base[base.policy == p].set_index("sku").loc[skus, metric]
            ax.bar(x + (i - (len(pols) - 1) / 2) * w, v, w, label=LABELS[p], color=COLORS[p])
        ax.set_xticks(x, skus)
        ax.set_ylabel(ylabel)
        if metric == "fill_rate":
            ax.set_ylim(max(0, base[metric].min() - 0.05), 1.005)
        ax.legend(frameon=False, ncols=len(pols))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)

    fig, axes = plt.subplots(len(skus), 1, figsize=(12, 2.0 * len(skus)), sharex=True)
    for ax, sku in zip(np.atleast_1d(axes), skus):
        for p in ["sS", "ppo"]:
            t = traces[(traces.sku == sku) & (traces.policy == p)]
            ax.plot(pd.to_datetime(t.date), t.on_hand_end, label=LABELS[p], color=COLORS[p], lw=1.2)
        ax.set_ylabel(sku, rotation=0, ha="right", va="center")
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].legend(frameon=False, ncols=2, loc="upper left")
    fig.suptitle("End-of-day stock on the test period: StockPilot PPO vs (s,S)")
    fig.tight_layout()
    fig.savefig(out_dir / "stock_over_time.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest StockPilot on the held-out test period")
    ap.add_argument("--mode", choices=["public", "live"], default=None)
    args = ap.parse_args()
    cfg = load_config()
    ds = load_dataset(args.mode or cfg["data"]["mode"], cfg)
    res, traces = backtest(cfg, ds.df)
    out = resolve(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    res["data_source"] = ds.source
    res[res.scenario == "base"].round(4).to_csv(out / "metrics.csv", index=False)
    res[res.scenario != "base"].round(4).to_csv(out / "stress_metrics.csv", index=False)
    traces.to_csv(out / "traces.csv", index=False)
    summ = summarise(res)
    summ.to_csv(out / "summary.csv", index=False)
    plots(res, traces, out)

    pd.set_option("display.width", 200)
    base = res[res.scenario == "base"]
    show = base[["sku", "policy", "total_cost", "fill_rate", "avg_inventory", "n_orders", "waste_units"]].copy()
    show["total_cost"] = show["total_cost"].round(1)
    show["fill_rate"] = show["fill_rate"].round(4)
    show["avg_inventory"] = show["avg_inventory"].round(0)
    print("=== Held-out test period, real demand ===")
    print(show.to_string(index=False))
    for scen in SCENARIOS:
        s = summ[summ.scenario == scen]
        tot_ppo, tot_ss = s.ppo_cost.sum(), s.sS_cost.sum()
        print(f"\n--- {scen}: PPO beats (s,S) on {int(s.ppo_beats_sS.sum())}/{len(s)} SKUs | "
              f"total cost PPO {tot_ppo:,.0f} vs (s,S) {tot_ss:,.0f} ({100 * (tot_ss - tot_ppo) / tot_ss:+.1f}% saving)")
        print(s[["sku", "ppo_cost", "sS_cost", "saving_vs_sS_%", "ppo_fill", "sS_fill", "best_baseline"]]
              .to_string(index=False))
    print(f"\nWrote metrics.csv, stress_metrics.csv, summary.csv, traces.csv and 3 plots to "
          f"{out.relative_to(resolve('.'))}/ ({ds.source} data)")


if __name__ == "__main__":
    main()
