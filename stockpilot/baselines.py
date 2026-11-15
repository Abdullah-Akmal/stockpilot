"""Classic inventory baselines, each tuned on the TRAIN period only.

  (s,S)          : if inventory position <= s, order up to S
  base-stock     : every day order up to B
  MA + safety    : order up to  f*(L+1) + z*sigma*sqrt(L+1), f/sigma from a w-day moving window
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .config import load_config, resolve
from .data import load_dataset, sku_contexts
from .simulate import State, simulate


def s_S_policy(s: float, S: float):
    def policy(st: State) -> float:
        return S - st.position if st.position <= s else 0
    return policy


def base_stock_policy(B: float):
    def policy(st: State) -> float:
        return max(B - st.position, 0)
    return policy


def ma_safety_policy(window: int, z: float):
    def policy(st: State) -> float:
        h = np.asarray(st.history[-window:], dtype=float)
        f, sd = h.mean(), h.std()
        target = f * (st.lead_time + 1) + z * sd * np.sqrt(st.lead_time + 1)
        return max(target - st.position, 0)
    return policy


def make_policy(name: str, prm: dict):
    if name == "sS":
        return s_S_policy(prm["s"], prm["S"])
    if name == "base_stock":
        return base_stock_policy(prm["B"])
    if name == "ma_safety":
        return ma_safety_policy(int(prm["window"]), prm["z"])
    raise ValueError(name)


def _score(m: dict, target_fill: float) -> tuple:
    # feasible (fill >= target) first, then lowest cost
    return (0 if m["fill_rate"] >= target_fill else 1, m["total_cost"])


def grid(name: str, mu: float, L: int, center: dict | None = None) -> list[dict]:
    """Coarse grid (center=None) or a fine grid around a coarse optimum. Values in units of mean demand mu."""
    LL = L + 1
    if name == "sS":
        if center is None:
            pts = [(a, b) for a in np.arange(0, 6 * LL + 0.01, 1.0) for b in np.arange(1, 45.01, 1.0)]
        else:
            a0, b0 = center["s"] / mu, (center["S"] - center["s"]) / mu
            pts = [(a, b) for a in np.arange(max(a0 - 2, 0), a0 + 2.01, 0.25)
                   for b in np.arange(max(b0 - 2, 0.25), b0 + 2.01, 0.25)]
        return [{"s": round(mu * a, 2), "S": round(mu * (a + b), 2)} for a, b in pts]
    if name == "base_stock":
        if center is None:
            return [{"B": round(mu * a, 2)} for a in np.arange(0.5, 8 * LL + 0.01, 0.5)]
        a0 = center["B"] / mu
        return [{"B": round(mu * a, 2)} for a in np.arange(max(a0 - 0.5, 0.1), a0 + 0.51, 0.1)]
    if name == "ma_safety":
        if center is None:
            return [{"window": w, "z": round(float(z), 2)} for w in (7, 14, 28, 56) for z in np.arange(0, 8.01, 0.5)]
        return [{"window": center["window"], "z": round(float(z), 2)}
                for z in np.arange(max(center["z"] - 0.5, 0), center["z"] + 0.51, 0.1)]
    raise ValueError(name)


def tune(name: str, params: dict, replay: dict, target_fill: float, overrides: dict | None = None,
         lead_time: int | None = None) -> tuple[dict, dict]:
    p = {**params, **(overrides or {})}
    L = int(p["lead_time"] if lead_time is None else lead_time)
    best, best_m = None, None
    for stage in (0, 1):
        for prm in grid(name, p["mean_demand"], L, None if stage == 0 else best):
            m, _ = simulate(make_policy(name, prm), p, replay["demand"], replay["dows"], replay["history"],
                            lead_time=L)
            if best_m is None or _score(m, target_fill) < _score(best_m, target_fill):
                best, best_m = prm, m
    return best, best_m


BASELINES = ["sS", "base_stock", "ma_safety"]


def tune_all(cfg: dict, ctxs) -> dict:
    out = {}
    for c in ctxs:
        out[c.sku] = {}
        for name in BASELINES:
            prm, m = tune(name, c.params, c.train_replay(), cfg["costs"]["target_fill_rate"])
            out[c.sku][name] = {"params": prm, "train_cost": round(m["total_cost"], 2),
                                "train_fill_rate": round(m["fill_rate"], 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune baselines on the training period")
    ap.add_argument("--mode", choices=["public", "live"], default=None)
    args = ap.parse_args()
    cfg = load_config()
    ds = load_dataset(args.mode or cfg["data"]["mode"], cfg)
    ctxs = sku_contexts(ds.df, cfg)
    t0 = time.time()
    tuned = tune_all(cfg, ctxs)
    models = resolve(cfg["paths"]["models_dir"])
    rows = []
    for c in ctxs:
        d = models / c.sku
        d.mkdir(parents=True, exist_ok=True)
        (d / "baselines.json").write_text(json.dumps(tuned[c.sku], indent=2))
        for name, v in tuned[c.sku].items():
            rows.append({"sku": c.sku, "policy": name, "params": v["params"],
                         "train_cost": v["train_cost"], "train_fill": v["train_fill_rate"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nTuned {len(ctxs)} SKUs x {len(BASELINES)} baselines on TRAIN only in {time.time() - t0:.1f}s "
          f"-> models/<sku>/baselines.json")


if __name__ == "__main__":
    main()
