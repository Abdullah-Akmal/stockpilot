"""Today's order recommendations from the latest row of each SKU."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import make_policy, tune
from .config import resolve
from .data import sku_params, sku_series
from .simulate import State


def current_state(series: pd.DataFrame, lead_time: int, H: int = 14) -> State:
    """State at the start of the day after the last row (on_hand/on_order are end-of-day values)."""
    last = series.iloc[-1]
    on_order = int(last.get("on_order", 0) or 0) if "on_order" in series.columns else 0
    L = int(lead_time)
    pipeline = [0] * L
    if L and on_order:
        # Arrival dates are unknown: spread outstanding units evenly over the lead time.
        base, rem = divmod(on_order, L)
        pipeline = [base + (1 if i < rem else 0) for i in range(L)]
    elif on_order and L == 0:
        pipeline = []
    on_hand = int(last["on_hand"]) + (on_order if L == 0 else 0)
    hist = series["units_sold"].astype(int).tolist()[-H:]
    dow = (pd.Timestamp(last["date"]) + pd.Timedelta(days=1)).dayofweek
    return State(on_hand, pipeline, on_hand + sum(pipeline), hist, int(dow), L)


def recommend(df: pd.DataFrame, cfg: dict, load_agent_fn, descriptions: dict | None = None) -> pd.DataFrame:
    models = resolve(cfg["paths"]["models_dir"])
    H = int(cfg["env"]["history_len"])
    rows = []
    for sku in sorted(df["sku"].unique()):
        s = sku_series(df, sku)
        L = int(s["lead_time_days"].iloc[-1])
        st = current_state(s, L, H)
        d: Path = models / sku
        agent = load_agent_fn(str(d))
        bl_path = d / "baselines.json"
        if bl_path.exists():
            sS_params = json.loads(bl_path.read_text())["sS"]["params"]
            note = ""
        else:
            # No trained artefacts for this SKU (e.g. a new SKU in the live sheet): tune (s,S) on its own history.
            p = sku_params(s, cfg)
            hist = s["units_sold"].to_numpy()
            if len(hist) < H + 14:
                rows.append({"sku": sku, "description": (descriptions or {}).get(sku, ""), "on_hand": st.on_hand,
                             "on_order": sum(st.pipeline), "position": st.position,
                             "avg_demand_14d": float(np.mean(st.history)), "lead_time": L,
                             "stockpilot_order": None, "sS_order": None,
                             "recent_demand": st.history, "note": "Not enough history (need 28+ days)."})
                continue
            rep = {"demand": hist[H:], "dows": s["date"].dt.dayofweek.to_numpy()[H:], "history": hist[:H]}
            sS_params, _ = tune("sS", p, rep, cfg["costs"]["target_fill_rate"])
            note = "No trained agent for this SKU yet: (s,S) tuned on its sheet history; retrain to add PPO."
        sS_q = int(round(make_policy("sS", sS_params)(st)))
        ppo_q = int(round(agent[0](st))) if agent else None
        if agent and L > int(agent[1]["env_cfg"]["max_lead_time"]):
            ppo_q, note = None, "Lead time above the agent's maximum."
        rows.append({
            "sku": sku, "description": (descriptions or {}).get(sku, ""), "on_hand": st.on_hand,
            "on_order": int(sum(st.pipeline)), "position": st.position,
            "avg_demand_14d": round(float(np.mean(st.history)), 1), "lead_time": L,
            "stockpilot_order": ppo_q, "sS_order": sS_q,
            "s": round(sS_params["s"]), "S": round(sS_params["S"]),
            "recent_demand": st.history, "note": note,
            "as_of": pd.Timestamp(s["date"].iloc[-1]).date().isoformat(),
        })
    return pd.DataFrame(rows)
