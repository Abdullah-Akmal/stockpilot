"""One replay simulator used by baselines, the PPO backtest and the What-if page."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .env import sim_from_params


@dataclass
class State:
    on_hand: int
    pipeline: list
    position: int
    history: list       # past daily demand, most recent last
    dow: int
    lead_time: int


Policy = Callable[[State], float]


def simulate(policy: Policy, params: dict, demand, dows, history, on_hand: int | None = None,
             pipeline: list | None = None, lead_time: int | None = None, demand_scale: float = 1.0,
             overrides: dict | None = None, trace: bool = False):
    """Replay a demand sequence under a policy. Returns (metrics, trace_df or None)."""
    p = {**params, **(overrides or {})}
    L = int(p["lead_time"] if lead_time is None else lead_time)
    sim = sim_from_params(p, L)
    mu = float(p["mean_demand"])
    d = np.round(np.asarray(demand, dtype=float) * demand_scale).astype(int)
    hist = list(np.round(np.asarray(history, dtype=float) * demand_scale).astype(int))
    sim.reset(int(round(mu * (L + 1))) if on_hand is None else int(on_hand), pipeline)
    tot = dict(holding=0.0, ordering=0.0, lost_sales=0.0, waste=0.0, profit=0.0)
    sold = lost = wasted = n_orders = 0
    inv_sum = 0.0
    rows = []
    for t in range(len(d)):
        st = State(sim.on_hand, list(sim.pipeline), sim.position, hist, int(dows[t]), L)
        q = max(int(round(policy(st))), 0)
        start_on_hand = sim.on_hand
        r = sim.step(q, int(d[t]))
        hist.append(int(d[t]))
        tot["holding"] += r.holding_cost
        tot["ordering"] += r.order_cost
        tot["lost_sales"] += r.lost_sales_cost
        tot["waste"] += r.waste_cost
        tot["profit"] += r.profit
        sold += r.sold
        lost += r.lost
        wasted += r.wasted
        n_orders += int(q > 0)
        inv_sum += r.on_hand_end
        if trace:
            rows.append({"t": t, "on_hand_start": start_on_hand, "order": q, "demand": r.demand,
                         "sold": r.sold, "lost": r.lost, "on_hand_end": r.on_hand_end, "cost": r.total_cost})
    n = max(len(d), 1)
    total_demand = sold + lost
    metrics = {
        "total_cost": tot["holding"] + tot["ordering"] + tot["lost_sales"] + tot["waste"],
        "holding_cost": tot["holding"], "ordering_cost": tot["ordering"],
        "lost_sales_cost": tot["lost_sales"], "waste_cost": tot["waste"], "profit": tot["profit"],
        "fill_rate": sold / total_demand if total_demand else 1.0,
        "avg_inventory": inv_sum / n, "n_orders": n_orders, "waste_units": wasted,
        "lost_units": lost, "demand_units": total_demand, "days": n,
    }
    return metrics, (pd.DataFrame(rows) if trace else None)
