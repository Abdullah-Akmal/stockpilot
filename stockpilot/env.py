"""StockPilot inventory simulator and Gymnasium environment.

Daily timeline (day t, state observed at the start of the day):
  1. place the order q (fixed cost K if q > 0); it becomes sellable on day t + L
     (L = 0 means it is available for today's demand)
  2. demand d arrives; stock is sold FIFO (oldest lots first); unmet demand is lost
  3. lots age by one day; lots reaching the shelf life expire (waste)
  4. holding cost on the end-of-day stock
  5. the pipeline advances; the order due tomorrow joins on-hand stock
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class StepResult:
    order: int
    demand: int
    sold: int
    lost: int
    wasted: int
    on_hand_end: int
    holding_cost: float
    order_cost: float
    lost_sales_cost: float
    waste_cost: float
    profit: float           # margin * sold - goodwill * lost - holding - ordering - waste

    @property
    def total_cost(self) -> float:
        return self.holding_cost + self.order_cost + self.lost_sales_cost + self.waste_cost


@dataclass
class InventorySim:
    """Single-SKU lost-sales inventory system with lead time and optional FIFO expiry."""
    lead_time: int
    holding_cost: float          # per unit per day
    order_cost: float            # fixed per order
    stockout_penalty: float      # per lost unit (lost margin + goodwill)
    unit_cost: float
    price: float
    shelf_life: int | None = None
    lots: list = field(default_factory=list)       # [[qty, age], ...] oldest first
    pipeline: list = field(default_factory=list)   # pipeline[i] arrives in i+1 days

    def reset(self, on_hand: int = 0, pipeline: list | None = None) -> None:
        self.lots = [[int(on_hand), 0]] if on_hand > 0 else []
        p = list(pipeline or [])
        n = max(self.lead_time, 0)
        self.pipeline = (p + [0] * n)[:n] if n else []

    @property
    def on_hand(self) -> int:
        return int(sum(q for q, _ in self.lots))

    @property
    def position(self) -> int:
        """Inventory position = on-hand + on-order."""
        return self.on_hand + int(sum(self.pipeline))

    def _add_lot(self, qty: int) -> None:
        if qty <= 0:
            return
        if self.shelf_life is None and self.lots:   # no expiry: keep a single aggregated lot
            self.lots[0][0] += qty
        else:
            self.lots.append([int(qty), 0])

    def step(self, order: int, demand: int) -> StepResult:
        order, demand = max(int(order), 0), max(int(demand), 0)
        oc = self.order_cost if order > 0 else 0.0
        if self.lead_time == 0:
            self._add_lot(order)
        elif order > 0:
            self.pipeline[-1] += order

        # sell FIFO
        remaining = demand
        for lot in self.lots:
            take = min(lot[0], remaining)
            lot[0] -= take
            remaining -= take
            if remaining == 0:
                break
        self.lots = [lot for lot in self.lots if lot[0] > 0]
        sold, lost = demand - remaining, remaining

        # ageing and expiry
        wasted = 0
        if self.shelf_life is not None:
            for lot in self.lots:
                lot[1] += 1
            wasted = sum(q for q, a in self.lots if a >= self.shelf_life)
            self.lots = [lot for lot in self.lots if lot[1] < self.shelf_life]

        end = self.on_hand
        hc = self.holding_cost * end
        lc = self.stockout_penalty * lost
        wc = self.unit_cost * wasted
        margin = self.price - self.unit_cost
        goodwill = self.stockout_penalty - margin
        profit = margin * sold - goodwill * lost - hc - oc - wc

        # pipeline advances: tomorrow's arrival joins stock
        if self.lead_time > 0:
            arriving = self.pipeline.pop(0)
            self.pipeline.append(0)
            self._add_lot(arriving)

        return StepResult(order, demand, sold, lost, wasted, end, hc, oc, lc, wc, profit)


def sim_from_params(p: dict, lead_time: int | None = None) -> InventorySim:
    return InventorySim(
        lead_time=int(p["lead_time"] if lead_time is None else lead_time),
        holding_cost=p["holding_cost"], order_cost=p["order_cost"],
        stockout_penalty=p["stockout_penalty"], unit_cost=p["unit_cost"], price=p["price"],
        shelf_life=p.get("shelf_life"))


def make_obs(on_hand: int, pipeline: list, lead_time: int, history: list, dow: int,
             mu: float, max_L: int, H: int) -> np.ndarray:
    """Observation vector shared by the env and by policy evaluation."""
    s = 10.0 * mu
    pipe = np.zeros(max_L)
    pipe[:len(pipeline)] = pipeline
    hist = np.clip(np.asarray(list(history)[-H:], dtype=float) / mu, 0, 20)
    if len(hist) < H:
        hist = np.concatenate([np.ones(H - len(hist)), hist])
    dw = np.zeros(7)
    dw[int(dow) % 7] = 1.0
    position = on_hand + float(np.sum(pipeline))
    return np.concatenate([[on_hand / s, position / s], pipe / s, [lead_time / max_L], hist, dw]).astype(np.float32)


def action_unit(params: dict, env_cfg: dict) -> float:
    train_L = int(params.get("train_lead_time", params["lead_time"]))
    return env_cfg["action_max_mult"] * float(params["mean_demand"]) * max(train_L, 1) / (int(env_cfg["n_actions"]) - 1)


def block_bootstrap(demand: np.ndarray, dows: np.ndarray, length: int, block: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate random contiguous blocks of real history (keeps weekday and short-term autocorrelation)."""
    n = len(demand)
    block = min(block, n)
    out_d, out_w = [], []
    while sum(len(x) for x in out_d) < length:
        s = int(rng.integers(0, n - block + 1))
        out_d.append(demand[s:s + block])
        out_w.append(dows[s:s + block])
    return np.concatenate(out_d)[:length], np.concatenate(out_w)[:length]


class StockPilotEnv(gym.Env):
    """Gymnasium env for one SKU.

    Observation (fixed size, independent of lead time):
      [on_hand/(10*mu), position/(10*mu), pipeline[0..max_L-1]/(10*mu), L/max_L, last H demands/mu (clipped), dow one-hot(7)]
    Action: Discrete(n_actions) -> order qty = k/(n-1) * action_max_mult * mu * max(L_train, 1)
    Reward: -(daily cost) / (margin * mu)  [or daily profit / (margin * mu) with env.reward=profit]
    """
    metadata = {"render_modes": []}

    def __init__(self, params: dict, env_cfg: dict, train_demand: np.ndarray | None = None,
                 train_dows: np.ndarray | None = None, replay: dict | None = None,
                 demand_scale: float = 1.0, lead_time: int | None = None, seed: int | None = None):
        super().__init__()
        self.p = dict(params)
        self.cfg = env_cfg
        self.mu = float(params["mean_demand"])
        self.max_L = int(env_cfg["max_lead_time"])
        self.H = int(env_cfg["history_len"])
        self.n_actions = int(env_cfg["n_actions"])
        self.train_L = int(params.get("train_lead_time", params["lead_time"]))
        self.L = int(params["lead_time"] if lead_time is None else lead_time)
        if not 0 <= self.L <= self.max_L:
            raise ValueError(f"lead time must be in [0, {self.max_L}]")
        self.action_unit = action_unit(params, env_cfg)
        self.reward_scale = max(params["margin"] * self.mu, 1e-6)
        self.reward_mode = env_cfg.get("reward", "cost")
        self.train_demand = None if train_demand is None else np.asarray(train_demand, dtype=float)
        self.train_dows = None if train_dows is None else np.asarray(train_dows, dtype=int)
        self.replay = replay
        self.demand_scale = float(demand_scale)
        self.sim = sim_from_params(self.p, self.L)
        self.observation_space = spaces.Box(-1.0, np.inf, shape=(2 + self.max_L + 1 + self.H + 7,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)
        self._rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- helpers
    def action_to_qty(self, a: int) -> int:
        return int(round(int(a) * self.action_unit))

    def qty_to_action(self, q: float) -> int:
        return int(np.clip(round(q / self.action_unit), 0, self.n_actions - 1))

    def _scaled(self, d: np.ndarray) -> np.ndarray:
        return np.round(np.asarray(d, dtype=float) * self.demand_scale).astype(int)

    def _obs(self) -> np.ndarray:
        return make_obs(self.sim.on_hand, self.sim.pipeline, self.L, self.history,
                        int(self.dows[min(self.t, len(self.dows) - 1)]), self.mu, self.max_L, self.H)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.replay is not None:
            self.demand = self._scaled(self.replay["demand"])
            self.dows = np.asarray(self.replay["dows"], dtype=int)
            self.history = list(self._scaled(self.replay["history"]))[-self.H:]
            on_hand = int(self.replay.get("on_hand", round(self.mu * (self.L + 1))))
            pipeline = self.replay.get("pipeline")
        else:
            n = int(self.cfg["episode_len"]) + self.H
            d, w = block_bootstrap(self.train_demand, self.train_dows, n, int(self.cfg["block_len"]), self._rng)
            d = self._scaled(d)
            self.history, self.demand, self.dows = list(d[:self.H]), d[self.H:], w[self.H:]
            on_hand = int(self._rng.uniform(0, 2.5) * self.mu * (self.L + 1))
            pipeline = None
        if len(self.history) < self.H:
            self.history = [int(round(self.mu))] * (self.H - len(self.history)) + list(self.history)
        self.sim.reset(on_hand, pipeline)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        q = self.action_to_qty(action)
        r = self.sim.step(q, int(self.demand[self.t]))
        self.history.append(r.demand)
        self.t += 1
        terminated = self.t >= len(self.demand)
        # "cost" = -(holding + ordering + stockout penalty + waste). It equals profit minus margin*demand,
        # a term the agent cannot influence, so the optimal policy is identical but the variance is far lower.
        raw = -r.total_cost if self.reward_mode == "cost" else r.profit
        return self._obs(), float(raw / self.reward_scale), terminated, False, {"step": r}
