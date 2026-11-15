"""Train one PPO agent per SKU (Stable-Baselines3, CPU, fixed seeds).

Checkpoint selection uses replay cost on the TRAIN period only; the test period is never seen.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .config import load_config, merged, resolve
from .data import SkuContext, load_dataset, sku_contexts
from .env import StockPilotEnv, action_unit, make_obs
from .simulate import State, simulate


# --------------------------------------------------------------------------- policy wrapper
class PPOPolicy:
    """Callable policy for `simulate`: State -> order quantity."""

    def __init__(self, model: PPO, obs_rms, clip_obs: float, epsilon: float, params: dict, env_cfg: dict):
        self.model, self.obs_rms, self.clip, self.eps = model, obs_rms, clip_obs, epsilon
        self.mu = float(params["mean_demand"])
        self.unit = action_unit(params, env_cfg)
        self.max_L, self.H = int(env_cfg["max_lead_time"]), int(env_cfg["history_len"])

    def obs(self, st: State) -> np.ndarray:
        o = make_obs(st.on_hand, st.pipeline, st.lead_time, st.history, st.dow, self.mu, self.max_L, self.H)
        return np.clip((o - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.eps), -self.clip, self.clip)

    def action(self, st: State) -> int:
        a, _ = self.model.predict(self.obs(st)[None].astype(np.float32), deterministic=True)
        return int(np.asarray(a).ravel()[0])

    def __call__(self, st: State) -> float:
        return self.action(st) * self.unit


def load_agent(sku_dir: Path) -> tuple[PPOPolicy, dict] | None:
    sku_dir = Path(sku_dir)
    if not (sku_dir / "ppo.zip").exists():
        return None
    meta = json.loads((sku_dir / "meta.json").read_text())
    model = PPO.load(sku_dir / "ppo.zip", device="cpu")
    with open(sku_dir / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    return PPOPolicy(model, vn.obs_rms, vn.clip_obs, vn.epsilon, meta["params"], meta["env_cfg"]), meta


# --------------------------------------------------------------------------- training
class TrainReplaySelector(BaseCallback):
    """Every `every` steps, score the current policy on the TRAIN replay; keep the cheapest."""

    def __init__(self, ctx: SkuContext, env_cfg: dict, out_dir: Path, every: int):
        super().__init__()
        self.ctx, self.env_cfg, self.out_dir, self.every = ctx, env_cfg, out_dir, every
        self.best_cost, self.history, self._next = np.inf, [], every

    def score(self) -> dict:
        vn: VecNormalize = self.model.get_env()
        pol = PPOPolicy(self.model, vn.obs_rms, vn.clip_obs, vn.epsilon, self.ctx.params, self.env_cfg)
        r = self.ctx.train_replay(int(self.env_cfg["history_len"]))
        m, _ = simulate(pol, self.ctx.params, r["demand"], r["dows"], r["history"])
        return m

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            m = self.score()
            self.history.append((self.num_timesteps, round(m["total_cost"], 1), round(m["fill_rate"], 4)))
            if m["total_cost"] < self.best_cost:
                self.best_cost = m["total_cost"]
                self.model.save(self.out_dir / "ppo.zip")
                self.model.get_env().save(str(self.out_dir / "vecnormalize.pkl"))
        return True


def init_action_bias(model: PPO, p_zero: float) -> None:
    """Start with P(order 0) = p_zero so early exploration does not flood the warehouse.

    With uniform random actions the agent orders ~4.5x mean demand per day, stock explodes and it never
    visits the low-stock states where decisions matter. Only the initial bias changes; training is unchanged.
    """
    if not 0 < p_zero < 1:
        return
    n = model.action_space.n
    with torch.no_grad():
        b = model.policy.action_net.bias
        b.zero_()
        b[0] = float(np.log(p_zero * (n - 1) / (1 - p_zero)))


def train_sku(ctx: SkuContext, cfg: dict, steps: int | None = None, seed: int | None = None,
              out_root: Path | None = None, verbose: bool = True) -> dict:
    cfg = merged(cfg, (cfg.get("sku_overrides") or {}).get(ctx.sku))   # per-SKU tuning from config
    pc, ec = cfg["ppo"], cfg["env"]
    steps = int(steps or pc["total_timesteps"])
    seed = int(pc["seed"] if seed is None else seed)
    out_dir = (out_root or resolve(cfg["paths"]["models_dir"])) / ctx.sku
    out_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(seed)
    torch.set_num_threads(1)

    def make(i: int):
        def _f():
            return Monitor(StockPilotEnv(ctx.params, ec, ctx.train_demand, ctx.train_dows, seed=seed + i))
        return _f

    venv = VecNormalize(DummyVecEnv([make(i) for i in range(int(pc["n_envs"]))]),
                        norm_obs=True, norm_reward=True, gamma=pc["gamma"], clip_obs=10.0)
    venv.seed(seed)
    model = PPO("MlpPolicy", venv, device="cpu", seed=seed, verbose=0,
                n_steps=pc["n_steps"], batch_size=pc["batch_size"], n_epochs=pc["n_epochs"],
                learning_rate=pc["learning_rate"], gamma=pc["gamma"], gae_lambda=pc["gae_lambda"],
                ent_coef=pc["ent_coef"], clip_range=pc["clip_range"],
                policy_kwargs={"net_arch": list(pc["net_arch"])})
    init_action_bias(model, float(pc.get("init_p_zero", 0.0)))
    sel = TrainReplaySelector(ctx, ec, out_dir, every=max(steps // 20, pc["n_steps"] * pc["n_envs"]))
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=sel)
    sel.num_timesteps = model.num_timesteps
    sel._next = 0
    sel._on_step()  # score the final policy too
    secs = time.time() - t0
    meta = {
        "sku": ctx.sku, "params": ctx.params, "env_cfg": ec, "ppo": pc, "timesteps": steps, "seed": seed,
        "train_seconds": round(secs, 1), "best_train_cost": round(float(sel.best_cost), 2),
        "selection_curve": sel.history, "train_start": str(ctx.train["date"].min().date()),
        "train_end": str(ctx.train["date"].max().date()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    if verbose:
        curve = " ".join(f"{s // 1000}k:{c:.0f}" for s, c, _ in sel.history)
        print(f"[{ctx.sku}] {steps} steps in {secs:.0f}s | best train cost {sel.best_cost:.1f} | {curve}", flush=True)
    return meta


def score_on_train(sku_dir: Path, ctx: SkuContext, cfg: dict) -> float:
    """Training-period replay cost of the agent saved in sku_dir (inf if none). Never touches test data."""
    agent = load_agent(sku_dir)
    if agent is None:
        return float("inf")
    r = ctx.train_replay(int(cfg["env"]["history_len"]))
    m, _ = simulate(agent[0], ctx.params, r["demand"], r["dows"], r["history"])
    return float(m["total_cost"])


AGENT_FILES = ("ppo.zip", "vecnormalize.pkl", "meta.json")


def train_and_promote(ctx: SkuContext, cfg: dict, steps: int | None = None, n_seeds: int | None = None) -> dict:
    """Train n_seeds challengers, keep the best by training cost, and replace the current (champion) agent
    only if the challenger is cheaper on the training period. PPO varies a lot between runs, so a blind
    overwrite can silently make a deployed agent worse."""
    eff = merged(cfg, (cfg.get("sku_overrides") or {}).get(ctx.sku))
    n_seeds = int(n_seeds or eff["ppo"].get("n_seeds", 1))
    base_seed = int(eff["ppo"]["seed"])
    models = resolve(cfg["paths"]["models_dir"])
    staging = models / ".staging"
    cands = []
    for k in range(n_seeds):
        root = staging / f"seed{k}"
        meta = train_sku(ctx, cfg, steps=steps, seed=base_seed + 1000 * k, out_root=root)
        cands.append((score_on_train(root / ctx.sku, ctx, cfg), root / ctx.sku, meta["seed"]))
    chall_cost, chall_dir, chall_seed = min(cands, key=lambda c: c[0])
    target = models / ctx.sku
    champ_cost = score_on_train(target, ctx, cfg)
    promoted = chall_cost < champ_cost
    if promoted:
        target.mkdir(parents=True, exist_ok=True)
        for f in AGENT_FILES:
            shutil.copy(chall_dir / f, target / f)
    shutil.rmtree(staging, ignore_errors=True)
    verdict = "PROMOTED new agent" if promoted else "KEPT current agent"
    print(f"[{ctx.sku}] {verdict}: challenger (seed {chall_seed}) train cost {chall_cost:.1f} vs "
          f"current {champ_cost:.1f}  [seeds tried: {', '.join(f'{c:.0f}' for c, _, _ in cands)}]", flush=True)
    return {"sku": ctx.sku, "promoted": promoted, "challenger_cost": chall_cost, "champion_cost": champ_cost}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train StockPilot PPO agents")
    ap.add_argument("--mode", choices=["public", "live"], default=None)
    ap.add_argument("--sku", action="append", help="train only these SKUs (repeatable)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None, help="experiment output dir (skips champion/challenger promotion)")
    ap.add_argument("--seeds", type=int, default=None, help="challenger seeds per SKU (default ppo.n_seeds)")
    ap.add_argument("--set", action="append", default=[],
                    help="override config, e.g. --set env.action_max_mult=4 --set ppo.ent_coef=0.02")
    args = ap.parse_args()
    cfg = load_config()
    for kv in args.set:
        k, v = kv.split("=", 1)
        sect, key = k.split(".")
        cfg = merged(cfg, {sect: {key: json.loads(v) if v[0] in "[{0123456789-.tfn" else v}})
    ds = load_dataset(args.mode or cfg["data"]["mode"], cfg)
    ctxs = sku_contexts(ds.df, cfg)
    if args.sku:
        ctxs = [c for c in ctxs if c.sku in set(args.sku)]
    t0 = time.time()
    for c in ctxs:
        if args.out:   # experiment: write straight to another folder, no promotion
            train_sku(c, cfg, steps=args.steps, out_root=Path(args.out))
        else:
            train_and_promote(c, cfg, steps=args.steps, n_seeds=args.seeds)
    print(f"Done: {len(ctxs)} SKU(s) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
