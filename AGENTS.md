# AGENTS.md: guide for coding agents working on StockPilot

StockPilot recommends a daily order quantity per SKU using one PPO agent per SKU. It is trained on real
demand (UCI Online Retail II) or a live Google Sheet, proven against classic inventory policies on held-out
data, served as a Streamlit dashboard and retrained weekly by GitHub Actions.

- Human overview and current results: `README.md`. Long-form explainer: `docs/StockPilot_Portfolio.pdf`.
- What has been tried and what happened: `docs/EXPERIMENTS.md`. **Read it before tuning anything.**
- Live app: https://stockpilot-7wqs2osthj56yvsma2mmaq.streamlit.app/ · Repo: https://github.com/Abdullah-Akmal/stockpilot

## Golden rules (do not break these)

1. **Never use test-period results to make a choice.** Hyperparameters, seeds, tuning variants, checkpoints
   and promotion are all decided by **training-period** replay cost (`score_on_train` in `stockpilot/train.py`).
   The test period (last 25% of dates) is only *reported*. Picking the variant that scores best on test is
   leakage, even if it makes the README look better. This has cost us results twice (85099B, demo COF-CUP-12OZ);
   we kept the honest choice both times.
2. **Report losses honestly.** If the agent loses to (s,S) on a SKU, say so in README/results. Do not hide or
   drop SKUs.
3. **Never train inside the app.** `app/streamlit_app.py` only loads `models/` and runs inference.
4. **Do not blindly overwrite deployed agents.** Use `train_and_promote` (the default CLI path): a challenger
   replaces `models/<sku>/` only if it is cheaper on the training period. `--out <dir>` is for experiments and
   never touches `models/`.
5. **README numbers must come from `results/*.csv`.** Recompute them; never type figures by hand (a hand-typed
   figure was wrong once). Rebuild the PDF with `scripts/build_portfolio_pdf.py` after results change.
6. **Baselines must be tuned as hard as the agent.** If a tuned parameter lands on the edge of its grid, widen the
   grid (`grid()` in `stockpilot/baselines.py`); an under-tuned baseline flatters the agent.

## Project map

| Path | Role |
|---|---|
| `stockpilot/data.py` | Load UCI / live sheet / sample, `validate_schema`, cleaning, `chronological_split`, `sku_params` (train-only stats), `SkuContext` (train/test replays) |
| `stockpilot/env.py` | `InventorySim` (single source of truth for dynamics and costs), `make_obs`, `StockPilotEnv` (Gymnasium), `block_bootstrap` |
| `stockpilot/simulate.py` | `simulate(policy, params, demand, ...)`: the replay loop used by baselines, backtest, What-if and model selection. A policy is `State -> qty` |
| `stockpilot/baselines.py` | (s,S), base-stock, MA + safety stock; coarse-to-fine grid search on train |
| `stockpilot/train.py` | PPO training (`train_sku`), checkpoint selection on train (`TrainReplaySelector`), `init_action_bias`, `PPOPolicy` wrapper, `load_agent`, `score_on_train`, `train_and_promote` |
| `stockpilot/evaluate.py` | Backtest + ±30% stress test → `results/` CSVs and PNGs |
| `stockpilot/recommend.py` | Today's orders from each SKU's latest row (on_hand/on_order are END-of-day) |
| `stockpilot/config.py` | `load_config()`: explicit path > `$STOCKPILOT_CONFIG` > `configs/config.yaml`; `merged()` deep-merge |
| `app/streamlit_app.py` | 4-page dashboard (Today's Orders, Backtest, What-if, About) |
| `configs/config.yaml` | Economics, env, PPO, `sku_overrides` (per-SKU tuning, reproduced by CI) |
| `configs/demo.yaml` | Same pipeline on the synthetic demo shop, writing to `models_demo/`, `results_demo/` (git-ignored) |
| `models/<sku>/` | `ppo.zip`, `vecnormalize.pkl`, `meta.json` (params + env cfg used at train time), `baselines.json`. Committed (~0.6 MB/SKU) |
| `results/` | `metrics.csv` (base scenario), `stress_metrics.csv`, `summary.csv` (per SKU × scenario), `traces.csv`, PNGs. Committed |
| `data/processed/` | Cleaned UCI daily data (committed). `data/raw/` is git-ignored (45 MB download, parquet cache) |
| `data/sample/` | Live-schema fallback so the app always runs |
| `data/demo/` | Synthetic shop + deliberately broken copy (`scripts/make_demo_data.py`) |
| `.github/workflows/retrain.yml` | Weekly/manual: tests → data → baselines → train (3 seeds, promote) → evaluate → commit |
| `tests/` | `test_env.py` (simulator invariants and costs), `test_data.py` (validation, split), `test_app.py` (every page, both modes, via Streamlit AppTest) |

## Commands

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
python -m pytest                                    # 36 tests, ~20 s; must stay green
python -m stockpilot.data --mode public             # download + clean UCI (~1-2 min first time)
python -m stockpilot.baselines                      # tune baselines on train
python -m stockpilot.train --sku 85123A             # train + champion/challenger promote (seeds from config)
python -m stockpilot.train --sku 85123A --steps 800000 --set env.action_max_mult=4 --set ppo.seed=7 --out variants/x   # experiment only
python -m stockpilot.evaluate                       # backtest → results/
streamlit run app/streamlit_app.py
python scripts/build_portfolio_pdf.py               # needs reportlab (docs-only, not in requirements)
```

- Train SKUs **in parallel as separate processes** (one per SKU); a single process trains sequentially. ~3,000 env
  steps/s per process on CPU; 500k steps ≈ 2-5 min.
- `--set section.key=value` overrides one level deep (values parsed as JSON when they look like numbers/lists).
- Demo pipeline: `STOCKPILOT_CONFIG=configs/demo.yaml LIVE_CSV_URL=<url of data/demo/demo_shop.csv> python -m stockpilot.<step>`.
  `LIVE_CSV_URL` must be http(s); serve a local file with `python -m http.server`, or use the raw GitHub URL.

## How the pieces fit

- **Timing within a day** (`InventorySim.step`): place order (fixed cost if > 0) → serve demand FIFO, unmet demand
  is lost → age lots / expire if shelf life → holding cost on end-of-day stock → pipeline advances. An order placed
  on day t is sellable on day t+L (L=0 means same day). `tests/test_env.py` pins this; keep it passing.
- **Observation** (38 dims, fixed size): on_hand, position, 14-slot padded pipeline (÷ 10·μ), L/14, last 14 demands
  ÷ μ (clipped at 20), weekday one-hot. Changing it **invalidates every saved model**: retrain all SKUs and bump
  the shape in `tests/test_env.py`.
- **Action**: `Discrete(21)`; quantity = k × `action_unit`, where unit = `action_max_mult × μ × max(L_train, 1) / 20`.
  `PPOPolicy` recomputes the unit from the model's own `meta.json`, so per-SKU `action_max_mult` is safe.
- **Reward**: `-total_cost / (margin × μ)` (`env.reward: cost`). Same optimum as profit; much lower variance.
- **Inference must match training normalisation**: `PPOPolicy` applies the saved `VecNormalize` obs stats manually.
- **μ, prices, costs are train-period statistics** (`sku_params`), stored in `meta.json`.

## Known pitfalls (each one has bitten this project)

- **Windows Smart App Control** blocked pandas 3.x DLLs; pandas is pinned to 2.3.3. Don't bump without testing an import.
- **SKU codes look numeric** ("15036"). Always read CSVs with `dtype={"sku": str}`, and set Plotly
  `xaxes(type="category")`, or charts show "20k, 40k".
- **Parallel training** must use per-SKU staging (`models/.staging/<sku>`). A shared staging dir once let the first
  finished process delete the others' work, and CI stayed green. The workflow now checks every PID's exit code.
- **Exploration collapse**: without `init_p_zero` (initial P(order 0) = 0.9) the agent floods stock and learns a
  constant order that ignores state. Keep it.
- **Run-to-run variance**: same seed on a different CPU gives different agents. Compare multiple seeds; rely on
  promotion, not a single run.
- **Economics drive feasibility**: with £15/order the EOQ cycle (~36 days) exceeded the action range and PPO lost
  by construction. If you change costs, check that the optimal order size fits `action_max_mult`.
- **Streamlit Cloud** main file is `app/streamlit_app.py` (not the root). Torch on Linux is pinned to the CPU wheel URL
  in `requirements.txt`; keep the three `python_version` markers in sync when bumping torch.
- Sheet semantics: `on_hand`/`on_order` are **end-of-day**. `recommend.current_state` spreads `on_order` evenly over L.
- `use_container_width` is deprecated in Streamlit; use `width="stretch"`.

## How to improve it safely (suggested workflow)

1. Read `docs/EXPERIMENTS.md`; pick a SKU or idea; write the hypothesis there.
2. Run variants with `--out variants/<name>` (git-ignored), several seeds each, in parallel processes.
3. Select per SKU by `score_on_train`; copy the winner into `models/<sku>/` only if it beats the current agent on train.
4. Record the winning settings in `configs/config.yaml` → `sku_overrides` so CI reproduces them.
5. `python -m stockpilot.evaluate`, then update README tables from `results/` (and rebuild the PDF).
6. `python -m pytest`; log the outcome, including failures, in `docs/EXPERIMENTS.md`; commit.

Promising directions (see README "Limitations"): un-censor demand on stockout days, train on scaled demand
(domain randomisation) for the −30% regime, one shared policy with SKU features, stochastic lead times,
MOQ/pack-size constraints, rolling-origin backtests for confidence intervals, a hybrid agent that adjusts (s,S)
parameters instead of raw quantities (likely to help steady SKUs where (s,S) is near-optimal).

## Conventions

- Python 3.11, type hints, small pure functions; config in YAML, not constants in code.
- New behaviour gets a test; simulator changes need a hand-checked arithmetic test.
- Commit messages explain *why*; CI commits come from `stockpilot-bot`.
- Keep `models/` and `results/` committed (they are the app's inputs); keep `variants*/`, `models_demo/`,
  `results_demo/`, `data/raw/`, `logs/`, `.streamlit/secrets.toml` out of git.
