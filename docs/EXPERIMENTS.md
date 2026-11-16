# Experiment log

Every experiment run on StockPilot so far, in order, including the ones that failed. Costs are total cost in GBP.
"Train cost" = replay cost on the training period (used for all decisions). "Test" = held-out last 25% of dates
(reported only). Add new entries at the bottom: **hypothesis → setup → result → decision**.

Baseline reference (public data, test period): (s,S) total £11,792 · base-stock £8,495 · MA + safety stock £18,550.

---

## E1 · Economics sanity check (before any training)
- **Problem:** with holding 25%/yr and £15/order, the EOQ cycle was ~36 days of demand, far beyond the agent's max
  order (3 × lead-time demand ≈ 9 days). PPO would lose by construction.
- **Change:** holding 100%/yr (capital + storage + obsolescence of seasonal giftware), £2/order → optimal cycles of
  2-8 days.
- **Decision:** adopted; documented as an assumption. If you change costs, re-check that EOQ fits the action range.

## E2 · Baseline grid edges
- **Problem:** first tuning put several optima on the grid edge (e.g. 15036 s = 12μ = max; MA z = 3.0 = max).
- **Change:** wider coarse grid (s up to 6(L+1)μ, S−s up to 45μ, z up to 8, window up to 56) + fine grid around the best.
- **Result:** 15036 (s,S) train cost 1,976 → 958; 85099B needed a ±2μ refinement window.
- **Decision:** adopted. Always check for edge optima after changing anything.

## E3 · First PPO runs on SKU 21212 (train cost; (s,S) = 705)
| Run | Setup | Best train cost | Diagnosis |
|---|---|---:|---|
| a | profit reward, 300k steps | 7,271 | avg inventory 16,674 (~100 days); holding 91% of cost |
| b | cost reward (same optimum, less variance) | 5,960 | now under-stocks (fill 89%); curve swings 6k↔23k |
| c | 4 hyper-param variants (lr 1e-4/5e-5, bigger batch, 128×128, γ .995, clip .1) | 33,779-51,359 | all worse |
| d | probe of (c) | | **action = 3 for every on-hand level 0…8,000**; stochastic policy avg stock 36k |
- **Root cause:** exploration collapse. Uniform random actions order ~4.5× demand/day, stock explodes, low-stock
  states are never visited, and the policy learns a constant order.

## E4 · Fix exploration (SKU 21212, 400k steps)
| Variant | Best train cost |
|---|---:|
| `init_p_zero=0.8` + inventory position feature, default PPO | 830 |
| same + lr 1e-4, batch 1024 | 1,359 (slow start) |
| **`init_p_zero=0.9` + 128×128 net** | **798** (converged by 60k, stable) |
| `init_p_zero=0.8` + action_max_mult 4 | 1,126 |
- **Decision:** adopted `init_p_zero=0.9`, `net_arch [128,128]`, cost reward, position feature, 500k steps.

## E5 · Full run, 10 public SKUs, 1 seed, 500k steps
- Test: PPO beats (s,S) on **5/10**, aggregate −31.0% cost (driven by 22197 and 84879).
- Losers: 22492 (−67%), 21977 (−25%), 85099B (−22%), 84991 (−3%), 21212 (−1%).

## E6 · Tuning round 1 on the 5 losers (800k steps; selection by train cost)
Variants: `seed=7`, `action_max_mult=4`, `lr=1e-4 + n_epochs 5 + γ 0.995`.
| SKU | Picked | Train cost before → after |
|---|---|---|
| 21212 | seed 7 | 786 → 742 |
| 21977 | act4 | 927 → 896 |
| 22492 | act4 | 847 → 529 |
| 84991 | act4 | 679 → 668 |
| 85099B | act4 | 2,553 → 2,487 |
- The low-lr variant was the worst on every SKU (1.3-3.6× higher cost). Don't retry it without a reason.
- Test: **8/10**, −41.9%.

## E7 · Tuning round 2 (21977, 85099B): act4 + seeds 13/21, act4 1.5M steps
- Picked act4 seed 21 for both (train 874 and 2,222). 1.5M steps did **not** beat 800k.
- Test: 8/10, −42.5%. 21977 still −17%, 85099B −14%.

## E8 · First CI retrain (GitHub runner, 1 seed, blind overwrite)
- Same seeds, different CPU → different agents: **4/10, +15.2% cost (worse than (s,S))**, overwrote the good models.
- **Decision:** champion/challenger promotion (`train_and_promote`): 3 seeds, promote only if train cost improves.
  Applied to local vs CI agents: kept 7 local, 3 CI → 8/10, −41.5%. 85099B's CI agent was better on train
  (2,202 vs 2,222) but worse on test (−42% vs −14%); kept anyway (no test peeking).

## E9 · Second CI retrain (3 seeds, promotion)
- Promoted 22492 (529 → 522) and 85099B (2,202 → 2,180); kept the rest. Test: **9/10, −45.4%** (current results).
- **Caveat found later (E10):** 5 of 10 SKUs crashed in this run (shared staging dir) while the job stayed green,
  so they were not actually retrained. Fixed; a full clean CI retrain has not run since.

## E10 · Synthetic demo shop (configs/demo.yaml), 7 trainable SKUs, 92 test days
- Running 7 SKUs in parallel exposed the shared-staging bug (first finisher deleted the others' work). Fixed with
  per-SKU staging; workflow now checks every PID's exit code.
- Out of the box (1 seed, 500k): **3/7**, −10.4%. Lost on steady SKUs: TEA-EARLGREY −173%, COF-CUP-12OZ −58%.
- Round 1 (act4 / act2 / seed 7 / act2+seed 7, 800k): act4 won 3 of 4 SKUs (FILTER-PAPERS 906 → 329, TEA 1,082 → 360);
  act2 won only MUG. **4/7, −20.7%.**
- Round 2 (act4 seeds 7/21, act5, act4 1.5M): COF-CUP picked act4 seed 21 (train 134.3 vs 136.5) but test went
  −11% → −56%; TEA improved to −36%. **4/7, −20.4%.**
- Lesson: on steady, low-variance SKUs a tuned (s,S) is near-optimal and tiny train-cost differences between
  seeds are noise. A wider action range helps more than a narrower, finer one.

---

## Open questions / next experiments
- **Steady SKUs (21977, TEA, COF-CUP, MUG):** try a hybrid action (agent chooses (s,S) parameters or an order-up-to
  level) instead of raw quantities.
- **Demand drop regime (wins only 4/10 at −30%):** domain randomisation, i.e. train with `demand_scale` sampled in
  [0.7, 1.3] per episode (the env already supports `demand_scale`).
- **Selection noise:** score candidates on several bootstrap replays of the train period, not one replay, so
  promotion isn't decided by noise (E8, E10).
- **Censored demand:** sales on stockout days under-count demand; impute before training.
- **Run a clean CI retrain** now that the staging bug is fixed, and check all 10 SKUs report PROMOTED/KEPT.
