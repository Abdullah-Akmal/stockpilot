"""Generate a realistic dummy shop in the LIVE sheet schema, for testing StockPilot's LIVE mode.

    python scripts/make_demo_data.py            # writes data/demo/demo_shop.csv (+ a deliberately broken copy)

"Corner Coffee & Goods" is fictional: 7 SKUs, one year of daily rows ending yesterday, with weekly seasonality,
trends, promotions, a December peak, different lead times and prices. Stock is run by a simple, imperfect
reorder rule, so there are real stockouts (units_sold = what stock allowed, not full demand), like real shop data.
One SKU was launched 20 days ago to exercise the "not enough history" path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# sku, base daily demand, weekday profile (Mon..Sun), price, unit cost, lead time, trend %/yr, noise (neg-bin k)
SKUS = [
    ("COF-BEANS-1KG", 18, [0.9, 0.9, 1.0, 1.0, 1.2, 1.4, 0.6], 24.00, 13.50, 5, 0.15, 6),
    ("COF-CUP-12OZ", 140, [1.1, 1.1, 1.0, 1.0, 1.1, 0.9, 0.5], 0.35, 0.12, 3, 0.05, 15),
    ("MUG-CERAMIC", 6, [0.7, 0.8, 0.9, 1.0, 1.2, 1.6, 0.9], 12.50, 5.20, 7, 0.00, 2),
    ("TEA-EARLGREY", 25, [1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 0.8], 6.80, 3.10, 2, -0.10, 8),
    ("SYRUP-VANILLA", 9, [0.8, 0.9, 1.0, 1.0, 1.1, 1.4, 0.8], 8.90, 4.00, 4, 0.30, 3),
    ("FILTER-PAPERS", 30, [1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7], 3.20, 1.10, 3, 0.00, 2),  # lumpy: cafes buy in bulk
    ("GIFT-SET-XMAS", 4, [0.8, 0.8, 0.9, 1.0, 1.2, 1.7, 1.2], 34.00, 17.00, 6, 0.00, 2),
]
NEW_SKU = ("COLD-BREW-CAN", 22, [1.0, 1.0, 1.0, 1.1, 1.2, 1.5, 1.0], 3.50, 1.40, 2, 0.0, 5)


def demand_series(dates: pd.DatetimeIndex, base, week, trend, k, rng, sku):
    t = np.arange(len(dates)) / 365.0
    dow = np.array(week)[dates.dayofweek]
    season = 1 + 0.12 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365)          # gentle annual cycle
    mu = base * dow * season * (1 + trend * t)
    if sku == "GIFT-SET-XMAS":                                                     # strong Nov-Dec peak
        mu *= np.where(dates.month == 12, 6.0, np.where(dates.month == 11, 3.0, 0.6))
    else:
        mu *= np.where((dates.month == 12) & (dates.day <= 24), 1.35, 1.0)
    promo = np.zeros(len(dates))                                                   # ~monthly 3-day promotions
    for s in rng.choice(len(dates) - 3, size=len(dates) // 30, replace=False):
        promo[s:s + 3] = 1
    mu *= 1 + 0.6 * promo
    if sku == "FILTER-PAPERS":                                                     # occasional bulk buys
        mu = mu + (rng.random(len(dates)) < 0.05) * base * 8
    p = k / (k + mu)
    return rng.negative_binomial(k, p).astype(int), promo


def run_shop(demand: np.ndarray, lead: int, rng, review_day: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Imperfect manual reordering: weekly-ish review, order-up-to based on last 4 weeks, sometimes forgotten."""
    n = len(demand)
    sold, on_hand, on_order = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    stock = int(demand[:14].mean() * (lead + 7))
    pipe = [0] * lead
    for t in range(n):
        stock += pipe.pop(0) if lead else 0
        pipe.append(0) if lead else None
        sold[t] = min(stock, demand[t])
        stock -= sold[t]
        avg = demand[max(0, t - 27): t + 1].mean()
        if (t + review_day) % 7 == 0 and rng.random() > 0.1:                       # weekly review per supplier, 10% forgotten
            target = avg * (lead + 9)
            q = max(int(round(target - stock - sum(pipe))), 0)
            if lead:
                pipe[-1] += q
            else:
                stock += q
        on_hand[t], on_order[t] = stock, sum(pipe)
    return sold, on_hand, on_order


def build(end: pd.Timestamp, days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i, spec in enumerate(SKUS + [NEW_SKU]):
        sku, base, week, price, cost, lead, trend, k = spec
        n = 20 if spec is NEW_SKU else days
        dates = pd.date_range(end=end, periods=n, freq="D")
        dmd, _ = demand_series(dates, base, week, trend, k, rng, sku)
        sold, oh, oo = run_shop(dmd, lead, rng, review_day=i % 7)
        frames.append(pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"), "sku": sku, "units_sold": sold, "on_hand": oh,
            "unit_cost": cost, "price": price, "lead_time_days": lead, "on_order": oo}))
    return pd.concat(frames, ignore_index=True)


def broken_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Same data with typical sheet mistakes, to check the app's validation messages."""
    b = df.drop(columns=["price"]).astype({"on_hand": object, "date": object})  # missing column
    b.loc[3, "date"] = "31/02/2026"                              # impossible date
    b.loc[10, "units_sold"] = -4                                 # negative sales
    b.loc[11, "on_hand"] = "n/a"                                 # text in a number column
    b = pd.concat([b, b.iloc[[20]]])                             # duplicate row
    return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None, help="last date (default: yesterday)")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    end = pd.Timestamp(a.end) if a.end else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    df = build(end, a.days, a.seed)
    out = ROOT / "data" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "demo_shop.csv", index=False)
    broken_copy(df).to_csv(out / "demo_shop_broken.csv", index=False)
    print(f"Wrote data/demo/demo_shop.csv ({len(df)} rows, {df.sku.nunique()} SKUs, {df.date.min()} .. {df.date.max()})")
    print("Wrote data/demo/demo_shop_broken.csv (deliberate mistakes for testing validation)")


if __name__ == "__main__":
    main()
