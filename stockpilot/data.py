"""StockPilot data pipeline.

Two sources, one schema:
  LIVE   - a Google Sheet published as CSV (or read with a gspread service account)
  PUBLIC - UCI Online Retail II, cleaned and aggregated to daily units per SKU

Schema (one row per date x sku):
  date, sku, units_sold, on_hand, unit_cost, price, lead_time_days   (+ optional on_order)
  on_hand  = units in stock at the END of that day (after sales)
  on_order = units ordered but not yet received at the end of that day (optional)
"""
from __future__ import annotations

import argparse
import io
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import load_config, resolve

REQUIRED_COLUMNS = ["date", "sku", "units_sold", "on_hand", "unit_cost", "price", "lead_time_days"]
OPTIONAL_COLUMNS = ["on_order"]
PRODUCT_CODE = re.compile(r"^\d{5}[A-Za-z]{0,2}$")


# --------------------------------------------------------------------------- validation
def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return human-readable problems with a sales table (empty list = valid)."""
    errors: list[str] = []
    cols = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}. "
                      f"Expected: {', '.join(REQUIRED_COLUMNS)}.")
    if df.empty:
        return errors + ["The sheet has no data rows."]
    d = df.copy()
    d.columns = cols
    if "date" not in d or "sku" not in d:
        return errors          # cannot check rows without date and sku
    dates = pd.to_datetime(d["date"], errors="coerce", format="mixed")
    if dates.isna().any():
        bad = d.loc[dates.isna(), "date"].astype(str).head(3).tolist()
        errors.append(f"{int(dates.isna().sum())} row(s) have an unreadable date (e.g. {bad}). Use YYYY-MM-DD.")
    if d["sku"].isna().any() or (d["sku"].astype(str).str.strip() == "").any():
        errors.append("Some rows have an empty sku.")
    for col in [c for c in ["units_sold", "on_hand", "unit_cost", "price", "lead_time_days"] if c in d]:
        num = pd.to_numeric(d[col], errors="coerce")
        if num.isna().any():
            errors.append(f"Column '{col}' has {int(num.isna().sum())} non-numeric or empty value(s).")
        elif (num < 0).any():
            errors.append(f"Column '{col}' has {int((num < 0).sum())} negative value(s).")
    lt = pd.to_numeric(d["lead_time_days"], errors="coerce") if "lead_time_days" in d else pd.Series(dtype=float)
    if lt.notna().all() and (lt > 14).any():
        errors.append("lead_time_days must be at most 14.")
    ok = dates.notna()
    if ok.any():
        dup = d[ok].assign(date=dates[ok]).duplicated(subset=["date", "sku"]).sum()
        if dup:
            errors.append(f"{int(dup)} duplicate (date, sku) row(s). Keep one row per SKU per day.")
    return errors


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a validated table to canonical dtypes and order."""
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["sku"] = d["sku"].astype(str).str.strip()
    for col in ["units_sold", "on_hand", "unit_cost", "price", "lead_time_days"] + \
            [c for c in OPTIONAL_COLUMNS if c in d.columns]:
        d[col] = pd.to_numeric(d[col])
    d["lead_time_days"] = d["lead_time_days"].round().astype(int)
    keep = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in d.columns]
    return d[keep].sort_values(["sku", "date"]).reset_index(drop=True)


class DataError(Exception):
    """Raised with a user-facing message when a data source cannot be used."""


# --------------------------------------------------------------------------- LIVE
def read_live_csv(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        raise DataError(f"Could not fetch the published sheet CSV: {e}") from e
    if "text/html" in r.headers.get("content-type", "") and b"<html" in r.content[:500].lower():
        raise DataError("The URL returned a web page, not CSV. In Google Sheets use "
                        "File > Share > Publish to web > (sheet) > Comma-separated values (.csv).")
    return pd.read_csv(io.BytesIO(r.content), dtype={"sku": str})


def read_live_gspread(service_account_info: dict, sheet_key: str, worksheet: str | None = None) -> pd.DataFrame:
    import gspread  # optional dependency path

    gc = gspread.service_account_from_dict(dict(service_account_info))
    sh = gc.open_by_key(sheet_key)
    ws = sh.worksheet(worksheet) if worksheet else sh.sheet1
    return pd.DataFrame(ws.get_all_records())


def load_live(url: str | None = None, service_account_info: dict | None = None,
              sheet_key: str | None = None, worksheet: str | None = None) -> pd.DataFrame:
    if service_account_info and sheet_key:
        raw = read_live_gspread(service_account_info, sheet_key, worksheet)
    elif url:
        raw = read_live_csv(url)
    else:
        raise DataError("LIVE mode needs a published CSV URL (secret live.csv_url) "
                        "or a gspread service account + sheet_key.")
    errors = validate_schema(raw)
    if errors:
        raise DataError("The live sheet failed validation:\n- " + "\n- ".join(errors))
    return normalise(raw)


# --------------------------------------------------------------------------- PUBLIC
def download_uci(cfg: dict) -> Path:
    raw_dir = resolve(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    zpath = raw_dir / "online_retail_ii.zip"
    if not (zpath.exists() and zipfile.is_zipfile(zpath)):
        print(f"Downloading {cfg['data']['uci_url']} ...")
        for attempt in range(5):
            try:
                with requests.get(cfg["data"]["uci_url"], stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(zpath, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                if zipfile.is_zipfile(zpath):
                    break
            except requests.RequestException as e:
                print(f"  attempt {attempt + 1} failed: {e}")
        else:
            raise DataError("Could not download the UCI Online Retail II dataset.")
    return zpath


def read_uci_transactions(cfg: dict) -> pd.DataFrame:
    """Read both sheets of the UCI workbook (cached as parquet after the first read)."""
    raw_dir = resolve(cfg["data"]["raw_dir"])
    cache = raw_dir / "online_retail_ii.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    zpath = download_uci(cfg)
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
        print(f"Reading {name} (both sheets, takes a few minutes the first time) ...")
        sheets = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    tx = pd.concat(sheets.values(), ignore_index=True)
    tx = tx.rename(columns={"Customer ID": "CustomerID"})
    tx["Invoice"] = tx["Invoice"].astype(str)
    tx["StockCode"] = tx["StockCode"].astype(str).str.strip().str.upper()
    tx["CustomerID"] = tx["CustomerID"].astype("float").astype("Int64").astype(str)
    tx["Description"] = tx["Description"].astype(str)
    tx.to_parquet(cache, index=False)
    return tx


def clean_transactions(tx: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop cancellations/returns (and the sales they cancel), keep real product lines."""
    stats = {"raw_rows": len(tx)}
    tx = tx.drop_duplicates()
    is_cancel = tx["Invoice"].str.startswith("C") | (tx["Quantity"] < 0)
    cancels = tx[is_cancel]
    stats["cancellation_rows"] = int(is_cancel.sum())
    sales = tx[~is_cancel & (tx["Quantity"] > 0) & (tx["Price"] > 0)].copy()
    # Remove the original sale lines that a cancellation reverses (same customer, item, qty).
    ckey = cancels.assign(Quantity=cancels["Quantity"].abs())[["CustomerID", "StockCode", "Quantity"]]
    ckey = ckey[ckey["CustomerID"] != "<NA>"].drop_duplicates()
    before = len(sales)
    sales = sales.merge(ckey.assign(_c=1), on=["CustomerID", "StockCode", "Quantity"], how="left")
    sales = sales[sales["_c"].isna()].drop(columns="_c")
    stats["sales_removed_as_cancelled"] = before - len(sales)
    sales = sales[sales["StockCode"].str.match(PRODUCT_CODE)]
    stats["clean_rows"] = len(sales)
    return sales, stats


def build_public_daily(cfg: dict) -> tuple[pd.DataFrame, dict]:
    tx = read_uci_transactions(cfg)
    sales, stats = clean_transactions(tx)
    sales["date"] = pd.to_datetime(sales["InvoiceDate"]).dt.normalize()
    calendar = np.sort(sales["date"].unique())          # store trading days
    stats["trading_days"] = len(calendar)
    stats["date_range"] = f"{pd.Timestamp(calendar[0]).date()} .. {pd.Timestamp(calendar[-1]).date()}"

    sales["revenue"] = sales["Quantity"] * sales["Price"]
    daily = sales.groupby(["StockCode", "date"]).agg(
        units_sold=("Quantity", "sum"), revenue=("revenue", "sum")).reset_index()

    dcfg = cfg["data"]
    per_sku = daily.groupby("StockCode").agg(units=("units_sold", "sum"), days=("date", "nunique"),
                                             first=("date", "min"))
    span = per_sku["first"].map(lambda d: int((calendar >= d).sum()))
    per_sku["coverage"] = per_sku["days"] / span
    eligible = per_sku[(span >= dcfg["min_trading_days"]) & (per_sku["coverage"] >= dcfg["min_coverage"])]
    top = eligible.sort_values("units", ascending=False).head(dcfg["n_skus"])
    stats["eligible_skus"] = len(eligible)

    desc = sales.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0])
    frames = []
    cal = pd.DatetimeIndex(calendar)
    lt = int(cfg["costs"]["lead_time_days"])
    for sku in top.index:
        d = daily[daily["StockCode"] == sku].set_index("date")
        d = d.reindex(cal[cal >= d.index.min()])
        units = d["units_sold"].fillna(0).astype(int)
        price = (d["revenue"] / d["units_sold"]).ffill().bfill()
        # The dataset has no inventory records: simulate on-hand with a simple base-stock rule.
        on_hand, on_order = simulate_history_inventory(units.to_numpy(), lt)
        frames.append(pd.DataFrame({
            "date": units.index, "sku": sku, "units_sold": units.values,
            "on_hand": on_hand, "on_order": on_order,
            "unit_cost": np.round(price.values * cfg["costs"]["unit_cost_frac"], 4),
            "price": np.round(price.values, 4), "lead_time_days": lt,
        }))
    out = pd.concat(frames, ignore_index=True)
    out = normalise(out)
    stats["descriptions"] = {k: desc[k] for k in top.index}
    return out, stats


def simulate_history_inventory(units: np.ndarray, lead_time: int) -> tuple[np.ndarray, np.ndarray]:
    """Plausible END-OF-DAY on-hand / on-order for a dataset without stock records (MA base-stock rule)."""
    on_hand = np.zeros(len(units), dtype=int)
    on_order = np.zeros(len(units), dtype=int)
    stock = int(round(units[:28].mean() * (lead_time + 2)))
    pipe = [0] * max(lead_time, 1)
    for t, dmd in enumerate(units):
        stock += pipe.pop(0)
        pipe.append(0)
        stock = max(stock - int(dmd), 0)
        ma = units[max(0, t - 27): t + 1].mean()
        target = int(round(ma * (lead_time + 2)))
        pipe[-1] += max(target - stock - sum(pipe), 0)
        on_hand[t] = stock
        on_order[t] = sum(pipe)
    return on_hand, on_order


# --------------------------------------------------------------------------- loading API
@dataclass
class Dataset:
    df: pd.DataFrame
    source: str          # "live", "public", or "sample"
    message: str = ""


def load_dataset(mode: str, cfg: dict | None = None, live_url: str | None = None,
                 service_account_info: dict | None = None, sheet_key: str | None = None) -> Dataset:
    """Load data for a mode, falling back public -> sample so the app always runs."""
    cfg = cfg or load_config()
    if mode == "live":
        url = live_url or os.environ.get(cfg["data"]["live"]["csv_url_env"])
        return Dataset(load_live(url, service_account_info, sheet_key), "live")
    p = resolve(cfg["data"]["processed_path"])
    if p.exists():
        return Dataset(normalise(pd.read_csv(p, dtype={"sku": str})), "public")
    s = resolve(cfg["data"]["sample_path"])
    return Dataset(normalise(pd.read_csv(s, dtype={"sku": str})), "sample",
                   "Processed public data not found; using the bundled sample.")


def chronological_split(df: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split on a single global date cut: first train_frac of dates -> train, rest -> test."""
    dates = np.sort(df["date"].unique())
    cut = pd.Timestamp(dates[int(len(dates) * train_frac)])
    return df[df["date"] < cut].copy(), df[df["date"] >= cut].copy(), cut


def sku_series(df: pd.DataFrame, sku: str) -> pd.DataFrame:
    return df[df["sku"] == sku].sort_values("date").reset_index(drop=True)


def sku_params(train_sku: pd.DataFrame, cfg: dict) -> dict:
    """Per-SKU economics and demand statistics, computed from the TRAIN slice only."""
    c = cfg["costs"]
    price = float(train_sku["price"].median())
    unit_cost = float(train_sku["unit_cost"].median())
    margin = max(price - unit_cost, 1e-6)
    mean_d = float(train_sku["units_sold"].mean())
    return {
        "price": price, "unit_cost": unit_cost, "margin": margin,
        "holding_cost": c["holding_rate_annual"] / 365.0 * unit_cost,
        "order_cost": float(c["order_cost"]),
        "stockout_penalty": margin + c["goodwill_frac"] * price,   # lost margin + goodwill
        "goodwill": c["goodwill_frac"] * price,
        "lead_time": int(train_sku["lead_time_days"].iloc[-1]),
        "shelf_life": c.get("shelf_life_days"),
        "mean_demand": max(mean_d, 1e-3),
        "std_demand": float(train_sku["units_sold"].std(ddof=0)),
    }


@dataclass
class SkuContext:
    sku: str
    params: dict
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_demand(self) -> np.ndarray:
        return self.train["units_sold"].to_numpy()

    @property
    def train_dows(self) -> np.ndarray:
        return self.train["date"].dt.dayofweek.to_numpy()

    def train_replay(self, H: int = 14) -> dict:
        """Train period as a replay (first H days are the initial history)."""
        d = self.train_demand
        return {"demand": d[H:], "dows": self.train_dows[H:], "history": d[:H]}

    def test_replay(self, H: int = 14) -> dict:
        """Test period as a replay; history = the last H TRAIN days (no leakage)."""
        return {"demand": self.test["units_sold"].to_numpy(), "dows": self.test["date"].dt.dayofweek.to_numpy(),
                "history": self.train_demand[-H:], "dates": self.test["date"].to_numpy()}


def sku_contexts(df: pd.DataFrame, cfg: dict) -> list[SkuContext]:
    train, test, _ = chronological_split(df, cfg["data"]["train_frac"])
    out = []
    for sku in sorted(df["sku"].unique()):
        tr, te = sku_series(train, sku), sku_series(test, sku)
        if len(tr) < 30 or len(te) < 7:
            continue
        p = sku_params(tr, cfg)
        p["train_lead_time"] = p["lead_time"]
        out.append(SkuContext(sku, p, tr, te))
    return out


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="StockPilot data pipeline")
    ap.add_argument("--mode", choices=["public", "live"], default=None)
    args = ap.parse_args()
    cfg = load_config()
    mode = args.mode or cfg["data"]["mode"]
    if mode == "public":
        df, stats = build_public_daily(cfg)
        out = resolve(cfg["data"]["processed_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        # bundled sample: last 120 trading days, live schema
        sample = df[df["date"] >= np.sort(df["date"].unique())[-120]]
        sp = resolve(cfg["data"]["sample_path"])
        sp.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(sp, index=False)
        descs = stats.pop("descriptions")
        import json
        (out.parent / "sku_descriptions.json").write_text(json.dumps(descs, indent=2))
        for k, v in stats.items():
            print(f"{k:>28}: {v}")
        print(f"Wrote {out.relative_to(resolve('.'))} ({len(df)} rows) and {sp.relative_to(resolve('.'))} ({len(sample)} rows)")
    else:
        df = load_dataset("live", cfg).df
        out = resolve("data/processed/daily_live.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        descs = {}
        print(f"Live sheet OK: {len(df)} rows, {df['sku'].nunique()} SKUs -> {out}")

    train, test, cut = chronological_split(df, cfg["data"]["train_frac"])
    print(f"\nChronological split at {cut.date()}: train {train['date'].min().date()}..{train['date'].max().date()}"
          f" | test {test['date'].min().date()}..{test['date'].max().date()}")
    rows = []
    for sku in df["sku"].unique():
        tr, te = train[train.sku == sku], test[test.sku == sku]
        rows.append({"sku": sku, "description": descs.get(sku, "")[:30], "train_days": len(tr), "test_days": len(te),
                     "mean_daily_units": round(tr["units_sold"].mean(), 1),
                     "cv": round(tr["units_sold"].std() / max(tr["units_sold"].mean(), 1e-9), 2),
                     "zero_days_%": round(100 * (tr["units_sold"] == 0).mean(), 1),
                     "price": round(tr["price"].median(), 2)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
