import pandas as pd

from stockpilot.data import chronological_split, normalise, validate_schema


def good_frame():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        "sku": ["A", "A", "B", "B"], "units_sold": [3, 4, 1, 0], "on_hand": [10, 7, 5, 5],
        "unit_cost": [1.0, 1.0, 2.0, 2.0], "price": [2.0, 2.0, 3.5, 3.5], "lead_time_days": [2, 2, 3, 3]})


def test_valid_schema_passes():
    assert validate_schema(good_frame()) == []


def test_missing_column_reported():
    errs = validate_schema(good_frame().drop(columns=["price"]))
    assert errs and "price" in errs[0]


def test_bad_values_reported():
    df = good_frame()
    df.loc[0, "date"] = "not a date"
    df.loc[1, "units_sold"] = -2
    df.loc[2, "price"] = "abc"
    errs = " ".join(validate_schema(df))
    assert "date" in errs and "units_sold" in errs and "price" in errs


def test_duplicates_reported():
    df = pd.concat([good_frame(), good_frame().iloc[[0]]])
    assert any("duplicate" in e for e in validate_schema(df))


def test_split_is_chronological_without_overlap():
    df = normalise(good_frame())
    train, test, cut = chronological_split(df, 0.5)
    assert train["date"].max() < cut <= test["date"].min()
