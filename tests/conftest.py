"""Tests build their own small data: the CSVs are not in git."""
import numpy as np
import pandas as pd
import pytest

from canteen.config import load_params


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture()
def raw_df():
    """A small messy frame with a sentinel, a duplicate and a stuck price window."""
    rng = np.random.default_rng(0)
    n = 120
    dates = pd.date_range("2026-01-01", periods=n // 4).repeat(4)
    df = pd.DataFrame({
        "record_id": [f"R{i:05d}" for i in range(n)],
        "date": dates.strftime("%Y-%m-%d"),
        "students_on_campus": rng.integers(800, 6000, n),
        "menu_price_rs": rng.integers(30, 120, n),
        "temp_c": rng.uniform(10, 45, n).round(1),
        "exam_days_left": rng.integers(0, 60, n),
        "special_dishes": rng.integers(0, 7, n),
        "queue_minutes_yesterday": rng.uniform(2, 40, n).round(1),
        "day_type": rng.choice(["weekday", "saturday", "holiday"], n),
        "block": rng.choice(["A-Block", "B-Block", "Hostel"], n),
    })
    df["plates_sold"] = (600 - 10 * df["queue_minutes_yesterday"] + rng.normal(0, 5, n)).round().astype(int)
    df.loc[5:14, "exam_days_left"] = 999                       # sentinel
    stuck = (df["date"] >= "2026-01-10") & (df["date"] <= "2026-01-20")
    df.loc[stuck, "menu_price_rs"] = 72                          # stuck sensor, 11 days
    df = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)  # 3 duplicate rows
    return df
