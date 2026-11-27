import numpy as np
import pandas as pd

from canteen import monitoring


def test_psi_zero_for_same_distribution():
    x = pd.Series(np.random.default_rng(1).normal(size=2000))
    assert monitoring.psi(x, x) < 1e-9


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(1)
    assert monitoring.psi(pd.Series(rng.normal(0, 1, 2000)), pd.Series(rng.normal(3, 1, 2000))) > 1.0


def test_promotion_rule_needs_five_percent():
    assert monitoring.decide(100, 95, 0.05) == "promote"
    assert monitoring.decide(100, 95.1, 0.05) == "keep"
    assert monitoring.decide(100, 120, 0.05) == "keep"


def test_split_later_is_by_date_70_30(params):
    later = pd.DataFrame({"date": pd.date_range("2026-07-01", periods=100)[::-1].strftime("%Y-%m-%d"), "x": range(100)})
    first, check = monitoring.split_later(later, params)
    assert (len(first), len(check)) == (70, 30)
    assert first["date"].max() < check["date"].min()
