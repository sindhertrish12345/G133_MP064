"""Drift monitoring: PSI per numeric column and MAE per group."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import feature_columns, mae


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with `bins` quantile bins taken from `expected`.

    PSI = sum((a% - e%) * ln(a% / e%)). Empty bins are floored at 1e-4.
    """
    e = pd.Series(expected).dropna().to_numpy(dtype=float)
    a = pd.Series(actual).dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(e, edges)[0] / len(e)
    a_pct = np.histogram(a, edges)[0] / len(a)
    e_pct, a_pct = np.clip(e_pct, 1e-4, None), np.clip(a_pct, 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_table(train: pd.DataFrame, later: pd.DataFrame, params: dict) -> dict:
    bins = params["monitoring"]["psi_bins"]
    return {c: psi(train[c], later[c], bins) for c in params["data"]["numeric_features"]}


def mae_by_group(model, df: pd.DataFrame, params: dict) -> dict:
    d = params["data"]
    pred = model.predict(df[feature_columns(params)])
    res = pd.DataFrame({"g": df[d["group_column"]].to_numpy(), "err": np.abs(df[d["target"]].to_numpy() - pred)})
    return {str(g): float(v) for g, v in res.groupby("g")["err"].mean().items()}


def split_later(later: pd.DataFrame, params: dict):
    """Sort by date; first 70% for retraining, last 30% for the comparison."""
    d = params["data"]
    s = later.sort_values(d["date_column"], kind="stable").reset_index(drop=True)
    cut = int(round(len(s) * params["monitoring"]["retrain_fraction"]))
    return s.iloc[:cut].copy(), s.iloc[cut:].copy()


def decide(mae_v1: float, mae_v2: float, min_improvement: float) -> str:
    """Promote v2 only if its MAE is at least `min_improvement` lower than v1's."""
    return "promote" if mae_v2 <= (1 - min_improvement) * mae_v1 else "keep"


__all__ = ["psi", "psi_table", "mae_by_group", "split_later", "decide", "mae"]
