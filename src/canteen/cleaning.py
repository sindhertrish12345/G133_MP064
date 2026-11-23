"""Cleaning rules learned from train.csv and applied to train, later and holdout."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import data_checks


def _key(x) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def learn_rules(train: pd.DataFrame, later: pd.DataFrame, holdout: pd.DataFrame, params: dict) -> dict:
    """Run the seven checks and keep what is needed to clean any file."""
    issues = data_checks.run_all(train, later, holdout, params)
    return {
        "drop_columns": [issues["leaking_column"]["column"]] if issues["leaking_column"]["present"] else [],
        "sentinel": issues["sentinel_value"] if issues["sentinel_value"]["present"] else None,
        "stuck": issues["stuck_sensor"] if issues["stuck_sensor"]["present"] else None,
        "unit": issues["unit_mixup"] if issues["unit_mixup"]["present"] else None,
        "drop_duplicates": issues["duplicate_rows"]["present"],
        "typos": issues["target_typos"]["present"],
        "levels": params["data"]["levels"],
        "issues": issues,
    }


def fix_spelling(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    """Map 'rain ', 'RAI.' style variants onto the real category names."""
    df = df.copy()
    for col, good in levels.items():
        if col in df.columns:
            lookup = {_key(g): g for g in good}
            df[col] = df[col].map(lambda v: lookup.get(_key(v), v))
    return df


def clean(df: pd.DataFrame, rules: dict, params: dict, is_train: bool) -> pd.DataFrame:
    """Return a cleaned copy. Bad values become NaN (imputed later in the model pipeline).

    Row removal (duplicates, target typos) and the stuck-sensor window only
    apply to train.csv, because later/holdout are clean.
    """
    d = params["data"]
    out = df.copy()
    out = out.drop(columns=[c for c in rules["drop_columns"] if c in out.columns])
    if rules["drop_duplicates"]:
        out = out.drop_duplicates()
    out = fix_spelling(out, rules["levels"])
    s = rules["sentinel"]
    if s:
        out.loc[out[s["column"]] == s["value"], s["column"]] = np.nan
    u = rules["unit"]
    if u:
        mask = out[d["group_column"]] == u["group_value"]
        out.loc[mask, u["column"]] = out.loc[mask, u["column"]] / u["factor"]
    if is_train:
        k = rules["stuck"]
        if k:
            dates = pd.to_datetime(out[d["date_column"]])
            window = (dates >= k["start_date"]) & (dates <= k["end_date"])
            out.loc[window, k["column"]] = np.nan
        if rules["typos"]:
            t = d["target"]
            out = out[out[t] <= params["checks"]["typo_ratio"] * out[t].median()]
    out = out.sort_values(d["date_column"], kind="stable").reset_index(drop=True)
    return out


def save_rules(rules: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rules, indent=2, default=str))
