"""Detect the seven kinds of data problem in train.csv.

Every function looks at the data itself (nothing is hard-coded) and returns a
dict with the fields findings.yaml needs. Run it with:

    python -m canteen.data_checks
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import load_params, resolve


# 1 ---------------------------------------------------------------------------
def check_leaking_column(train, holdout, params) -> dict:
    """A column that is missing from the holdout file, or that copies the target."""
    d = params["data"]
    target = d["target"]
    feature_like = [c for c in train.columns if c not in (target,)]
    missing_in_holdout = [c for c in feature_like if c not in holdout.columns]
    suspects = list(missing_in_holdout)
    thr = params["checks"]["leak_corr_threshold"]
    for c in train.select_dtypes("number").columns:
        if c == target:
            continue
        if abs(train[c].corr(train[target])) >= thr and c not in suspects:
            suspects.append(c)
    return {"present": bool(suspects), "column": suspects[0] if suspects else None}


# 2 ---------------------------------------------------------------------------
def check_unit_mixup(train, params) -> dict:
    """One group value whose numbers are ~10/12/60/100/1000x another column's norm."""
    d, ck = params["data"], params["checks"]
    group = d["group_column"]
    cols = [c for c in d["numeric_features"] + [d["target"]] if c in train.columns]
    for col in cols:
        med = train.groupby(group)[col].median()
        for g, m in med.items():
            others = med.drop(g).median()
            if not others or not m:
                continue
            ratio = m / others
            for f in ck["unit_factors"]:
                for r, factor in ((ratio, f), (1 / ratio, 1 / f)):
                    if abs(r - f) / f <= ck["unit_tolerance"]:
                        return {"present": True, "column": col, "group_value": g,
                                "factor": int(f) if factor == f else 1 / f}
    return {"present": False, "column": None, "group_value": None, "factor": None}


# 3 ---------------------------------------------------------------------------
def check_sentinel_value(train, params) -> dict:
    """An impossible number typed in place of a blank (-1, 999, 9999, ...)."""
    ck, d = params["checks"], params["data"]
    for col in d["numeric_features"]:
        s = train[col]
        for cand in ck["sentinel_candidates"]:
            n = int((s == cand).sum())
            if n < ck["sentinel_min_count"]:
                continue
            rest = s[s != cand]
            if cand > rest.max() or cand < rest.min():
                return {"present": True, "column": col, "value": cand, "count": n}
    return {"present": False, "column": None, "value": None, "count": None}


# 4 ---------------------------------------------------------------------------
def check_duplicate_rows(train, params=None) -> dict:
    n = int(train.duplicated().sum())
    return {"present": n > 0, "count": n if n else None}


# 5 ---------------------------------------------------------------------------
def check_stuck_sensor(train, params) -> dict:
    """A numeric column with one single value on every row of a long run of days."""
    d, ck = params["data"], params["checks"]
    df = train.copy()
    df[d["date_column"]] = pd.to_datetime(df[d["date_column"]])
    best = None
    for col in d["numeric_features"]:
        per_day = df.groupby(d["date_column"])[col].agg(["nunique", "first"]).sort_index()
        run_start, prev_day, prev_val, run_len = None, None, None, 0
        for day, row in per_day.iterrows():
            same = (
                row["nunique"] == 1
                and prev_day is not None
                and row["first"] == prev_val
                and (day - prev_day).days == 1
            )
            if same:
                run_len += 1
            else:
                run_start, run_len = day, 1
            if row["nunique"] == 1:
                prev_day, prev_val = day, row["first"]
            else:
                prev_day, prev_val, run_len = None, None, 0
            if run_len >= ck["stuck_min_days"] and (best is None or run_len > best[0]):
                best = (run_len, col, run_start, day, row["first"])
    if best is None:
        return {"present": False, "column": None, "start_date": None, "end_date": None}
    _, col, start, end, val = best
    return {"present": True, "column": col, "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"), "value": float(val)}


# 6 ---------------------------------------------------------------------------
def check_target_typos(train, params) -> dict:
    """Targets typed 10x too big: far above the median, after a big empty gap."""
    t = train[params["data"]["target"]]
    limit = params["checks"]["typo_ratio"] * t.median()
    n = int((t > limit).sum())
    return {"present": n > 0, "count": n if n else None}


# 7 ---------------------------------------------------------------------------
def check_spelling_variants(train, later, params) -> dict:
    """A text column whose values are not the clean spellings seen in later.csv."""
    d = params["data"]
    for col in d["categorical_features"]:
        clean = set(later[col].dropna().unique())
        bad = set(train[col].dropna().unique()) - clean
        if bad:
            return {"present": True, "column": col, "true_levels": len(clean),
                    "variants": sorted(bad), "all_map_to_clean": all(_key(b)[:3] in {k[:3] for k in keys} for b in bad)}
    return {"present": False, "column": None, "true_levels": None}


# -----------------------------------------------------------------------------
def run_all(train: pd.DataFrame, later: pd.DataFrame, holdout: pd.DataFrame, params: dict) -> dict:
    return {
        "leaking_column": check_leaking_column(train, holdout, params),
        "unit_mixup": check_unit_mixup(train, params),
        "sentinel_value": check_sentinel_value(train, params),
        "duplicate_rows": check_duplicate_rows(train, params),
        "stuck_sensor": check_stuck_sensor(train, params),
        "target_typos": check_target_typos(train, params),
        "spelling_variants": check_spelling_variants(train, later, params),
    }


def main() -> dict:
    params = load_params()
    p = params["paths"]
    train, later, hold = (pd.read_csv(resolve(p[k])) for k in ("train", "later", "holdout"))
    issues = run_all(train, later, hold, params)
    out = resolve(p["data_issues"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(issues, indent=2, default=str))
    print(json.dumps(issues, indent=2, default=str))
    return issues


if __name__ == "__main__":
    main()
