"""Run the whole pipeline end to end:

    python -m canteen.pipeline

data checks -> cleaning -> v1 (5 MLflow runs) -> drift on later.csv -> v2 ->
promotion rule -> champion -> holdout predictions -> findings.yaml
"""
from __future__ import annotations

import json
import os

import pandas as pd
import yaml

from . import cleaning, monitoring, predict, train
from .config import ROOT, load_params, resolve, sha256_file


def fill_findings(params: dict, issues: dict, tracking: dict, mon: dict) -> None:
    path = resolve(params["paths"]["findings"])
    f = yaml.safe_load(path.read_text())
    for kind, vals in issues.items():
        for k in f["data_issues"][kind]:
            f["data_issues"][kind][k] = vals.get(k)
    f["tracking"].update(tracking)
    f["monitoring"].update(mon)
    path.write_text(
        "# MP064 findings, filled in by `python -m canteen.pipeline` from our own data.\n"
        + yaml.safe_dump(f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    )


def main() -> dict:
    os.chdir(ROOT)
    params = load_params()
    p = params["paths"]
    sha = sha256_file(resolve(p["train"]))
    raw_train, raw_later, raw_hold = (pd.read_csv(resolve(p[k])) for k in ("train", "later", "holdout"))

    rules = cleaning.learn_rules(raw_train, raw_later, raw_hold, params)
    cleaning.save_rules(rules, resolve(p["cleaning_rules"]))
    resolve(p["data_issues"]).write_text(json.dumps(rules["issues"], indent=2, default=str))
    train_c = cleaning.clean(raw_train, rules, params, is_train=True)
    later_c = cleaning.clean(raw_later, rules, params, is_train=False)
    hold_c = cleaning.clean(raw_hold, rules, params, is_train=False)
    assert len(hold_c) == len(raw_hold), "holdout rows must never be dropped"

    train.setup_mlflow(params)
    v1 = train.train_v1(train_c, params, sha)
    best = v1["best"]

    # drift on later.csv
    psi = monitoring.psi_table(train_c, later_c, params)
    most_drifted = max(psi, key=psi.get)
    by_group = monitoring.mae_by_group(best["model"], later_c, params)
    worst = max(by_group, key=by_group.get)
    mae_later = monitoring.mae(later_c[params["data"]["target"]],
                               best["model"].predict(later_c[[*params["data"]["numeric_features"], *params["data"]["categorical_features"]]]))
    from mlflow import MlflowClient
    MlflowClient().log_metric(best["run_id"], "mae_later", mae_later)

    # v2 and the promotion rule
    first, check = monitoring.split_later(later_c, params)
    v2 = train.train_v2(train_c, first, check, best["model"], params, sha)
    decision = monitoring.decide(v2["mae_v1"], v2["mae_v2"], params["monitoring"]["promote_min_improvement"])
    champion_version = v2["version"] if decision == "promote" else best["version"]
    champion_model = v2["model"] if decision == "promote" else best["model"]
    if decision == "promote":
        train.promote(v2["version"], params)

    # make sure the exported model is exactly what the registry serves
    served = train.load_champion(params)
    preds = predict.export(served, hold_c, params)

    mon = {
        "psi": {k: round(v, 4) for k, v in psi.items()},
        "most_drifted_feature": most_drifted,
        "mae_by_group": {k: round(v, 2) for k, v in sorted(by_group.items())},
        "worst_served_group": worst,
        "mae_v1_on_check": round(v2["mae_v1"], 2),
        "mae_v2_on_check": round(v2["mae_v2"], 2),
        "decision": decision,
    }
    tracking = {
        "experiment_name": params["mlflow"]["experiment_name"],
        "best_run_id": best["run_id"],
        "registered_model_name": params["mlflow"]["registered_model_name"],
        "champion_version": int(champion_version),
    }
    fill_findings(params, rules["issues"], tracking, mon)
    report = {"tracking": tracking, "monitoring": mon, "mae_v1_on_later": mae_later,
              "val_mae_v1": {r["name"]: round(r["mae_val"], 3) for r in v1["all"]},
              "v1_setting": best["name"], "v2_setting": v2["setting"],
              "inner_val_mae_v2": {k: round(v, 3) for k, v in v2["inner_scores"].items()},
              "rows": {"train_clean": len(train_c), "later": len(later_c), "holdout_predictions": len(preds)}}
    resolve(p["monitoring_report"]).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
