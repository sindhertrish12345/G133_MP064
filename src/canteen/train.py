"""Train v1 with MLflow tracking, register the best run, and retrain v2."""
from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from .features import build_pipeline, feature_columns, mae


def setup_mlflow(params: dict) -> None:
    m = params["mlflow"]
    mlflow.set_tracking_uri(m["tracking_uri"])
    if mlflow.get_experiment_by_name(m["experiment_name"]) is None:
        mlflow.create_experiment(m["experiment_name"], artifact_location="mlartifacts")
    mlflow.set_experiment(m["experiment_name"])


def _log_model(model, name: str):
    return mlflow.sklearn.log_model(model, name=name, serialization_format="cloudpickle")


def train_v1(train_clean: pd.DataFrame, params: dict, data_sha256: str) -> dict:
    """Try every setting in params.yaml; the lowest validation MAE becomes version 1.

    Each run: fit on the first 80% of cleaned train (by date), measure mae_val on
    the last 20%, then refit on all of cleaned train and log that model.
    """
    m, d = params["mlflow"], params["data"]
    feats, target = feature_columns(params), d["target"]
    df = train_clean.sort_values(d["date_column"], kind="stable")
    cut = int(len(df) * (1 - m["validation_fraction"]))
    fit_part, val_part = df.iloc[:cut], df.iloc[cut:]
    results = []
    for exp in m["experiments"]:
        with mlflow.start_run(run_name=exp["name"]) as run:
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model", exp["model"])
            mlflow.log_params({f"hp_{k}": v for k, v in exp["params"].items()})
            mlflow.log_param("train_rows", len(df))
            mlflow.set_tag("stage", "v1-candidate")
            probe = build_pipeline(exp["model"], exp["params"], params, m["random_state"])
            probe.fit(fit_part[feats], fit_part[target])
            val_mae = mae(val_part[target], probe.predict(val_part[feats]))
            mlflow.log_metric("mae_val", val_mae)
            final = build_pipeline(exp["model"], exp["params"], params, m["random_state"])
            final.fit(df[feats], df[target])
            mlflow.log_metric("mae_train_full", mae(df[target], final.predict(df[feats])))
            _log_model(final, "model")
            results.append({"run_id": run.info.run_id, "name": exp["name"], "mae_val": val_mae, "model": final})
    best = min(results, key=lambda r: r["mae_val"])
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", m["registered_model_name"])
    client = MlflowClient()
    client.set_registered_model_alias(m["registered_model_name"], m["champion_alias"], mv.version)
    client.set_tag(best["run_id"], "stage", "v1-registered")
    best["version"] = int(mv.version)
    return {"best": best, "all": results}


def train_v2(train_clean: pd.DataFrame, later_first: pd.DataFrame, later_check: pd.DataFrame,
             v1_model, params: dict, data_sha256: str) -> dict:
    """Retrain on cleaned train + first 70% of later; compare with v1 on the last 30%.

    The model setting is chosen WITHOUT looking at the last 30%: each setting is
    fitted on train + the first 80% of `later_first` and scored on the last 20%
    of `later_first` (mae_inner_val), then the winner is refitted on everything.
    """
    m, d = params["mlflow"], params["data"]
    feats, target = feature_columns(params), d["target"]
    cut = int(len(later_first) * 0.8)
    inner_fit = pd.concat([train_clean, later_first.iloc[:cut]], ignore_index=True)
    inner_val = later_first.iloc[cut:]
    scores = {}
    for exp in m["experiments"]:
        probe = build_pipeline(exp["model"], exp["params"], params, m["random_state"])
        probe.fit(inner_fit[feats], inner_fit[target])
        scores[exp["name"]] = mae(inner_val[target], probe.predict(inner_val[feats]))
    best_name = min(scores, key=scores.get)
    best_exp = next(e for e in m["experiments"] if e["name"] == best_name)

    combined = pd.concat([train_clean, later_first], ignore_index=True)
    with mlflow.start_run(run_name="v2_retrain") as run:
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model", best_exp["model"])
        mlflow.log_param("setting", best_name)
        mlflow.log_params({f"hp_{k}": v for k, v in best_exp["params"].items()})
        mlflow.log_param("train_rows", len(combined))
        mlflow.set_tag("stage", "v2-retrain")
        for name, score in scores.items():
            mlflow.log_metric(f"mae_inner_val_{name}", score)
        v2 = build_pipeline(best_exp["model"], best_exp["params"], params, m["random_state"])
        v2.fit(combined[feats], combined[target])
        mae_v1 = mae(later_check[target], v1_model.predict(later_check[feats]))
        mae_v2 = mae(later_check[target], v2.predict(later_check[feats]))
        mlflow.log_metric("mae_check_v1", mae_v1)
        mlflow.log_metric("mae_check_v2", mae_v2)
        mlflow.log_metric("mae_check", mae_v2)
        _log_model(v2, "model")
        mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", m["registered_model_name"])
    return {"run_id": run.info.run_id, "version": int(mv.version), "model": v2, "setting": best_name,
            "mae_v1": mae_v1, "mae_v2": mae_v2, "inner_scores": scores}


def promote(version: int, params: dict) -> None:
    m = params["mlflow"]
    MlflowClient().set_registered_model_alias(m["registered_model_name"], m["champion_alias"], version)


def load_champion(params: dict):
    m = params["mlflow"]
    mlflow.set_tracking_uri(m["tracking_uri"])
    return mlflow.sklearn.load_model(f"models:/{m['registered_model_name']}@{m['champion_alias']}")
