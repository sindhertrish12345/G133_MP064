"""Predict the holdout file with the champion and export the model for the API."""
from __future__ import annotations

import joblib
import pandas as pd

from .config import resolve
from .features import feature_columns


def predict_holdout(model, holdout_clean: pd.DataFrame, params: dict) -> pd.DataFrame:
    d = params["data"]
    pred = model.predict(holdout_clean[feature_columns(params)])
    return pd.DataFrame({d["id_column"]: holdout_clean[d["id_column"]].to_numpy(), "prediction": pred.round(2)})


def export(model, holdout_clean: pd.DataFrame, params: dict) -> pd.DataFrame:
    p = params["paths"]
    resolve(p["model"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, resolve(p["model"]))
    out = predict_holdout(model, holdout_clean, params)
    resolve(p["predictions"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(resolve(p["predictions"]), index=False)
    return out
