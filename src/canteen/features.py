"""The scikit-learn pipeline: imputation + scaling/encoding + a regressor."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELS = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def feature_columns(params: dict) -> list[str]:
    d = params["data"]
    return d["numeric_features"] + d["categorical_features"]


def build_pipeline(model: str, model_params: dict, params: dict, random_state: int = 42) -> Pipeline:
    d = params["data"]
    kwargs = dict(model_params)
    if model != "ridge":
        kwargs["random_state"] = random_state
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", numeric, d["numeric_features"]),
        ("cat", categorical, d["categorical_features"]),
    ])
    return Pipeline([("pre", pre), ("model", MODELS[model](**kwargs))])


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))
