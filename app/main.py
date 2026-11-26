"""FastAPI service for MP064: POST /predict and GET /health."""
from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from app.logging_config import get_logger

MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).resolve().parents[1] / "models" / "model.joblib"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "champion")
FEATURES = [
    "students_on_campus", "menu_price_rs", "temp_c", "exam_days_left",
    "special_dishes", "queue_minutes_yesterday", "day_type", "block",
]

log = get_logger()
app = FastAPI(title="MP064 College Canteen Lunch Rush", version="1.0")


class PredictRequest(BaseModel):
    """Exactly the eight fields the client sends. The client's two rules live here."""

    model_config = ConfigDict(extra="forbid")

    students_on_campus: int = Field(..., ge=0, description="students who entered campus that morning")
    menu_price_rs: float = Field(..., gt=0, description="price of the main thali in rupees")
    temp_c: float = Field(..., ge=5, le=60, description="noon temperature; below 5 is a typing mistake")
    exam_days_left: int = Field(..., ge=0, description="days left to the next exam")
    special_dishes: int = Field(..., ge=0, description="special dishes on the menu")
    queue_minutes_yesterday: float = Field(..., ge=0, le=70, description="average queue yesterday; above 70 is impossible")
    day_type: Literal["weekday", "saturday", "holiday"]
    block: Literal["A-Block", "B-Block", "Hostel"]


class PredictResponse(BaseModel):
    prediction: float
    model_version: str


@lru_cache(maxsize=1)
def get_model():
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    row = pd.DataFrame([req.model_dump()])[FEATURES]
    value = float(get_model().predict(row)[0])
    log.info(
        "prediction",
        extra={
            "event": "predict",
            "features": req.model_dump(),
            "prediction": round(value, 2),
            "model_version": MODEL_VERSION,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        },
    )
    return PredictResponse(prediction=round(value, 2), model_version=MODEL_VERSION)
