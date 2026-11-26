import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID = {
    "students_on_campus": 3500, "menu_price_rs": 70.0, "temp_c": 30.0, "exam_days_left": 20,
    "special_dishes": 3, "queue_minutes_yesterday": 20.0, "day_type": "weekday", "block": "A-Block",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_a_number():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], float) and 0 < r.json()["prediction"] < 2000


@pytest.mark.parametrize("field,value", [("queue_minutes_yesterday", 70.1), ("temp_c", 4.9)])
def test_client_rules_return_422(field, value):
    r = client.post("/predict", json={**VALID, field: value})
    assert r.status_code == 422


@pytest.mark.parametrize("field,value", [("queue_minutes_yesterday", 70), ("temp_c", 5)])
def test_client_rule_boundaries_are_allowed(field, value):
    assert client.post("/predict", json={**VALID, field: value}).status_code == 200


def test_missing_extra_and_bad_category_are_422():
    missing = {k: v for k, v in VALID.items() if k != "block"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**VALID, "surprise": 1}).status_code == 422
    assert client.post("/predict", json={**VALID, "block": "C-Block"}).status_code == 422


def test_prediction_is_logged_as_json_line(caplog):
    import json
    import logging

    from app.logging_config import JsonFormatter

    with caplog.at_level(logging.INFO, logger="mp064-api"):
        caplog.handler.setFormatter(JsonFormatter())
        client.post("/predict", json=VALID)
    records = [r for r in caplog.records if getattr(r, "event", "") == "predict"]
    assert records, "no prediction log record"
    line = JsonFormatter().format(records[-1])
    rec = json.loads(line)  # one valid JSON object per line
    assert rec["event"] == "predict" and "prediction" in rec and "latency_ms" in rec and "\n" not in line
