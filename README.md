# MP064 — College Canteen Lunch Rush (Nagpur)

**Group:** G133 &nbsp;|&nbsp; **Problem:** MP064 &nbsp;|&nbsp; **Course:** SCSE3040 MLOps mini project &nbsp;|&nbsp; **Faculty:** Dr. Ravi Pratap &nbsp;|&nbsp; **Batch:** 18

| Enrolment no. | Member |
|---|---|
| S24CSEU0122 | Aviral Agrawal |
| S24CSEU1285 | Anubhav Singh |
| S24CSEU1278 | Mitaksh Jhanwar |
| S24CSEU0238 | Samaseni Reddy Yaswanth |
| S24CSEU0081 | Trish |

Predict `plates_sold` (plates sold between 12 and 3 pm) for the Nagpur Institute Canteen
Committee. Regression, measured with **MAE**. The whole MLOps pipeline is here: DVC-tracked
data, seven data checks, cleaning, MLflow tracking and registry, a FastAPI service, Docker,
GitHub Actions, Kubernetes files, JSON logs, drift monitoring and a retrain.

## What we found in `train.csv` (all from `src/canteen/data_checks.py`)

| Problem | Result |
|---|---|
| leaking_column | not present (every column also exists in `holdout_features.csv`) |
| unit_mixup | not present (per-`block` medians agree for every column) |
| sentinel_value | **present**: `exam_days_left = 999`, 72 rows |
| duplicate_rows | **present**: 25 extra copies |
| stuck_sensor | **present**: `menu_price_rs` stuck at 72 from 2026-03-24 to 2026-04-09 |
| target_typos | not present (largest `plates_sold` is 673, no gap) |
| spelling_variants | not present (`day_type` and `block` already use the clean spellings) |

Cleaning (`src/canteen/cleaning.py`): drop duplicates, turn 999 into a missing value, blank the
stuck prices; the model pipeline imputes missing values with the median.

## Results

| Step | Result |
|---|---|
| v1 (Ridge, chosen by validation MAE over 6 settings) on `later.csv` | MAE 43.4 |
| Most drifted column (PSI, 10 bins) | `queue_minutes_yesterday` (PSI 3.63) |
| v1 MAE per `block` on `later.csv` | A-Block 34.1, B-Block 34.4, **Hostel 64.9** (worst) |
| v1 vs v2 on the last 30% of `later.csv` | 43.93 vs 27.21 (38% lower) |
| Promotion rule (needs at least 5% lower) | **promote** v2, now `@champion` |

Full numbers: `reports/findings.yaml` and `reports/monitoring.json`.

## Repository layout

```
params.yaml              all settings, read by the code with yaml.safe_load
src/canteen/             data_checks, cleaning, features, train, monitoring, predict, pipeline
app/main.py              FastAPI: POST /predict, GET /health (client rules in the pydantic schema)
app/logging_config.py    JSON logging to stdout
tests/                   pytest, builds its own small data (CSVs are not in git)
data/raw/*.dvc           DVC pointers for train.csv and later.csv
mlflow.db                MLflow tracking + registry (7 runs, 2 model versions, alias champion = v2)
models/model.joblib      the champion model the API loads
predictions/             holdout_predictions.csv (record_id,prediction)
k8s/                     deployment.yaml (mp064-api, 3 replicas), service.yaml (mp064-svc, 80 -> 8000)
Dockerfile               image mp064-api:1.0
.github/workflows/ci.yml pytest on every push
docs/                    problem statement, handbook, TEAM_PLAN.md
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# data: pull the DVC-tracked CSVs (or copy train.csv / later.csv from the zip into data/raw/)
dvc pull            # only if a DVC remote is configured

python -m canteen.data_checks      # the seven data checks -> reports/data_issues.json
python -m canteen.pipeline         # clean -> v1 -> drift -> v2 -> champion -> predictions -> findings.yaml
pytest                             # 20 tests, none need the CSVs

uvicorn app.main:app --port 8000   # then open http://localhost:8000/docs
```

Run these from the repository root (MLflow uses `sqlite:///mlflow.db`).

## API

`POST /predict` takes exactly: `students_on_campus`, `menu_price_rs`, `temp_c`, `exam_days_left`,
`special_dishes`, `queue_minutes_yesterday`, `day_type`, `block`. Client rules, enforced by the
schema (HTTP 422): `queue_minutes_yesterday <= 70` and `temp_c >= 5`.

```bash
curl -X POST localhost:8000/predict -H "content-type: application/json" -d '{
  "students_on_campus": 3500, "menu_price_rs": 70, "temp_c": 30, "exam_days_left": 20,
  "special_dishes": 3, "queue_minutes_yesterday": 20, "day_type": "weekday", "block": "A-Block"}'
```

## Docker and Kubernetes

```bash
docker build -t mp064-api:1.0 .
docker run -p 8000:8000 mp064-api:1.0
kubectl apply -f k8s/          # Deployment mp064-api (3 replicas) + Service mp064-svc (80 -> 8000)
```

## Data note

Never open `data/raw/train.csv` in Excel and save it: the marking tool matches the file's hash.
