# 🌸 Iris Classifier — Production ML Pipeline

A production-grade machine learning pipeline showcasing professional MLOps practices:
end-to-end training, experiment tracking, model versioning, REST inference, and containerisation.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Training pipeline** | `GridSearchCV` over `RandomForestClassifier`, triggered via CLI or cron |
| **Experiment tracking** | MLflow — parameters, metrics, artifacts, and model registry |
| **Model versioning** | MLflow Model Registry with `IrisClassifier` registered model |
| **REST inference** | FastAPI with single + batch prediction, Pydantic validation, auto-docs |
| **Containerisation** | Multi-stage Docker builds + Docker Compose orchestration |
| **CI/CD** | GitHub Actions — lint → test → training smoke test → Docker build |
| **Modular design** | Config, data, training, evaluation, inference, API fully decoupled |

---

## 🗂 Project Structure

```
ml-pipeline/
│
├── src/                          # Core Python package
│   ├── config.py                 # ← Single source of truth for all config
│   ├── data/
│   │   └── loader.py             # Dataset loading + train/test split
│   ├── training/
│   │   ├── pipeline.py           # sklearn Pipeline construction
│   │   └── trainer.py            # GridSearchCV + MLflow logging
│   ├── evaluation/
│   │   └── metrics.py            # Accuracy, F1, confusion matrix
│   └── inference/
│       └── predictor.py          # Model loading + prediction interface
│
├── api/                          # FastAPI application
│   ├── main.py                   # App factory, health endpoint
│   ├── schemas.py                # Pydantic request/response models
│   └── routers/
│       └── predict.py            # /predict and /predict/batch routes
│
├── scripts/
│   ├── train.py                  # CLI entry point (argparse)
│   └── run_pipeline.sh           # Bash wrapper for cron scheduling
│
├── tests/
│   ├── test_data_loader.py       # Data loading unit tests
│   ├── test_metrics.py           # Metrics unit tests
│   ├── test_predictor.py         # Inference unit tests (mocked artifacts)
│   └── test_api.py               # FastAPI integration tests
│
├── docker/
│   ├── Dockerfile.train          # Training container (with cron support)
│   └── Dockerfile.api            # Multi-stage FastAPI container
│
├── docker-compose.yml            # Trainer + API + MLflow UI
├── Makefile                      # Developer convenience commands
├── crontab.txt                   # Cron schedule for recurring training
├── requirements.txt
├── pyproject.toml                # pytest + ruff + mypy config
└── .github/workflows/ci.yml      # GitHub Actions CI pipeline
```

---

## 🚀 Quick Start

### Option A — Local (no Docker)

```bash
# 1. Clone and enter the repo
git clone https://github.com/aniket-1177/Iris-Classifier-ML-Pipeline.git
cd ml-pipeline

# 2. Create virtual environment and install dependencies
make install
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Copy environment config
cp .env.example .env

# 4. Train the model
make train
# Equivalent to: python scripts/train.py

# 5. Start the inference API
make serve
# → API docs: http://localhost:8000/docs

# 6. Open MLflow UI (in a separate terminal)
make mlflow
# → MLflow UI: http://localhost:5000
```

### Option B — Docker Compose (recommended)

```bash
# Start all services (MLflow UI + training + API)
make docker-up

# Or manually:
docker compose up --build
```

| Service | URL |
|---|---|
| FastAPI docs | http://localhost:8000/docs |
| FastAPI health | http://localhost:8000/health |
| MLflow UI | http://localhost:5000 |

> **Note:** The `trainer` container runs once, saves the model to a shared Docker volume, then exits. The `api` container reads from that same volume — no image rebuild needed when you retrain.

---

## 📡 API Reference

### `POST /predict/`

Single-sample prediction.

**Request body:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "predicted_class": "setosa",
  "confidence": 0.98,
  "class_probabilities": {
    "setosa": 0.98,
    "versicolor": 0.01,
    "virginica": 0.01
  }
}
```

### `POST /predict/batch`

Send up to 100 samples in a single request.

```json
{
  "samples": [
    { "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2 },
    { "sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5 }
  ]
}
```

### `GET /health`

Returns model load status and version.

---

## 🧪 Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov=api --cov-report=term-missing

# Run a specific test file
pytest tests/test_api.py -v

# Run only unit tests (fast, no artifacts needed)
pytest tests/test_data_loader.py tests/test_metrics.py -v
```

---

## ⏱ Scheduled Retraining (Cron)

The pipeline can be triggered on a schedule using the system cron daemon.

```bash
# Install the provided crontab (runs every Monday at 02:00 AM)
crontab crontab.txt

# Verify it's installed
crontab -l
```

Or trigger manually at any time:
```bash
bash scripts/run_pipeline.sh
```

Logs are saved to `logs/train_<timestamp>.log`. Training results (metrics + run ID) are written to `logs/results_<timestamp>.json`.

To run cron **inside Docker**, change the trainer's `CMD` in `docker-compose.yml`:
```yaml
# docker-compose.yml
trainer:
  command: cron -f   # runs the cron daemon in foreground
```

---

## 🧪 MLflow Experiment Tracking

Every training run automatically logs:

- **Parameters** — CV folds, test size, best hyperparameters from grid search
- **Metrics** — CV best score, test accuracy, macro F1, precision, recall, per-class F1
- **Artifacts** — trained pipeline `.pkl`, label encoder `.pkl`
- **Registered model** — `IrisClassifier` in the MLflow Model Registry

View the UI at `http://localhost:5000` (local) or `http://localhost:5000` (Docker).

---

## ⚙️ Configuration

All settings are controlled via environment variables (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `file://./mlruns` | MLflow backend store |
| `MLFLOW_EXPERIMENT_NAME` | `iris-classifier` | Experiment name |
| `API_HOST` | `0.0.0.0` | FastAPI host |
| `API_PORT` | `8000` | FastAPI port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 🛠 Make Targets

```
make install       Install dependencies into .venv
make train         Run training pipeline locally
make serve         Start FastAPI dev server with --reload
make mlflow        Open MLflow tracking UI
make docker-build  Build all Docker images
make docker-up     docker compose up --build
make docker-down   docker compose down
make docker-train  Retrain model inside Docker
make lint          Run ruff linter
make clean         Remove __pycache__ and build artifacts
make clean-all     Full clean including models and logs
```

---

## 🏛 Architecture Decisions

**Why `lru_cache` on `get_predictor()`?**
The model is loaded from disk once at startup and cached in memory. All subsequent API requests reuse the same `Predictor` instance — avoiding repeated I/O on every call.

**Why a shared Docker volume for models?**
Decouples retraining from deployment. The trainer writes `models/` to a named volume; the API reads from it. Retraining doesn't require rebuilding or redeploying the API image.

**Why `GridSearchCV` with a sklearn `Pipeline`?**
Wrapping the scaler inside the pipeline prevents data leakage during cross-validation — the scaler is fit only on each training fold, not the full dataset.

**Why a separate `Predictor` class?**
Keeps inference logic independent of FastAPI. The class can be imported and tested without any web framework, and could be swapped in a Celery worker or batch job without changes.

---

## 📦 Tech Stack

- **Scikit-learn** — ML pipeline and model training
- **MLflow** — experiment tracking and model registry
- **FastAPI** — async REST inference server
- **Pydantic v2** — request/response validation
- **Uvicorn** — ASGI server
- **Docker + Compose** — containerisation
- **GitHub Actions** — CI/CD
- **pytest** — unit + integration tests
- **Ruff** — linting

---

## 📈 Extending This Project

Some ideas for taking this further:

- **Add a new model** — swap `RandomForestClassifier` for XGBoost or LightGBM in `pipeline.py`
- **Real dataset** — replace the Iris loader with a CSV/database ingestion step in `loader.py`
- **Remote MLflow** — point `MLFLOW_TRACKING_URI` at a hosted MLflow server (e.g., on AWS or Databricks)
- **Model promotion** — add a script that promotes the best run to "Production" in the MLflow registry
- **Alerting** — send a Slack/email notification from `run_pipeline.sh` on failure
- **Async inference** — add a Celery + Redis worker for heavy batch jobs
