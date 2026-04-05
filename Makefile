# ─────────────────────────────────────────────────────────────────────────────
# Makefile  —  Developer convenience commands
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install train serve mlflow docker-build docker-up docker-down lint clean

# Default target
help:
	@echo ""
	@echo "  Iris Classifier ML Pipeline — Make Targets"
	@echo "  ─────────────────────────────────────────────"
	@echo "  install       Install dependencies into venv"
	@echo "  train         Run training pipeline (local)"
	@echo "  serve         Start FastAPI inference server (local)"
	@echo "  mlflow        Open MLflow UI in browser"
	@echo "  docker-build  Build all Docker images"
	@echo "  docker-up     Start all services via Docker Compose"
	@echo "  docker-down   Stop all Docker Compose services"
	@echo "  lint          Run code quality checks"
	@echo "  clean         Remove build artifacts"
	@echo ""

# ── Local development ─────────────────────────────────────────────────────────

install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "✅ Dependencies installed. Activate with: source .venv/bin/activate"

train:
	python scripts/train.py

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	mlflow ui --backend-store-uri mlruns --port 5000

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down --remove-orphans

docker-train:
	docker compose run --rm trainer

# ── Quality ───────────────────────────────────────────────────────────────────

lint:
	ruff check src/ api/ scripts/
	mypy src/ api/

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	@echo "✅ Cleaned build artifacts."

clean-models:
	rm -f models/*.pkl
	@echo "✅ Removed saved models."

clean-all: clean clean-models
	rm -rf mlruns/ logs/
	@echo "✅ Full clean complete."
