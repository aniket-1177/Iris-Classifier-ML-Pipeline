#!/usr/bin/env python3
"""
Training entry-point script.

Run manually:
    python scripts/train.py

Or via Makefile:
    make train

Or scheduled via cron (see crontab.txt):
    0 2 * * 1  cd /app && python scripts/train.py >> /var/log/ml_train.log 2>&1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Make sure `src` is importable when script is run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LOG_FORMAT, LOG_LEVEL, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from src.training.trainer import run_training

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Iris Classifier ML pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=MLFLOW_EXPERIMENT_NAME,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=MLFLOW_TRACKING_URI,
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write training results as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Override config with CLI args if provided
    import src.config as cfg
    import os
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment
    os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri

    logger.info("=" * 55)
    logger.info("  ML Pipeline Training Run")
    logger.info("  Experiment : %s", args.experiment)
    logger.info("  Tracking   : %s", args.tracking_uri)
    logger.info("=" * 55)

    try:
        results = run_training()
    except Exception:
        logger.exception("Training failed with an unhandled exception.")
        sys.exit(1)

    logger.info("=" * 55)
    logger.info("Training finished successfully.")
    logger.info("  Run ID     : %s", results["run_id"])
    logger.info("  Accuracy   : %.4f", results["metrics"]["accuracy"])
    logger.info("  Macro F1   : %.4f", results["metrics"]["macro_f1"])
    logger.info("  Model path : %s", results["model_path"])
    logger.info("=" * 55)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results written to %s", out)


if __name__ == "__main__":
    main()
