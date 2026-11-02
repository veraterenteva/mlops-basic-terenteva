"""Evaluation stage: score the trained model on the test split."""

import json
from pathlib import Path

from mlops.data import load_csv
from mlops.metrics import all_metrics
from mlops.model import load_model


def evaluate_model(
    test_path: Path, model_path: Path, metrics_path: Path
) -> dict[str, float]:
    """Score the model on the test split and write metrics to JSON."""
    dataset = load_csv(test_path)
    model = load_model(model_path)
    predictions = model.predict(dataset.features)
    metrics = all_metrics(dataset.targets, predictions)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
