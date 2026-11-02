"""Synthetic dataset generation.

The generator writes a deterministic CSV file: the same ``seed`` always
produces the same rows, which makes DVC stages reproducible.
"""

import csv
import random
from pathlib import Path

from mlops.config import Params

# The ground-truth coefficients are fixed so that a trained model can be
# compared against a known answer in tests.
TRUE_WEIGHTS = (2.5, -1.5, 0.75)


def build_row(
    features: list[float],
    weights: tuple[float, ...],
    noise: float,
    rng: random.Random,
) -> list[float]:
    """Return a feature vector with the target appended."""
    signal = sum(
        weight * value for weight, value in zip(weights, features, strict=True)
    )
    target = signal + rng.gauss(0.0, noise)
    return [*features, target]


def generate_rows(params: Params) -> list[list[float]]:
    """Generate ``n_samples`` rows of ``n_features`` features plus a target."""
    n_features = params["n_features"]
    if n_features != len(TRUE_WEIGHTS):
        raise ValueError(f"n_features must be {len(TRUE_WEIGHTS)}")
    if params["n_samples"] <= 0:
        raise ValueError("n_samples must be positive")
    if params["noise"] < 0:
        raise ValueError("noise must be non-negative")

    rng = random.Random(params["seed"])
    rows: list[list[float]] = []
    for _ in range(params["n_samples"]):
        features = [rng.uniform(-5.0, 5.0) for _ in range(n_features)]
        rows.append(build_row(features, TRUE_WEIGHTS, params["noise"], rng))
    return rows


def write_csv(path: Path, rows: list[list[float]], n_features: int) -> None:
    """Write rows to a CSV file with a header."""
    if not rows:
        raise ValueError("rows must not be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = [f"f{index}" for index in range(n_features)] + ["target"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def generate_dataset(params: Params, output: Path) -> Path:
    """Generate the raw dataset and return its path."""
    rows = generate_rows(params)
    write_csv(output, rows, params["n_features"])
    return output
