"""Shared fixtures for the pipeline tests.

How the fixtures fit together

They form a dependency chain:

    params -> raw_csv -> dataset -> clean_dataset -> splits -> trained_model

A test asks only for the link it needs, so the data preparation is written
once and every test still starts from a well-defined state.

Two rules apply to the whole suite:

1. Everything runs inside ``tmp_path``. No test reads the working
   directory, writes outside it, or depends on leftovers from a previous
   run.
2. No network access and no dependence on the current date.

``params`` deliberately shrinks ``n_samples`` to 40 and sets ``noise`` to
0.0, which keeps the 100+ tests fast and makes accuracy assertions exact.
"""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mlops.config import DEFAULT_PARAMS, Params
from mlops.data import (
    Dataset,
    Splits,
    StandardScaler,
    clean,
    load_csv,
    train_test_split,
)
from mlops.generate import generate_dataset
from mlops.model import RidgeRegression, save_model

# Parameters
# A small but valid parameter set. ``noise=0.0`` removes randomness from the
# target, so tests that check model accuracy do not need loose tolerances.


@pytest.fixture
def params(tmp_path: Path) -> Params:
    """Pipeline parameters with fewer rows and no noise."""
    return {
        **DEFAULT_PARAMS,
        "n_samples": 40,
        "noise": 0.0,
    }


@pytest.fixture
def params_file(tmp_path: Path, params: Params) -> Path:
    """``params.yaml`` on disk, for tests that read the configuration."""
    path = tmp_path / "params.yaml"
    lines = [f"{key}: {value}" for key, value in params.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# Data
# The chain that turns a raw CSV into a trained model. Splitting it into
# separate fixtures lets each block be tested at its own level without
# duplicating the preparation code.


@pytest.fixture
def raw_csv(tmp_path: Path, params: Params) -> Path:
    """A real CSV written by the real generate stage.

    This is not a mock: the tests consume exactly the format that the
    pipeline produces.
    """
    return generate_dataset(params, tmp_path / "data" / "raw.csv")


@pytest.fixture
def dataset(raw_csv: Path) -> Dataset:
    """The generated CSV loaded into a ``Dataset``."""
    return load_csv(raw_csv)


@pytest.fixture
def clean_dataset(dataset: Dataset) -> Dataset:
    """Cleaned dataset, the input for splitting and scaling."""
    return clean(dataset)


@pytest.fixture
def splits(clean_dataset: Dataset) -> Splits:
    """A train/test split with a fixed seed, therefore reproducible."""
    return train_test_split(clean_dataset, test_size=0.25, seed=7)


@pytest.fixture
def scaled_splits(splits: Splits) -> tuple[Splits, StandardScaler]:
    """Scaled splits together with the scaler that produced them.

    The scaler is returned separately because some tests assert on its
    statistics while others only need the transformed data.
    """
    scaler = StandardScaler.fit(splits.train)
    scaled = Splits(
        train=scaler.transform(splits.train),
        test=scaler.transform(splits.test),
    )
    return scaled, scaler


# Model
# A trained model is expensive relative to the rest of a test, so it is
# built once and shared through fixtures instead of being refitted.


@pytest.fixture
def trained_model(splits: Splits) -> RidgeRegression:
    """A model fitted on the train split.

    ``alpha`` is almost zero here: these tests probe the mathematics of the
    closed-form solution. Regularisation has its own dedicated test.
    """
    model = RidgeRegression(alpha=1e-9)
    return model.fit(splits.train.features, splits.train.targets)


@pytest.fixture
def model_file(tmp_path: Path, trained_model: RidgeRegression) -> Path:
    """The trained model persisted as JSON."""
    path = tmp_path / "model.json"
    save_model(path, trained_model)
    return path


# Pipeline working directory
# Integration and end-to-end tests drive the stages and the CLI, so they
# need a ready directory that already contains ``params.yaml``.


@pytest.fixture
def pipeline_root(tmp_path: Path, params: Params) -> Path:
    """A directory containing ``params.yaml`` and nothing else.

    This path is passed to ``--root``, so every artefact is created inside
    ``tmp_path`` and the repository is never polluted.
    """
    root = tmp_path / "pipeline"
    root.mkdir()
    lines = [f"{key}: {value}" for key, value in params.items()]
    (root / "params.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


# Data factory for negative tests
# Negative tests need malformed CSV files: no target column, a short row, a
# header-only file. A factory fixture builds each variant on the spot
# instead of shipping one static file per case.


@pytest.fixture
def csv_writer() -> Callable[[Path, list[str], list[list[Any]]], Path]:
    """Factory that writes a CSV with an arbitrary header and rows.

    A function is returned rather than a path because each negative test
    needs a different malformed file.
    """

    def write(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
        """Write ``rows`` under ``header`` and return the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    return write
