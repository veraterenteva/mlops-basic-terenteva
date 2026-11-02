"""Tests for the generate stage: synthetic dataset creation.

Block under test: ``mlops.generate``

The first stage of the pipeline. It writes ``data/raw.csv`` from a known
linear law ``y = 2.5*f0 - 1.5*f1 + 0.75*f2 + noise``. Because the
coefficients are known, model quality can later be checked against the
truth and not only against aggregate metrics.

What is verified

R1. A row follows the linear law: features are copied as-is and the target
    is the weighted sum plus noise.
R2. The same seed produces the same dataset, which DVC relies on to decide
    whether a stage is stale.
R3. The number of rows and features matches the parameters.
R4. Invalid parameters are rejected: zero rows, negative noise and an
    unsupported feature count.
R5. The CSV has the header ``f0..fn, target`` and nested directories are
    created on demand.
"""

import csv
import random

import pytest

from mlops.config import DEFAULT_PARAMS, Params
from mlops.generate import (
    TRUE_WEIGHTS,
    build_row,
    generate_dataset,
    generate_rows,
    write_csv,
)

# Row construction
# R1: features are preserved and the target is computed from them.


@pytest.mark.unit
def test_build_row_appends_target() -> None:
    row = build_row([1.0, 2.0, 3.0], TRUE_WEIGHTS, noise=0.0, rng=random.Random(0))
    expected = sum(w * v for w, v in zip(TRUE_WEIGHTS, [1.0, 2.0, 3.0], strict=True))
    # With zero noise the target must equal the weighted sum exactly.
    assert row[:3] == [1.0, 2.0, 3.0]
    assert row[3] == pytest.approx(expected)


# Dataset generation
# R2 and R3: reproducibility and parameter compliance.


@pytest.mark.unit
def test_generate_rows_is_deterministic(params: Params) -> None:
    # R2: the same seed must give byte-identical rows, otherwise DVC would
    # treat the stage as changed on every run.
    assert generate_rows(params) == generate_rows(params)


@pytest.mark.unit
def test_generate_rows_respects_sample_count(params: Params) -> None:
    # R3: n_samples rows, each of length n_features + 1 for the target.
    rows = generate_rows(params)
    assert len(rows) == params["n_samples"]
    assert all(len(row) == params["n_features"] + 1 for row in rows)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("n_samples", "message"),
    [
        # An empty dataset is meaningless.
        (0, "must be positive"),
        # A negative row count is nonsensical.
        (-5, "must be positive"),
    ],
)
def test_generate_rows_rejects_bad_sample_count(
    params: Params, n_samples: int, message: str
) -> None:
    # R4: both boundary values must fail with the same clear message.
    broken: Params = {**params, "n_samples": n_samples}
    with pytest.raises(ValueError, match=message):
        generate_rows(broken)


@pytest.mark.unit
def test_generate_rows_rejects_negative_noise(params: Params) -> None:
    # R4: a negative sigma has no meaning for a Gaussian distribution.
    broken: Params = {**params, "noise": -1.0}
    with pytest.raises(ValueError, match="non-negative"):
        generate_rows(broken)


@pytest.mark.unit
def test_generate_rows_rejects_unknown_feature_count(params: Params) -> None:
    # R4: there are only three true weights, so a different feature count
    # cannot be mapped onto them.
    broken: Params = {**params, "n_features": 5}
    with pytest.raises(ValueError, match="n_features must be"):
        generate_rows(broken)


# CSV writing
# R5: correct header, automatic directory creation, refusal on empty input.


@pytest.mark.unit
def test_write_csv_creates_header(tmp_path) -> None:
    # A path inside a nested directory proves that parents are created.
    path = tmp_path / "nested" / "out.csv"
    write_csv(path, [[1.0, 2.0, 3.0]], n_features=2)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["f0", "f1", "target"]
    assert rows[1] == ["1.0", "2.0", "3.0"]


@pytest.mark.unit
def test_write_csv_rejects_empty_rows(tmp_path) -> None:
    # R5: do not create a file that looks valid but holds no data.
    with pytest.raises(ValueError, match="must not be empty"):
        write_csv(tmp_path / "out.csv", [], n_features=1)


# Whole stage
# R3 and R5 together: a real file on disk with the expected number of lines.


@pytest.mark.integration
def test_generate_dataset_writes_file(tmp_path) -> None:
    path = generate_dataset(DEFAULT_PARAMS, tmp_path / "raw.csv")
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    # One header line plus one line per sample.
    assert len(lines) == DEFAULT_PARAMS["n_samples"] + 1
