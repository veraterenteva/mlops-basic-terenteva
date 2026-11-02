"""Block and integration tests for the DVC pipeline stages.

Blocks under test

Each stage in isolation (``run_generate``, ``run_prepare``, ``run_train``,
``run_evaluate``) and the whole chain (``run_all``). The mathematics is
covered by ``test_blocks`` and ``test_model``; here the contract of a stage
is under test: which files it creates, what is written into them and
whether the next stage can consume the result.

What is verified

R1. Each stage creates exactly its own artefacts and loses nothing.
R2. Artefacts are consistent: the two splits add up to the raw dataset,
    the number of weights equals the number of features, and the metrics
    on disk equal the metrics returned by the function.
R3. The full run reaches a high R2, because the data is linear by
    construction and the model must be able to learn it.
R4. The pipeline is reproducible: two consecutive runs give equal metrics.
R5. The ``test_size`` override is validated and actually applied.
"""

import json

import pytest

from mlops.cli import run_all, run_evaluate, run_generate, run_prepare, run_train
from mlops.config import Paths, load_params
from mlops.data import load_csv, load_scaler
from mlops.model import load_model

# Generate stage
# R1: raw.csv exists and holds exactly n_samples rows.


@pytest.mark.integration
def test_generate_stage_creates_raw_file(pipeline_root) -> None:
    paths = Paths(root=pipeline_root)
    assert run_generate(paths).exists()
    assert len(load_csv(paths.raw)) == load_params(paths.params)["n_samples"]


# Prepare stage
# R1 and R2: three artefacts are written, no row is lost and the scaler has
# one statistic per feature.


@pytest.mark.integration
def test_prepare_stage_creates_splits_and_scaler(pipeline_root) -> None:
    paths = Paths(root=pipeline_root)
    run_generate(paths)
    train_path, test_path, scaler_path = run_prepare(paths)

    assert train_path.exists()
    assert test_path.exists()
    assert scaler_path.exists()

    train = load_csv(train_path)
    test = load_csv(test_path)
    raw = load_csv(paths.raw)
    # The invariant that catches an off-by-one in the split logic.
    assert len(train) + len(test) == len(raw)

    scaler = load_scaler(scaler_path)
    assert len(scaler.means) == raw.n_features


# Train stage
# R2: the persisted model must match what the stage returned, otherwise
# evaluate would score a different model than the one just trained.


@pytest.mark.integration
def test_train_stage_creates_model(pipeline_root) -> None:
    paths = Paths(root=pipeline_root)
    run_generate(paths)
    run_prepare(paths)
    weights = run_train(paths)

    assert len(weights) == load_params(paths.params)["n_features"]
    model = load_model(paths.model)
    assert model.weights == pytest.approx(weights)


# Evaluate stage
# R2: the JSON file is the contract for DVC and for a human reader, so its
# content must equal the returned dictionary.


@pytest.mark.integration
def test_evaluate_stage_creates_metrics(pipeline_root) -> None:
    paths = Paths(root=pipeline_root)
    run_generate(paths)
    run_prepare(paths)
    run_train(paths)
    metrics = run_evaluate(paths)

    assert set(metrics) == {"mse", "rmse", "mae", "r2"}
    saved = json.loads(paths.metrics.read_text(encoding="utf-8"))
    assert saved == pytest.approx(metrics)


# Whole chain
# R3 and R4: end-to-end quality and reproducibility.


@pytest.mark.integration
def test_full_pipeline_reaches_high_r2(pipeline_root) -> None:
    paths = Paths(root=pipeline_root)
    metrics = run_all(paths)
    # The data is linear by construction, so R2 must be high. The threshold
    # is deliberately below the observed ~0.99: it is a regression guard,
    # not a quality target.
    assert metrics["r2"] > 0.9
    assert metrics["mse"] >= 0.0
    assert metrics["rmse"] == pytest.approx(metrics["mse"] ** 0.5)


@pytest.mark.integration
def test_pipeline_is_reproducible(pipeline_root) -> None:
    # R4: rerunning with the same params must give the same metrics. This is
    # the foundation of trusting the DVC pipeline.
    paths = Paths(root=pipeline_root)
    first = run_all(paths)
    second = run_all(paths)
    assert first == pytest.approx(second)


@pytest.mark.integration
def test_pipeline_rejects_bad_test_size_override(pipeline_root) -> None:
    # R5: an override is validated as strictly as the value in params.yaml.
    paths = Paths(root=pipeline_root)
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_all(paths, test_size=2.0)


@pytest.mark.integration
def test_pipeline_accepts_test_size_override(pipeline_root) -> None:
    # R5: the override must reach params.yaml and change the actual split.
    paths = Paths(root=pipeline_root)
    run_all(paths, test_size=0.4)
    assert load_params(paths.params)["test_size"] == pytest.approx(0.4)
    test = load_csv(paths.test)
    train = load_csv(paths.train)
    assert len(test) / (len(test) + len(train)) == pytest.approx(0.4, abs=0.05)
