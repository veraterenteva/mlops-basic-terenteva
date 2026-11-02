"""End-to-end tests: the whole pipeline through the command line interface.

System under test: ``mlops.cli``

The entry point that DVC invokes for every stage. These tests run it the
way a user would: with flags, inside a temporary directory, and then check
the result on disk. This is the only level that catches wiring mistakes:
wrong paths, a forgotten stage call, or a mismatch between the CLI
arguments and ``dvc.yaml``.

What is verified

R1. The ``all`` command runs every stage and creates all six artefacts.
R2. ``metrics.json`` is valid: the expected keys and consistent values.
R3. Individual stages can be run one by one in dependency order.
R4. ``--test-size`` really changes the configuration.
R5. The CLI without a command exits through ``SystemExit`` (argparse
    behaviour, no silent no-op).
"""

import json
from pathlib import Path

import pytest

from mlops.cli import main

# Full run
# R1: a single command takes the system from an empty directory to ready
# metrics.


@pytest.mark.e2e
def test_cli_all_produces_every_artifact(pipeline_root: Path) -> None:
    exit_code = main(["--root", str(pipeline_root), "all"])

    assert exit_code == 0
    assert (pipeline_root / "data" / "raw.csv").exists()
    assert (pipeline_root / "data" / "train.csv").exists()
    assert (pipeline_root / "data" / "test.csv").exists()
    assert (pipeline_root / "data" / "scaler.json").exists()
    assert (pipeline_root / "data" / "model.json").exists()
    assert (pipeline_root / "data" / "metrics.json").exists()


@pytest.mark.e2e
def test_cli_all_writes_valid_metrics(pipeline_root: Path) -> None:
    main(["--root", str(pipeline_root), "all"])

    payload = json.loads(
        (pipeline_root / "data" / "metrics.json").read_text(encoding="utf-8")
    )
    # R2: the metric keys are a contract, read by DVC and by a human.
    assert set(payload) == {"mse", "rmse", "mae", "r2"}
    # The data is linear, so a correct run cannot produce a low R2.
    assert payload["r2"] > 0.9
    # RMSE must still be the square root of MSE, even when read from a file.
    assert payload["rmse"] == pytest.approx(payload["mse"] ** 0.5)


# Stage by stage
# R3: the stages also work on their own, in dependency order, which is
# exactly how DVC calls them.


@pytest.mark.e2e
def test_cli_runs_stages_separately(pipeline_root: Path) -> None:
    for stage in ("generate", "prepare", "train", "evaluate"):
        assert main(["--root", str(pipeline_root), stage]) == 0

    assert (pipeline_root / "data" / "metrics.json").exists()


# Parameter override
# R4: the CLI flag must reach params.yaml.


@pytest.mark.e2e
def test_cli_all_accepts_test_size_override(pipeline_root: Path) -> None:
    assert main(["--root", str(pipeline_root), "all", "--test-size", "0.3"]) == 0
    params_text = (pipeline_root / "params.yaml").read_text(encoding="utf-8")
    assert "test_size: 0.3" in params_text


# CLI contract
# R5: without the required subcommand argparse terminates the process
# rather than doing nothing.


@pytest.mark.e2e
def test_cli_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        main([])
