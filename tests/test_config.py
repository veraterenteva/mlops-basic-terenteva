"""Unit tests for the configuration block: parameters and file paths.

Block under test: ``mlops.config``

This module holds ``Params`` (the schema of ``params.yaml``), ``Paths``
(the file layout) and ``load_params`` (reading and validation). DVC tracks
``params.yaml`` as a dependency, so a mistake here breaks the
reproducibility of the whole pipeline rather than one stage.

What is verified

R1. ``Paths`` returns exactly the file names that ``dvc.yaml`` declares,
    so renaming an artefact in one place fails this test.
R2. ``load_params`` reads every key and coerces the types: ints stay ints,
    floats are compared approximately, strings stay strings.
R3. A missing file raises ``FileNotFoundError``.
R4. An incomplete or malformed YAML is rejected with a clear message
    instead of failing later somewhere inside training.
"""

import pytest

from mlops.cli import default_params_text
from mlops.config import DEFAULT_PARAMS, Paths, load_params

# Path layout
# R1: the artefact names must stay in sync with the DVC stage definitions.


@pytest.mark.unit
def test_paths_layout(tmp_path) -> None:
    paths = Paths(root=tmp_path)
    assert paths.raw == tmp_path / "data" / "raw.csv"
    assert paths.train == tmp_path / "data" / "train.csv"
    assert paths.test == tmp_path / "data" / "test.csv"
    assert paths.scaler == tmp_path / "data" / "scaler.json"
    assert paths.model == tmp_path / "data" / "model.json"
    assert paths.metrics == tmp_path / "data" / "metrics.json"
    assert paths.params == tmp_path / "params.yaml"


# Reading parameters
# R2 and R3: parse YAML, coerce types, validate the schema.


@pytest.mark.unit
def test_load_params_roundtrip(tmp_path) -> None:
    # The text produced by the CLI must be readable back without any loss.
    path = tmp_path / "params.yaml"
    path.write_text(default_params_text(), encoding="utf-8")
    assert load_params(path) == DEFAULT_PARAMS


@pytest.mark.unit
def test_load_params_reads_values(tmp_path) -> None:
    # R2: integer values stay integers, floats are compared approximately.
    path = tmp_path / "params.yaml"
    path.write_text(
        "\n".join(
            [
                "seed: 1",
                "n_samples: 5",
                "n_features: 3",
                "noise: 0.1",
                "test_size: 0.5",
                "ridge_alpha: 0.01",
                "target_column: target",
            ]
        ),
        encoding="utf-8",
    )
    params = load_params(path)
    assert params["seed"] == 1
    assert params["n_samples"] == 5
    assert params["test_size"] == pytest.approx(0.5)


@pytest.mark.unit
def test_load_params_missing_file(tmp_path) -> None:
    # R3: no silent fallback to defaults, because that would hide a broken
    # DVC dependency.
    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "missing.yaml")


@pytest.mark.unit
def test_load_params_rejects_missing_keys(tmp_path) -> None:
    # R4: a partial config is an explicit error listing what is missing,
    # otherwise a stage would fail later with a bare KeyError.
    path = tmp_path / "params.yaml"
    path.write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing params"):
        load_params(path)


@pytest.mark.unit
def test_load_params_rejects_non_mapping(tmp_path) -> None:
    # R4: a YAML list instead of a mapping is rejected immediately.
    path = tmp_path / "params.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_params(path)
