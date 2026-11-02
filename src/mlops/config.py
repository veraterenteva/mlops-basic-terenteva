"""Pipeline parameters and project paths."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import yaml


class Params(TypedDict):
    """Parameters stored in ``params.yaml`` and tracked by DVC."""

    seed: int
    n_samples: int
    n_features: int
    noise: float
    test_size: float
    ridge_alpha: float
    target_column: str


DEFAULT_PARAMS: Params = {
    "seed": 42,
    "n_samples": 200,
    "n_features": 3,
    "noise": 0.5,
    "test_size": 0.2,
    "ridge_alpha": 1e-6,
    "target_column": "target",
}


@dataclass(frozen=True)
class Paths:
    """File layout of the pipeline relative to its working directory."""

    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data_dir / "raw.csv"

    @property
    def train(self) -> Path:
        return self.data_dir / "train.csv"

    @property
    def test(self) -> Path:
        return self.data_dir / "test.csv"

    @property
    def scaler(self) -> Path:
        return self.data_dir / "scaler.json"

    @property
    def model(self) -> Path:
        return self.data_dir / "model.json"

    @property
    def metrics(self) -> Path:
        return self.data_dir / "metrics.json"

    @property
    def params(self) -> Path:
        return self.root / "params.yaml"


def load_params(path: Path) -> Params:
    """Read ``params.yaml`` and validate that every expected key is present."""
    if not path.exists():
        raise FileNotFoundError(f"params file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("params file must contain a mapping")

    missing = [key for key in DEFAULT_PARAMS if key not in raw]
    if missing:
        raise ValueError(f"missing params: {', '.join(missing)}")

    params: Params = {
        "seed": int(raw["seed"]),
        "n_samples": int(raw["n_samples"]),
        "n_features": int(raw["n_features"]),
        "noise": float(raw["noise"]),
        "test_size": float(raw["test_size"]),
        "ridge_alpha": float(raw["ridge_alpha"]),
        "target_column": str(raw["target_column"]),
    }
    return params
