"""Ridge regression implemented from scratch on top of ``blocks``."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mlops.blocks import dot, solve_linear_system


@dataclass
class RidgeRegression:
    """Linear model trained with a closed-form ridge solution.

    ``fit`` stores the weights and the intercept; ``predict`` works on raw
    feature vectors.
    """

    alpha: float = 1e-6
    weights: list[float] | None = None
    intercept: float = 0.0

    def fit(
        self, features: Sequence[Sequence[float]], targets: Sequence[float]
    ) -> "RidgeRegression":
        """Fit the model with ``(X^T X + alpha * I) w = X^T y``."""
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not features:
            raise ValueError("features must not be empty")
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same length")

        n_features = len(features[0])
        if n_features == 0:
            raise ValueError("feature vectors must not be empty")
        if any(len(row) != n_features for row in features):
            raise ValueError("all feature vectors must have the same length")

        design = [[1.0, *[float(value) for value in row]] for row in features]
        target_values = [float(value) for value in targets]
        size = n_features + 1

        gram = [[0.0] * size for _ in range(size)]
        moment = [0.0] * size

        for row, target in zip(design, target_values, strict=True):
            for i in range(size):
                moment[i] += row[i] * target
                for j in range(size):
                    gram[i][j] += row[i] * row[j]

        # Do not penalise the intercept term.
        for index in range(1, size):
            gram[index][index] += self.alpha

        solution = solve_linear_system(gram, moment)
        self.intercept = solution[0]
        self.weights = solution[1:]
        return self

    def predict_one(self, features: Sequence[float]) -> float:
        """Predict a single row."""
        if self.weights is None:
            raise ValueError("model is not fitted")
        if len(features) != len(self.weights):
            raise ValueError("feature count does not match the model")
        return self.intercept + dot(self.weights, [float(value) for value in features])

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        """Predict a batch of rows."""
        return [self.predict_one(row) for row in features]

    def to_dict(self) -> dict[str, object]:
        """Serialise the model to a JSON-compatible mapping."""
        if self.weights is None:
            raise ValueError("model is not fitted")
        return {
            "alpha": self.alpha,
            "intercept": self.intercept,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RidgeRegression":
        """Restore a model from its JSON representation."""
        weights = payload.get("weights")
        if not isinstance(weights, list):
            raise ValueError("weights must be a list")
        model = cls(alpha=float(str(payload["alpha"])))
        model.intercept = float(str(payload["intercept"]))
        model.weights = [float(str(value)) for value in weights]
        return model


def save_model(path: Path, model: RidgeRegression) -> None:
    """Persist a trained model as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


def load_model(path: Path) -> RidgeRegression:
    """Load a model from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RidgeRegression.from_dict(payload)
