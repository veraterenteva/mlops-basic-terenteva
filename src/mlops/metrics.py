"""Regression metrics implemented without external libraries."""

from collections.abc import Sequence


def _validate(actual: Sequence[float], predicted: Sequence[float]) -> None:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        raise ValueError("inputs must not be empty")


def mse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean squared error."""
    _validate(actual, predicted)
    return sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True)) / len(
        actual
    )


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Root mean squared error."""
    return mse(actual, predicted) ** 0.5


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error."""
    _validate(actual, predicted)
    return sum(abs(a - p) for a, p in zip(actual, predicted, strict=True)) / len(actual)


def r2(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Coefficient of determination.

    Returns ``0.0`` when the targets have no variance, because the score is
    undefined in that case.
    """
    _validate(actual, predicted)
    mean_actual = sum(actual) / len(actual)
    total = sum((a - mean_actual) ** 2 for a in actual)
    if total == 0.0:
        return 0.0
    residual = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True))
    return 1.0 - residual / total


def all_metrics(
    actual: Sequence[float], predicted: Sequence[float]
) -> dict[str, float]:
    """Compute every supported metric at once."""
    return {
        "mse": mse(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "r2": r2(actual, predicted),
    }
