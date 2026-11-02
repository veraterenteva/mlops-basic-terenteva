"""Unit tests for regression metrics.

Block under test: ``mlops.metrics``

The metrics block turns predictions into the numbers that land in
``data/metrics.json``. They are what a reviewer looks at to decide whether
the model is good enough, so a wrong formula here would silently misreport
quality instead of failing.

What is verified

R1. Every formula matches its textbook definition: MSE, RMSE, MAE and R2.
R2. The metrics are mutually consistent: RMSE is exactly the square root
    of MSE.
R3. Boundary cases behave predictably: a perfect prediction, a constant
    target, a single element, negative values.
R4. Invalid input is rejected the same way by all four metrics: an empty
    sequence and mismatched lengths raise ``ValueError``.
"""

import pytest

from mlops.metrics import all_metrics, mae, mse, r2, rmse

# MSE and MAE
# Covers R1 and R3: MSE = mean((actual - predicted)^2) and
# MAE = mean(|actual - predicted|).


@pytest.mark.unit
@pytest.mark.parametrize(
    ("actual", "predicted", "expected_mse", "expected_mae"),
    [
        # A perfect prediction must give exactly zero for both metrics.
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0.0, 0.0),
        # Predicting the mean: every residual is 1/1/1, so MSE == MAE.
        ([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], 2.0 / 3.0, 2.0 / 3.0),
        # Squaring punishes large errors: 3^2 = 9 versus |3| = 3.
        ([0.0, 0.0], [3.0, -3.0], 9.0, 3.0),
        # Boundary length: a single element.
        ([10.0], [12.0], 4.0, 2.0),
    ],
)
def test_mse_and_mae(
    actual: list[float],
    predicted: list[float],
    expected_mse: float,
    expected_mae: float,
) -> None:
    assert mse(actual, predicted) == pytest.approx(expected_mse)
    assert mae(actual, predicted) == pytest.approx(expected_mae)


# RMSE
# R2: RMSE is not an independent formula but the square root of MSE. The
# test freezes that relationship so the two cannot drift apart.


@pytest.mark.unit
def test_rmse_is_square_root_of_mse() -> None:
    actual = [1.0, 2.0, 3.0]
    predicted = [1.5, 1.5, 3.5]
    assert rmse(actual, predicted) == pytest.approx(mse(actual, predicted) ** 0.5)


# R2 score
# R1 and R3: R2 = 1 - RSS / TSS. Project convention: when the targets have
# zero variance the score is undefined, so 0.0 is returned instead of
# dividing by zero.


@pytest.mark.unit
@pytest.mark.parametrize(
    ("actual", "predicted", "expected"),
    [
        # A perfect prediction leaves no residual, so R2 is exactly 1.
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0),
        # RSS = 0.5, TSS = 2.0, therefore 1 - 0.5 / 2.0 = 0.75.
        ([1.0, 2.0, 3.0], [1.5, 2.0, 2.5], 0.75),
        # Predicting the mean makes RSS equal TSS, so R2 is exactly zero.
        ([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], 0.0),
        # Zero variance in the targets: undefined, reported as zero by
        # the convention documented in the docstring.
        ([2.0, 2.0, 2.0], [1.0, 2.0, 3.0], 0.0),
    ],
)
def test_r2(actual: list[float], predicted: list[float], expected: float) -> None:
    assert r2(actual, predicted) == pytest.approx(expected)


# Aggregate metrics
# R1: one entry point that collects every metric. The set of keys is a
# contract, because the evaluate stage writes it to JSON and DVC reads it.


@pytest.mark.unit
def test_all_metrics_returns_every_key() -> None:
    metrics = all_metrics([1.0, 2.0], [1.0, 3.0])
    assert set(metrics) == {"mse", "rmse", "mae", "r2"}
    # Residuals are 0 and -1, so MSE = (0 + 1) / 2 = 0.5.
    assert metrics["mse"] == pytest.approx(0.5)


# Input validation
# R4: all four metrics share the same guard, so a mistake in the shapes
# cannot slip through for just one of them.


@pytest.mark.unit
@pytest.mark.parametrize(
    ("actual", "predicted", "message"),
    [
        # Nothing to compare.
        ([], [], "must not be empty"),
        # Two actual values but only one prediction.
        ([1.0, 2.0], [1.0], "same length"),
    ],
)
def test_metrics_validate_input(
    actual: list[float], predicted: list[float], message: str
) -> None:
    for metric in (mse, rmse, mae, r2):
        with pytest.raises(ValueError, match=message):
            metric(actual, predicted)
