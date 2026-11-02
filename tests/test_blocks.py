"""Unit tests for the numeric primitives of the pipeline.

Block under test: ``mlops.blocks``

These are the smallest mathematical units in the project: dot product,
mean, variance, standard deviation and a linear system solver. The ridge
regression model is built directly on top of them, so a silent error here
would corrupt training without raising anything anywhere else.

What is verified

R1. Formulas match hand-computed reference values.
R2. Edge lengths are handled: a single element, a zero vector, negative
    components.
R3. Invalid input fails loudly with a descriptive error: an empty
    sequence, vectors of different length, incompatible matrix shapes.
R4. The solver is numerically robust: it pivots when the leading element
    is zero, and it rejects a singular matrix instead of returning inf/nan.
"""

import pytest

from mlops.blocks import dot, mean, solve_linear_system, standard_deviation, variance

# Dot product
# Covers R1 and R2: the dot product of two equal-length vectors, including
# a zero vector, negative components and a one-element vector.


@pytest.mark.unit
@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        # Ordinary case: 1 * 3 + 2 * 4 = 11.
        ([1.0, 2.0], [3.0, 4.0], 11.0),
        # A zero vector must produce exactly zero, not a rounding artefact.
        ([0.0, 0.0], [5.0, 5.0], 0.0),
        # Negative components: (-1) * (-1) + 1 * 1 = 2.
        ([-1.0, 1.0], [-1.0, 1.0], 2.0),
        # Boundary length: a single element.
        ([2.0], [0.5], 1.0),
    ],
)
def test_dot(left: list[float], right: list[float], expected: float) -> None:
    assert dot(left, right) == pytest.approx(expected)


@pytest.mark.unit
def test_dot_rejects_different_lengths() -> None:
    # R3: mismatched lengths must raise. Calling zip() without a guard would
    # silently truncate and return a wrong number instead.
    with pytest.raises(ValueError):
        dot([1.0, 2.0], [1.0])


# Mean
# R1: the arithmetic mean of a non-empty sequence.


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # Ordinary case: (1 + 2 + 3) / 3.
        ([1.0, 2.0, 3.0], 2.0),
        # Boundary length: a single element.
        ([5.0], 5.0),
        # Symmetric values around zero cancel out exactly.
        ([-2.0, 2.0], 0.0),
    ],
)
def test_mean(values: list[float], expected: float) -> None:
    assert mean(values) == pytest.approx(expected)


@pytest.mark.unit
def test_mean_rejects_empty() -> None:
    # R3: an empty sequence would mean division by zero, so the function has
    # to fail with an explicit message instead.
    with pytest.raises(ValueError, match="must not be empty"):
        mean([])


# Variance and standard deviation
# R1: population variance (divide by n, not by n - 1) and its square root.
# The convention matters because the scaler in ``mlops.data`` relies on
# exactly the same definition.


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "expected_variance", "expected_std"),
    [
        # Mean is 2, deviations are +-1, so variance and std are both 1.
        ([1.0, 3.0], 1.0, 1.0),
        # A constant sequence has zero spread and must not raise.
        ([2.0, 2.0, 2.0], 0.0, 0.0),
        # Three points around the mean: variance = 8 / 3.
        ([0.0, 2.0, 4.0], 8.0 / 3.0, (8.0 / 3.0) ** 0.5),
    ],
)
def test_variance_and_deviation(
    values: list[float], expected_variance: float, expected_std: float
) -> None:
    assert variance(values) == pytest.approx(expected_variance)
    assert standard_deviation(values) == pytest.approx(expected_std)


@pytest.mark.unit
def test_variance_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        variance([])


@pytest.mark.unit
def test_standard_deviation_rejects_empty() -> None:
    # The error must propagate from ``variance`` and not be swallowed on the
    # way out through the square root.
    with pytest.raises(ValueError, match="must not be empty"):
        standard_deviation([])


# Linear system solver
# Covers R1 and R4: solve ``matrix @ x = rhs`` with Gaussian elimination and
# partial pivoting, and return the full solution vector.


@pytest.mark.unit
def test_solve_linear_system() -> None:
    # 2x + y = 5 and x + 3y = 10, therefore x = 1 and y = 3.
    solution = solve_linear_system([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
    assert solution == pytest.approx([1.0, 3.0])


@pytest.mark.unit
def test_solve_linear_system_with_pivoting() -> None:
    # R4: the leading element of the first row is zero. Without swapping
    # rows the algorithm would divide by zero, so pivoting is the behaviour
    # under test.
    solution = solve_linear_system([[0.0, 1.0], [1.0, 0.0]], [2.0, 3.0])
    assert solution == pytest.approx([3.0, 2.0])


@pytest.mark.unit
def test_solve_linear_system_identity() -> None:
    # An identity matrix must return the right-hand side unchanged.
    solution = solve_linear_system([[1.0, 0.0], [0.0, 1.0]], [2.0, 3.0])
    assert solution == pytest.approx([2.0, 3.0])


@pytest.mark.unit
def test_solve_linear_system_rejects_singular_matrix() -> None:
    # R4: the second row is the first row multiplied by two, so the system
    # has no unique solution. Returning inf or nan here would silently
    # poison the model weights later.
    with pytest.raises(ValueError, match="singular"):
        solve_linear_system([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("matrix", "rhs", "message"),
    [
        # No rows at all.
        ([], [], "must not be empty"),
        # One row with two columns: not a square system.
        ([[1.0, 2.0]], [1.0], "must be square"),
        # Two right-hand side values for a 1x1 matrix.
        ([[1.0]], [1.0, 2.0], "rhs length"),
    ],
)
def test_solve_linear_system_rejects_bad_shapes(
    matrix: list[list[float]], rhs: list[float], message: str
) -> None:
    # R3: each shape check is validated separately, so the error message
    # points at the actual problem instead of a generic failure.
    with pytest.raises(ValueError, match=message):
        solve_linear_system(matrix, rhs)
