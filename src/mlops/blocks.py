"""Reusable numeric blocks built on the standard library only.

These are the smallest units of the pipeline, which makes them convenient
for unit testing: dot product, mean, variance and a linear system solver.
"""

from collections.abc import Sequence
from math import sqrt


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the dot product of two equal-length vectors."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    return sum(a * b for a, b in zip(left, right, strict=True))


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty sequence."""
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def variance(values: Sequence[float]) -> float:
    """Return the population variance of a non-empty sequence."""
    mu = mean(values)
    return sum((value - mu) ** 2 for value in values) / len(values)


def standard_deviation(values: Sequence[float]) -> float:
    """Return the population standard deviation of a non-empty sequence."""
    return sqrt(variance(values))


def solve_linear_system(
    matrix: Sequence[Sequence[float]], rhs: Sequence[float]
) -> list[float]:
    """Solve ``matrix @ x = rhs`` with Gaussian elimination.

    Partial pivoting is used for numerical stability. A singular matrix
    raises ``ValueError``.
    """
    size = len(matrix)
    if size == 0:
        raise ValueError("matrix must not be empty")
    if len(rhs) != size:
        raise ValueError("rhs length must match matrix size")
    if any(len(row) != size for row in matrix):
        raise ValueError("matrix must be square")

    augmented = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]

        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * base
                for value, base in zip(augmented[row], augmented[column], strict=True)
            ]

    return [augmented[index][size] for index in range(size)]
