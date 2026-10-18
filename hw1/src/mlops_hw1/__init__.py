"""Home assignment 1: minimal package checked by Ruff and ty."""

from collections import Counter
from math import sqrt


def rms(values: list[float]) -> float:
    """Return the root mean square of a non-empty list of numbers."""
    if not values:
        raise ValueError("values must not be empty")
    return sqrt(sum(value**2 for value in values) / len(values))


def mode(values: list[float]) -> float:
    """Return the most frequent value of a non-empty list of numbers.

    If several values occur equally often, the smallest one is returned.
    """
    if not values:
        raise ValueError("values must not be empty")
    counts = Counter(values)
    best_count = max(counts.values())
    return min(value for value, count in counts.items() if count == best_count)


def main() -> None:
    """Entry point used for a smoke run of the package."""
    numbers = [3.0, 4.0, 4.0]
    print(f"RMS: {rms(numbers)}")
    print(f"Mode: {mode(numbers)}")


if __name__ == "__main__":
    main()
