"""Home assignment 1: minimal package checked by Ruff and ty."""


def mean(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list of numbers."""
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def main() -> None:
    """Entry point used for a smoke run of the package."""
    numbers = [1.0, 2.0, 3.0]
    print(f"Mean: {mean(numbers)}")


if __name__ == "__main__":
    main()
