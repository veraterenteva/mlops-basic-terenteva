import pytest

from mlops_hw1 import mean


def test_mean() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_mean_single_value() -> None:
    assert mean([5.0]) == 5.0


def test_mean_empty() -> None:
    with pytest.raises(ValueError):
        mean([])
