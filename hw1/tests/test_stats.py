from math import sqrt

import pytest

from mlops_hw1 import mode, rms


def test_rms() -> None:
    assert rms([3.0, 4.0]) == pytest.approx(sqrt(12.5))


def test_rms_single_value() -> None:
    assert rms([5.0]) == 5.0


def test_rms_empty() -> None:
    with pytest.raises(ValueError):
        rms([])


def test_mode() -> None:
    assert mode([1.0, 2.0, 2.0, 3.0]) == 2.0


def test_mode_tie_returns_smallest() -> None:
    assert mode([7.0, 5.0, 7.0, 5.0]) == 5.0


def test_mode_single_value() -> None:
    assert mode([4.0]) == 4.0


def test_mode_empty() -> None:
    with pytest.raises(ValueError):
        mode([])
