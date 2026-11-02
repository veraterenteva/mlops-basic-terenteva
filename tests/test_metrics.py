"""Unit tests for regression metrics.

Block under test
----------------
``mlops.metrics`` — метрики качества регрессии. Именно они попадают
в ``data/metrics.json`` на стадии ``evaluate`` и по ним принимается
решение, годится ли модель.

Requirements
------------
1. Формулы соответствуют классическим определениям MSE, RMSE, MAE, R².
2. Метрики согласованы между собой: RMSE = sqrt(MSE).
3. Граничные случаи обработаны: идеальное предсказание, константные
   таргеты, один элемент, отрицательные значения.
4. Некорректный вход (пустой список, разная длина) даёт ``ValueError``.
"""

import pytest

from mlops.metrics import all_metrics, mae, mse, r2, rmse

# --- Блок: mse / mae -------------------------------------------------------
# Требование: MSE = mean((a-p)^2), MAE = mean(|a-p|).


@pytest.mark.unit
@pytest.mark.parametrize(
    ("actual", "predicted", "expected_mse", "expected_mae"),
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0.0, 0.0),  # идеальное совпадение
        ([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], 2.0 / 3.0, 2.0 / 3.0),  # средняя ошибка
        ([0.0, 0.0], [3.0, -3.0], 9.0, 3.0),  # знак ошибки не важен для модуля
        ([10.0], [12.0], 4.0, 2.0),  # граничный случай: один элемент
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


# --- Блок: rmse ------------------------------------------------------------
# Требование: RMSE — не самостоятельная формула, а корень из MSE;
# тест закрепляет это соглашение, чтобы реализация не разошлась.


@pytest.mark.unit
def test_rmse_is_square_root_of_mse() -> None:
    actual = [1.0, 2.0, 3.0]
    predicted = [1.5, 1.5, 3.5]
    assert rmse(actual, predicted) == pytest.approx(mse(actual, predicted) ** 0.5)


# --- Блок: r2 --------------------------------------------------------------
# Требование: R² = 1 - RSS/TSS. Соглашение проекта: если у таргетов
# нулевая дисперсия, возвращаем 0.0 вместо деления на ноль.


@pytest.mark.unit
@pytest.mark.parametrize(
    ("actual", "predicted", "expected"),
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0),  # идеальное предсказание
        # RSS = 0.5, TSS = 2.0 -> 1 - 0.5 / 2.0 = 0.75
        ([1.0, 2.0, 3.0], [1.5, 2.0, 2.5], 0.75),
        # Предсказание средним даёт RSS = TSS, поэтому R2 ровно ноль.
        ([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], 0.0),
        # Нулевая дисперсия таргетов: метрика не определена,
        # по соглашению проекта возвращается 0.0.
        ([2.0, 2.0, 2.0], [1.0, 2.0, 3.0], 0.0),
    ],
)
def test_r2(actual: list[float], predicted: list[float], expected: float) -> None:
    assert r2(actual, predicted) == pytest.approx(expected)


# --- Блок: all_metrics -----------------------------------------------------
# Требование: единая точка сбора метрик, контракт ключей фиксирован,
# потому что их читает evaluate-стадия и парсит DVC.


@pytest.mark.unit
def test_all_metrics_returns_every_key() -> None:
    metrics = all_metrics([1.0, 2.0], [1.0, 3.0])
    assert set(metrics) == {"mse", "rmse", "mae", "r2"}
    assert metrics["mse"] == pytest.approx(0.5)


# --- Валидация входа -------------------------------------------------------
# Требование: все метрики одинаково строго проверяют размеры, чтобы
# нельзя было случайно посчитать метрику на несовпадающих массивах.


@pytest.mark.unit
@pytest.mark.parametrize(
    ("actual", "predicted", "message"),
    [
        ([], [], "must not be empty"),  # пустой вход
        ([1.0, 2.0], [1.0], "same length"),  # разная длина
    ],
)
def test_metrics_validate_input(
    actual: list[float], predicted: list[float], message: str
) -> None:
    for metric in (mse, rmse, mae, r2):
        with pytest.raises(ValueError, match=message):
            metric(actual, predicted)
