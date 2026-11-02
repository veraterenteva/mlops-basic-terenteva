"""Unit tests for the numeric primitives of the pipeline.

Block under test
----------------
``mlops.blocks`` — строительные блоки, на которых держится вся математика
пайплайна: скалярное произведение, среднее, дисперсия, стандартное
отклонение и решатель системы линейных уравнений. Модель
``RidgeRegression`` не работает без них, поэтому ошибка здесь ломает
обучение целиком.

Requirements
------------
1. Корректность формул: результаты совпадают с эталонными значениями,
   посчитанными вручную.
2. Работа с граничными длинами: один элемент, нулевой вектор,
   отрицательные значения.
3. Явные ошибки на некорректном входе: пустая последовательность,
   векторы разной длины, несовместимые размеры матрицы.
4. Устойчивость решателя: выбор главного элемента (pivoting) должен
   работать, а вырожденная матрица — отвергаться.
"""

import pytest

from mlops.blocks import dot, mean, solve_linear_system, standard_deviation, variance


# --- Блок: dot -------------------------------------------------------------
# Требование: скалярное произведение двух векторов одинаковой длины,
# включая нулевые, отрицательные значения и векторы из одного элемента.


@pytest.mark.unit
@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 2.0], [3.0, 4.0], 11.0),  # 1*3 + 2*4
        ([0.0, 0.0], [5.0, 5.0], 0.0),  # нулевой вектор
        ([-1.0, 1.0], [-1.0, 1.0], 2.0),  # отрицательные компоненты
        ([2.0], [0.5], 1.0),  # граничный случай: один элемент
    ],
)
def test_dot(left: list[float], right: list[float], expected: float) -> None:
    assert dot(left, right) == pytest.approx(expected)


@pytest.mark.unit
def test_dot_rejects_different_lengths() -> None:
    # Требование: несовпадение длины — ошибка, а не молчаливое усечение
    # через zip(), которое дало бы неверный результат без предупреждения.
    with pytest.raises(ValueError):
        dot([1.0, 2.0], [1.0])


# --- Блок: mean ------------------------------------------------------------
# Требование: арифметическое среднее непустой последовательности.


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 2.0, 3.0], 2.0),  # обычный случай
        ([5.0], 5.0),  # граничный случай: один элемент
        ([-2.0, 2.0], 0.0),  # симметричные значения около нуля
    ],
)
def test_mean(values: list[float], expected: float) -> None:
    assert mean(values) == pytest.approx(expected)


@pytest.mark.unit
def test_mean_rejects_empty() -> None:
    # Требование: пустая последовательность — деление на ноль,
    # поэтому функция обязана упасть с понятным сообщением.
    with pytest.raises(ValueError, match="must not be empty"):
        mean([])


# --- Блок: variance / standard_deviation -----------------------------------
# Требование: популяционная дисперсия (деление на n, а не на n-1)
# и её квадратный корень.


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "expected_variance", "expected_std"),
    [
        ([1.0, 3.0], 1.0, 1.0),  # среднее 2, отклонения ±1
        ([2.0, 2.0, 2.0], 0.0, 0.0),  # нулевая дисперсия без падения
        ([0.0, 2.0, 4.0], 8.0 / 3.0, (8.0 / 3.0) ** 0.5),  # три точки
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
    # Проверяем, что ошибка пробрасывается из variance, а не теряется.
    with pytest.raises(ValueError, match="must not be empty"):
        standard_deviation([])


# --- Блок: solve_linear_system --------------------------------------------
# Требование: решить matrix @ x = rhs методом Гаусса с выбором главного
# элемента, вернуть полный вектор решения.


@pytest.mark.unit
def test_solve_linear_system() -> None:
    # 2x + y = 5, x + 3y = 10  ->  x = 1, y = 3
    solution = solve_linear_system([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
    assert solution == pytest.approx([1.0, 3.0])


@pytest.mark.unit
def test_solve_linear_system_with_pivoting() -> None:
    # Нулевой ведущий элемент: без перестановки строк деление на ноль.
    # Требование: pivoting обязателен.
    solution = solve_linear_system([[0.0, 1.0], [1.0, 0.0]], [2.0, 3.0])
    assert solution == pytest.approx([3.0, 2.0])


@pytest.mark.unit
def test_solve_linear_system_identity() -> None:
    # Требование: единичная матрица возвращает правую часть без искажений.
    solution = solve_linear_system([[1.0, 0.0], [0.0, 1.0]], [2.0, 3.0])
    assert solution == pytest.approx([2.0, 3.0])


@pytest.mark.unit
def test_solve_linear_system_rejects_singular_matrix() -> None:
    # Требование: вырожденная матрица (вторая строка = первая * 2)
    # должна давать ValueError, а не бесконечность или NaN.
    with pytest.raises(ValueError, match="singular"):
        solve_linear_system([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("matrix", "rhs", "message"),
    [
        ([], [], "must not be empty"),  # пустая система
        ([[1.0, 2.0]], [1.0], "must be square"),  # не квадратная матрица
        ([[1.0]], [1.0, 2.0], "rhs length"),  # длина rhs не совпадает
    ],
)
def test_solve_linear_system_rejects_bad_shapes(
    matrix: list[list[float]], rhs: list[float], message: str
) -> None:
    # Требование: каждая проверка формы даёт своё сообщение, чтобы
    # диагностировать ошибку было просто.
    with pytest.raises(ValueError, match=message):
        solve_linear_system(matrix, rhs)
