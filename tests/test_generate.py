"""Tests for the generate stage: synthetic dataset creation.

Block under test
----------------
``mlops.generate`` — первая стадия пайплайна. Она создаёт
``data/raw.csv`` по известному закону
``y = 2.5*f0 - 1.5*f1 + 0.75*f2 + шум``. Благодаря известным
коэффициентам качество модели потом можно проверить по существу,
а не только по метрикам.

Requirements
------------
1. Строка данных собирается по линейному закону: признаки как есть,
   целевое значение = взвешенная сумма плюс шум.
2. Один и тот же seed даёт один и тот же датасет (нужно для DVC).
3. Число строк и признаков соответствует параметрам.
4. Некорректные параметры (нулевое число строк, отрицательный шум,
   неподдерживаемое число признаков) отвергаются.
5. CSV пишется с заголовком ``f0..fn, target`` и создаёт вложенные папки.
"""

import csv
import random

import pytest

from mlops.config import DEFAULT_PARAMS, Params
from mlops.generate import (
    TRUE_WEIGHTS,
    build_row,
    generate_dataset,
    generate_rows,
    write_csv,
)

# --- Блок: build_row -------------------------------------------------------
# Требование: признаки не меняются, целевое значение считается по формуле.


@pytest.mark.unit
def test_build_row_appends_target() -> None:
    row = build_row([1.0, 2.0, 3.0], TRUE_WEIGHTS, noise=0.0, rng=random.Random(0))
    expected = sum(w * v for w, v in zip(TRUE_WEIGHTS, [1.0, 2.0, 3.0], strict=True))
    assert row[:3] == [1.0, 2.0, 3.0]
    assert row[3] == pytest.approx(expected)


# --- Блок: generate_rows ---------------------------------------------------
# Требование: воспроизводимость и соответствие параметрам.


@pytest.mark.unit
def test_generate_rows_is_deterministic(params: Params) -> None:
    # Один seed -> побайтово одинаковые строки, иначе DVC будет
    # считать стадию изменившейся при каждом запуске.
    assert generate_rows(params) == generate_rows(params)


@pytest.mark.unit
def test_generate_rows_respects_sample_count(params: Params) -> None:
    # Требование: n_samples строк, каждая длиной n_features + 1 (target).
    rows = generate_rows(params)
    assert len(rows) == params["n_samples"]
    assert all(len(row) == params["n_features"] + 1 for row in rows)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("n_samples", "message"),
    [(0, "must be positive"), (-5, "must be positive")],
)
def test_generate_rows_rejects_bad_sample_count(
    params: Params, n_samples: int, message: str
) -> None:
    # Требование: пустой или отрицательный датасет — ошибка.
    broken: Params = {**params, "n_samples": n_samples}
    with pytest.raises(ValueError, match=message):
        generate_rows(broken)


@pytest.mark.unit
def test_generate_rows_rejects_negative_noise(params: Params) -> None:
    # Отрицательная сигма у gauss() не имеет смысла.
    broken: Params = {**params, "noise": -1.0}
    with pytest.raises(ValueError, match="non-negative"):
        generate_rows(broken)


@pytest.mark.unit
def test_generate_rows_rejects_unknown_feature_count(params: Params) -> None:
    # Истинных весов всего три, поэтому другое число признаков
    # не к чему применить.
    broken: Params = {**params, "n_features": 5}
    with pytest.raises(ValueError, match="n_features must be"):
        generate_rows(broken)


# --- Блок: write_csv -------------------------------------------------------
# Требование: корректный заголовок, создание директорий, отказ на пустом входе.


@pytest.mark.unit
def test_write_csv_creates_header(tmp_path) -> None:
    # Путь во вложенной папке проверяет, что parent создаётся сам.
    path = tmp_path / "nested" / "out.csv"
    write_csv(path, [[1.0, 2.0, 3.0]], n_features=2)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["f0", "f1", "target"]
    assert rows[1] == ["1.0", "2.0", "3.0"]


@pytest.mark.unit
def test_write_csv_rejects_empty_rows(tmp_path) -> None:
    # Требование: не создавать файл-обманку без данных.
    with pytest.raises(ValueError, match="must not be empty"):
        write_csv(tmp_path / "out.csv", [], n_features=1)


# --- Блок: generate_dataset (интеграция внутри стадии) ---------------------
# Требование: на выходе реальный файл с заголовком и нужным числом строк.


@pytest.mark.integration
def test_generate_dataset_writes_file(tmp_path) -> None:
    path = generate_dataset(DEFAULT_PARAMS, tmp_path / "raw.csv")
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == DEFAULT_PARAMS["n_samples"] + 1
