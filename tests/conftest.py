"""Shared fixtures for the pipeline tests.

Design notes
------------
Фикстуры выстроены в цепочку зависимостей: ``params`` -> ``raw_csv`` ->
``dataset`` -> ``clean_dataset`` -> ``splits`` -> ``trained_model``.
Каждый тест берёт только нужное звено, поэтому подготовка данных
не дублируется, а тесты остаются независимыми.

Все фикстуры работают во временных каталогах (``tmp_path``), поэтому
прогон не зависит от содержимого рабочей папки и не оставляет мусора.

Requirement for the whole suite
-------------------------------
Тесты не должны ходить в сеть, зависеть от текущей даты или от файлов вне
``tmp_path``. ``params`` специально уменьшает ``n_samples`` до 40, чтобы
набор из 100+ тестов проходил за секунды.
"""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mlops.config import DEFAULT_PARAMS, Params
from mlops.data import (
    Dataset,
    Splits,
    StandardScaler,
    clean,
    load_csv,
    train_test_split,
)
from mlops.generate import generate_dataset
from mlops.model import RidgeRegression, save_model

# --- Параметры -------------------------------------------------------------
# Требование: маленький, но валидный набор параметров, чтобы тесты
# были быстрыми. noise=0.0 упрощает проверки точности модели.


@pytest.fixture
def params(tmp_path: Path) -> Params:
    """Параметры пайплайна с уменьшенным числом строк и без шума."""
    return {
        **DEFAULT_PARAMS,
        "n_samples": 40,
        "noise": 0.0,
    }


@pytest.fixture
def params_file(tmp_path: Path, params: Params) -> Path:
    """``params.yaml`` на диске — нужен тестам, которые читают конфиг."""
    path = tmp_path / "params.yaml"
    lines = [f"{key}: {value}" for key, value in params.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- Данные ----------------------------------------------------------------
# Цепочка: сырой CSV -> Dataset -> очищенный -> сплиты -> обученная модель.
# Такая декомпозиция позволяет тестировать каждый блок на своём уровне
# и переиспользовать подготовку без копипасты.


@pytest.fixture
def raw_csv(tmp_path: Path, params: Params) -> Path:
    """Реальный CSV, созданный настоящей стадией генерации.

    Не мок: тесты работают с тем же форматом, что и пайплайн.
    """
    return generate_dataset(params, tmp_path / "data" / "raw.csv")


@pytest.fixture
def dataset(raw_csv: Path) -> Dataset:
    """Датасет, прочитанный из сгенерированного CSV."""
    return load_csv(raw_csv)


@pytest.fixture
def clean_dataset(dataset: Dataset) -> Dataset:
    """Очищенный датасет — вход для сплита и скейлера."""
    return clean(dataset)


@pytest.fixture
def splits(clean_dataset: Dataset) -> Splits:
    """Train/test разбиение с фиксированным seed для воспроизводимости."""
    return train_test_split(clean_dataset, test_size=0.25, seed=7)


@pytest.fixture
def scaled_splits(splits: Splits) -> tuple[Splits, StandardScaler]:
    """Масштабированные сплиты вместе со скейлером.

    Скейлер возвращается отдельно, потому что часть тестов проверяет
    его статистики, а часть — только преобразованные данные.
    """
    scaler = StandardScaler.fit(splits.train)
    scaled = Splits(
        train=scaler.transform(splits.train),
        test=scaler.transform(splits.test),
    )
    return scaled, scaler


# --- Модель ----------------------------------------------------------------
# Требование: обученная модель и её файловое представление доступны
# как отдельные фикстуры, чтобы не переобучать модель в каждом тесте.


@pytest.fixture
def trained_model(splits: Splits) -> RidgeRegression:
    """Модель, обученная на train-части.

    ``alpha`` почти нулевая: тесты проверяют математику решения,
    а не эффект регуляризации (для неё есть отдельный тест).
    """
    model = RidgeRegression(alpha=1e-9)
    return model.fit(splits.train.features, splits.train.targets)


@pytest.fixture
def model_file(tmp_path: Path, trained_model: RidgeRegression) -> Path:
    """Обученная модель, сохранённая в JSON."""
    path = tmp_path / "model.json"
    save_model(path, trained_model)
    return path


# --- Пайплайн --------------------------------------------------------------
# Требование: готовый рабочий каталог, чтобы e2e и интеграционные тесты
# запускали CLI и стадии без ручной подготовки файлов.


@pytest.fixture
def pipeline_root(tmp_path: Path, params: Params) -> Path:
    """Каталог, в котором уже лежит ``params.yaml``.

    Именно этот путь передаётся в ``--root`` при запуске CLI, поэтому
    все артефакты появляются внутри ``tmp_path`` и не засоряют репозиторий.
    """
    root = tmp_path / "pipeline"
    root.mkdir()
    lines = [f"{key}: {value}" for key, value in params.items()]
    (root / "params.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


# --- Фабрика данных для негативных тестов ---------------------------------
# Требование: возможность собрать произвольный «плохой» CSV прямо в тесте,
# не заводя отдельный файл-фикстуру на каждый случай.


@pytest.fixture
def csv_writer() -> Callable[[Path, list[str], list[list[Any]]], Path]:
    """Фабрика: пишет CSV с произвольным заголовком и строками.

    Возвращает функцию, а не путь, потому что негативным тестам нужны
    разные варианты файла (без целевой колонки, с короткой строкой,
    только с заголовком).
    """

    def write(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
        """Записать ``rows`` с заголовком ``header`` и вернуть путь."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    return write
