"""Block and integration tests for the DVC pipeline stages.

Blocks under test
-----------------
Каждая стадия пайплайна в отдельности (``run_generate``, ``run_prepare``,
``run_train``, ``run_evaluate``) и их связка (``run_all``). Проверяется
не математика (она покрыта в ``test_blocks``/``test_model``), а контракт
стадии: какие файлы создаются, что записано в артефакты и совпадает ли
это с тем, что ожидают следующие стадии.

Requirements
------------
1. Каждая стадия создаёт ровно свои артефакты и ничего не теряет.
2. Артефакты согласованы: размеры сплитов в сумме дают исходный датасет,
   число весов равно числу признаков, метрики на диске равны возвращённым.
3. Полный прогон даёт высокий R² — данные линейны, модель должна их выучить.
4. Пайплайн воспроизводим: два прогона дают одинаковые метрики.
5. Переопределение ``test_size`` проверяется и применяется.
"""

import json

import pytest

from mlops.cli import run_all, run_evaluate, run_generate, run_prepare, run_train
from mlops.config import Paths, load_params
from mlops.data import load_csv, load_scaler
from mlops.model import load_model


# --- Стадия generate -------------------------------------------------------
# Требование: создан raw.csv с числом строк, равным n_samples.


@pytest.mark.integration
def test_generate_stage_creates_raw_file(pipeline_root) -> None:
    paths = Paths(root=pipeline_root)
    assert run_generate(paths).exists()
    assert len(load_csv(paths.raw)) == load_params(paths.params)["n_samples"]


# --- Стадия prepare --------------------------------------------------------
# Требование: три артефакта (train, test, scaler), строки не потеряны,
# размерности скейлера совпадают с числом признаков.


@pytest.mark.integration
def test_prepare_stage_creates_splits_and_scaler(pipeline_root) -> None:
    paths = Paths(root=pipeline_root)
    run_generate(paths)
    train_path, test_path, scaler_path = run_prepare(paths)

    assert train_path.exists()
    assert test_path.exists()
    assert scaler_path.exists()

    train = load_csv(train_path)
    test = load_csv(test_path)
    raw = load_csv(paths.raw)
    # Инвариант: split не добавляет и не теряет строки.
    assert len(train) + len(test) == len(raw)

    scaler = load_scaler(scaler_path)
    assert len(scaler.means) == raw.n_features


# --- Стадия train ----------------------------------------------------------
# Требование: модель сохранена на диск и совпадает с тем,
# что вернула стадия (иначе evaluate посчитает метрики по другой модели).


@pytest.mark.integration
def test_train_stage_creates_model(pipeline_root) -> None:
    paths = Paths(root=pipeline_root)
    run_generate(paths)
    run_prepare(paths)
    weights = run_train(paths)

    assert len(weights) == load_params(paths.params)["n_features"]
    model = load_model(paths.model)
    assert model.weights == pytest.approx(weights)


# --- Стадия evaluate -------------------------------------------------------
# Требование: полный набор метрик в JSON, значения совпадают
# с возвращённым словарём (DVC читает именно файл).


@pytest.mark.integration
def test_evaluate_stage_creates_metrics(pipeline_root) -> None:
    paths = Paths(root=pipeline_root)
    run_generate(paths)
    run_prepare(paths)
    run_train(paths)
    metrics = run_evaluate(paths)

    assert set(metrics) == {"mse", "rmse", "mae", "r2"}
    saved = json.loads(paths.metrics.read_text(encoding="utf-8"))
    assert saved == pytest.approx(metrics)


# --- Связка стадий run_all -------------------------------------------------
# Требование: сквозной прогон даёт осмысленное качество и согласованные метрики.


@pytest.mark.integration
def test_full_pipeline_reaches_high_r2(pipeline_root) -> None:
    paths = Paths(root=pipeline_root)
    metrics = run_all(paths)
    # Данные линейны по построению, поэтому R² обязан быть высоким;
    # порог 0.9 заведомо ниже ожидаемых ~0.99 и ловит регрессии.
    assert metrics["r2"] > 0.9
    assert metrics["mse"] >= 0.0
    assert metrics["rmse"] == pytest.approx(metrics["mse"] ** 0.5)


@pytest.mark.integration
def test_pipeline_is_reproducible(pipeline_root) -> None:
    # Требование: повторный прогон на тех же params даёт те же метрики —
    # это основа доверия к DVC-пайплайну.
    paths = Paths(root=pipeline_root)
    first = run_all(paths)
    second = run_all(paths)
    assert first == pytest.approx(second)


@pytest.mark.integration
def test_pipeline_rejects_bad_test_size_override(pipeline_root) -> None:
    # Требование: переопределение параметра валидируется так же строго,
    # как и значение из params.yaml.
    paths = Paths(root=pipeline_root)
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_all(paths, test_size=2.0)


@pytest.mark.integration
def test_pipeline_accepts_test_size_override(pipeline_root) -> None:
    # Требование: переопределение доходит до params.yaml и до сплита.
    paths = Paths(root=pipeline_root)
    run_all(paths, test_size=0.4)
    assert load_params(paths.params)["test_size"] == pytest.approx(0.4)
    test = load_csv(paths.test)
    train = load_csv(paths.train)
    assert len(test) / (len(test) + len(train)) == pytest.approx(0.4, abs=0.05)
