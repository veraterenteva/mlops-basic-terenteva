"""Unit tests for the data block: loading, cleaning, splitting, scaling.

Block under test
----------------
``mlops.data`` — слой данных. Он стоит между генератором и моделью:
читает CSV, вычищает мусор, делит выборку и масштабирует признаки.
Ошибка здесь незаметна для модели, но портит и обучение, и метрики,
поэтому блок проверяется особенно подробно.

Requirements
------------
1. ``load_csv`` разбирает заголовок, отделяет целевую колонку и
   сообщает об ошибках формата (нет файла, нет колонки, короткая строка,
   файл только с заголовком).
2. ``clean`` убирает дубликаты и NaN-строки, но не «съедает» датасет
   целиком: полностью пустой результат — ошибка.
3. ``train_test_split`` детерминирован при одном seed, меняется при
   другом, сохраняет общее число строк и корректно считает размер теста.
4. ``StandardScaler`` даёт нулевое среднее и единичную дисперсию,
   не падает на константном столбце, проверяет число признаков и
   переживает сериализацию в JSON.
"""

import pytest

from mlops.data import (
    Dataset,
    StandardScaler,
    clean,
    load_csv,
    load_scaler,
    save_scaler,
    train_test_split,
    write_dataset,
)

# --- Блок: load_csv --------------------------------------------------------
# Требование: корректный разбор файла, который создаёт стадия generate.


@pytest.mark.unit
def test_load_csv_reads_generated_file(dataset: Dataset) -> None:
    # Датасет из фикстуры: 40 строк, 3 признака, имена f0..f2 из заголовка.
    assert len(dataset) == 40
    assert dataset.n_features == 3
    assert dataset.feature_names == ["f0", "f1", "f2"]


@pytest.mark.unit
def test_load_csv_missing_file(tmp_path) -> None:
    # Требование: отсутствующий файл — FileNotFoundError, а не пустой датасет.
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")


@pytest.mark.unit
def test_load_csv_rejects_unknown_target(tmp_path, csv_writer) -> None:
    # Требование: если целевой колонки нет, обучение бессмысленно — ошибка.
    path = csv_writer(tmp_path / "x.csv", ["f0", "y"], [[1.0, 2.0]])
    with pytest.raises(ValueError, match="target column not found"):
        load_csv(path, target_column="target")


@pytest.mark.unit
def test_load_csv_rejects_short_row(tmp_path, csv_writer) -> None:
    # Требование: битая строка не должна молча превращаться в строку
    # с другим числом признаков — сообщение содержит номер и ожидание.
    path = csv_writer(tmp_path / "x.csv", ["f0", "target"], [[1.0]])
    with pytest.raises(ValueError, match="expected 2"):
        load_csv(path)


@pytest.mark.unit
def test_load_csv_rejects_header_only(tmp_path, csv_writer) -> None:
    # Требование: пустая таблица бесполезна для обучения.
    path = csv_writer(tmp_path / "x.csv", ["f0", "target"], [])
    with pytest.raises(ValueError, match="at least one row"):
        load_csv(path)


# --- Блок: clean -----------------------------------------------------------
# Требование: убрать шум, но не скрыть проблему полного опустошения.


@pytest.mark.unit
def test_clean_removes_duplicates() -> None:
    # Две одинаковые строки -> остаётся одна, уникальная сохраняется.
    dataset = Dataset(
        features=[[1.0], [1.0], [2.0]],
        targets=[1.0, 1.0, 2.0],
        feature_names=["f0"],
    )
    cleaned = clean(dataset)
    assert len(cleaned) == 2


@pytest.mark.unit
def test_clean_removes_nan_rows() -> None:
    # Требование: строка с NaN выкидывается, остальные не затрагиваются.
    dataset = Dataset(
        features=[[float("nan")], [2.0]],
        targets=[1.0, 2.0],
        feature_names=["f0"],
    )
    cleaned = clean(dataset)
    assert len(cleaned) == 1
    assert cleaned.features == [[2.0]]


@pytest.mark.unit
def test_clean_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        clean(Dataset(features=[], targets=[], feature_names=[]))


@pytest.mark.unit
def test_clean_rejects_all_rows_dropped() -> None:
    # Требование: нельзя вернуть пустой датасет — лучше явная ошибка,
    # иначе следующая стадия упадёт с непонятным сообщением.
    dataset = Dataset(
        features=[[float("nan")]],
        targets=[1.0],
        feature_names=["f0"],
    )
    with pytest.raises(ValueError, match="no rows left"):
        clean(dataset)


# --- Блок: train_test_split -----------------------------------------------
# Требование: воспроизводимое разбиение с контролем размера теста.


@pytest.mark.unit
@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_bad_test_size(clean_dataset: Dataset, test_size: float) -> None:
    # Границы: 0 и 1 недопустимы, отрицательное и >1 тоже.
    with pytest.raises(ValueError, match="between 0 and 1"):
        train_test_split(clean_dataset, test_size=test_size, seed=1)


@pytest.mark.unit
def test_split_rejects_single_row() -> None:
    # Одну строку нельзя разделить на две части.
    dataset = Dataset(features=[[1.0]], targets=[1.0], feature_names=["f0"])
    with pytest.raises(ValueError, match="at least two rows"):
        train_test_split(dataset, test_size=0.5, seed=1)


@pytest.mark.unit
@pytest.mark.parametrize("test_size", [0.1, 0.25, 0.5])
def test_split_sizes(clean_dataset: Dataset, test_size: float) -> None:
    # Требование: строки не теряются и не дублируются,
    # размер теста считается как round(n * test_size), но не меньше 1.
    splits = train_test_split(clean_dataset, test_size=test_size, seed=3)
    total = len(splits.train) + len(splits.test)
    assert total == len(clean_dataset)
    assert len(splits.test) == max(1, round(len(clean_dataset) * test_size))


@pytest.mark.unit
def test_split_is_deterministic(clean_dataset: Dataset) -> None:
    # Требование: один seed -> одинаковое разбиение (нужно для DVC).
    first = train_test_split(clean_dataset, test_size=0.25, seed=5)
    second = train_test_split(clean_dataset, test_size=0.25, seed=5)
    assert first.train.features == second.train.features
    assert first.test.features == second.test.features


@pytest.mark.unit
def test_split_differs_for_different_seeds(clean_dataset: Dataset) -> None:
    # Требование: seed действительно влияет, а не игнорируется.
    first = train_test_split(clean_dataset, test_size=0.25, seed=5)
    second = train_test_split(clean_dataset, test_size=0.25, seed=6)
    assert first.train.features != second.train.features


# --- Блок: StandardScaler -------------------------------------------------
# Требование: масштабирование по статистикам train-части, применяемое
# затем к тесту без утечки данных.


@pytest.mark.unit
def test_scaler_fit_produces_zero_mean_unit_std(splits) -> None:
    # Главное свойство скейлера: после transform среднее 0, дисперсия 1.
    scaler = StandardScaler.fit(splits.train)
    scaled = scaler.transform(splits.train)
    for column in range(scaled.n_features):
        values = [row[column] for row in scaled.features]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        assert mean == pytest.approx(0.0, abs=1e-9)
        assert variance == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_scaler_handles_constant_column() -> None:
    # Требование: константный признак не должен давать деление на ноль;
    # std подменяется на 1.0, а значения вырождаются в нули.
    dataset = Dataset(
        features=[[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]],
        targets=[0.0, 1.0, 2.0],
        feature_names=["f0", "f1"],
    )
    scaler = StandardScaler.fit(dataset)
    assert scaler.stds[0] == 1.0
    scaled = scaler.transform(dataset)
    assert [row[0] for row in scaled.features] == [0.0, 0.0, 0.0]


@pytest.mark.unit
def test_scaler_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        StandardScaler.fit(Dataset(features=[], targets=[], feature_names=[]))


@pytest.mark.unit
def test_scaler_rejects_feature_mismatch(splits) -> None:
    # Требование: нельзя применить скейлер с другим числом признаков —
    # это признак перепутанных артефактов из DVC-кэша.
    scaler = StandardScaler(means=[0.0], stds=[1.0])
    with pytest.raises(ValueError, match="does not match the scaler"):
        scaler.transform(splits.train)


@pytest.mark.unit
def test_scaler_roundtrip(tmp_path, splits) -> None:
    # Требование: сохранение и чтение JSON не искажает статистики,
    # иначе train и evaluate разойдутся.
    scaler = StandardScaler.fit(splits.train)
    path = tmp_path / "scaler.json"
    save_scaler(path, scaler)
    restored = load_scaler(path)
    assert restored.means == pytest.approx(scaler.means)
    assert restored.stds == pytest.approx(scaler.stds)


@pytest.mark.unit
def test_load_scaler_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_scaler(tmp_path / "missing.json")


@pytest.mark.unit
def test_scaler_from_dict_validates_length() -> None:
    # Требование: битый JSON не должен давать скейлер с «рваными» списками.
    with pytest.raises(ValueError, match="same length"):
        StandardScaler.from_dict({"means": [0.0, 1.0], "stds": [1.0]})


# --- Блок: write_dataset ---------------------------------------------------
# Требование: записанный датасет читается обратно без потерь.


@pytest.mark.unit
def test_write_dataset_roundtrip(tmp_path, clean_dataset: Dataset) -> None:
    path = tmp_path / "out.csv"
    write_dataset(path, clean_dataset)
    restored = load_csv(path)
    assert len(restored) == len(clean_dataset)
    assert restored.feature_names == clean_dataset.feature_names
