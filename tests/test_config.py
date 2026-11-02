"""Unit tests for the configuration block: parameters and file paths.

Block under test
----------------
``mlops.config`` — конфигурация пайплайна. Здесь живут ``Params``
(схема ``params.yaml``), ``Paths`` (раскладка файлов) и ``load_params``
(чтение и валидация). DVC отслеживает именно ``params.yaml``, поэтому
ошибка в этом блоке ломает воспроизводимость всего пайплайна.

Requirements
------------
1. ``Paths`` возвращает ровно те пути, которые ожидают стадии DVC.
2. ``load_params`` читает все ключи и приводит типы (int/float/str).
3. Отсутствующий файл даёт ``FileNotFoundError``.
4. Неполный или неверный YAML отвергается с понятным сообщением,
   а не падает позже в недрах обучения.
"""

import pytest

from mlops.cli import default_params_text
from mlops.config import DEFAULT_PARAMS, Paths, load_params

# --- Блок: Paths -----------------------------------------------------------
# Требование: имена артефактов синхронизированы с dvc.yaml;
# переименование в одном месте должно ломать этот тест.


@pytest.mark.unit
def test_paths_layout(tmp_path) -> None:
    paths = Paths(root=tmp_path)
    assert paths.raw == tmp_path / "data" / "raw.csv"
    assert paths.train == tmp_path / "data" / "train.csv"
    assert paths.test == tmp_path / "data" / "test.csv"
    assert paths.scaler == tmp_path / "data" / "scaler.json"
    assert paths.model == tmp_path / "data" / "model.json"
    assert paths.metrics == tmp_path / "data" / "metrics.json"
    assert paths.params == tmp_path / "params.yaml"


# --- Блок: load_params -----------------------------------------------------
# Требование: разбор YAML с приведением типов и валидацией схемы.


@pytest.mark.unit
def test_load_params_roundtrip(tmp_path) -> None:
    # Текст, который пишет CLI, должен читаться обратно без потерь.
    path = tmp_path / "params.yaml"
    path.write_text(default_params_text(), encoding="utf-8")
    assert load_params(path) == DEFAULT_PARAMS


@pytest.mark.unit
def test_load_params_reads_values(tmp_path) -> None:
    # Требование: значения приводятся к нужным типам —
    # int остаётся int, float сравнивается приближённо.
    path = tmp_path / "params.yaml"
    path.write_text(
        "\n".join(
            [
                "seed: 1",
                "n_samples: 5",
                "n_features: 3",
                "noise: 0.1",
                "test_size: 0.5",
                "ridge_alpha: 0.01",
                "target_column: target",
            ]
        ),
        encoding="utf-8",
    )
    params = load_params(path)
    assert params["seed"] == 1
    assert params["n_samples"] == 5
    assert params["test_size"] == pytest.approx(0.5)


@pytest.mark.unit
def test_load_params_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "missing.yaml")


@pytest.mark.unit
def test_load_params_rejects_missing_keys(tmp_path) -> None:
    # Требование: неполный конфиг — явная ошибка со списком ключей,
    # иначе стадия упадёт с KeyError в случайном месте.
    path = tmp_path / "params.yaml"
    path.write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing params"):
        load_params(path)


@pytest.mark.unit
def test_load_params_rejects_non_mapping(tmp_path) -> None:
    # Требование: YAML-список вместо словаря отвергается сразу.
    path = tmp_path / "params.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_params(path)
