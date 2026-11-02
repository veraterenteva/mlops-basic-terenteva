"""End-to-end tests: the whole pipeline through the command line interface.

System under test
-----------------
``mlops.cli`` — точка входа, которую вызывает DVC в каждой стадии.
E2E-тесты запускают её как пользователь: с флагами, во временном
каталоге, и проверяют результат на диске. Это единственный уровень,
который ловит ошибки склейки: неправильные пути, забытый вызов стадии,
несовпадение аргументов CLI и ``dvc.yaml``.

Requirements
------------
1. Команда ``all`` проходит все стадии и создаёт все шесть артефактов.
2. ``metrics.json`` валиден: нужные ключи и согласованные значения.
3. Стадии можно запускать по одной в правильном порядке.
4. ``--test-size`` действительно меняет конфигурацию.
5. CLI без команды завершается через ``SystemExit`` (поведение argparse).
"""

import json
from pathlib import Path

import pytest

from mlops.cli import main

# --- Полный прогон ---------------------------------------------------------
# Требование: одна команда доводит систему от пустого каталога
# до готовых метрик.


@pytest.mark.e2e
def test_cli_all_produces_every_artifact(pipeline_root: Path) -> None:
    exit_code = main(["--root", str(pipeline_root), "all"])

    assert exit_code == 0
    assert (pipeline_root / "data" / "raw.csv").exists()
    assert (pipeline_root / "data" / "train.csv").exists()
    assert (pipeline_root / "data" / "test.csv").exists()
    assert (pipeline_root / "data" / "scaler.json").exists()
    assert (pipeline_root / "data" / "model.json").exists()
    assert (pipeline_root / "data" / "metrics.json").exists()


@pytest.mark.e2e
def test_cli_all_writes_valid_metrics(pipeline_root: Path) -> None:
    main(["--root", str(pipeline_root), "all"])

    payload = json.loads(
        (pipeline_root / "data" / "metrics.json").read_text(encoding="utf-8")
    )
    # Контракт файла метрик: эти ключи читает DVC и человек.
    assert set(payload) == {"mse", "rmse", "mae", "r2"}
    assert payload["r2"] > 0.9
    assert payload["rmse"] == pytest.approx(payload["mse"] ** 0.5)


# --- Постадийный запуск ----------------------------------------------------
# Требование: стадии работают и по отдельности, в порядке зависимостей —
# именно так их вызывает DVC.


@pytest.mark.e2e
def test_cli_runs_stages_separately(pipeline_root: Path) -> None:
    for stage in ("generate", "prepare", "train", "evaluate"):
        assert main(["--root", str(pipeline_root), stage]) == 0

    assert (pipeline_root / "data" / "metrics.json").exists()


# --- Переопределение параметра --------------------------------------------
# Требование: флаг CLI доходит до params.yaml.


@pytest.mark.e2e
def test_cli_all_accepts_test_size_override(pipeline_root: Path) -> None:
    assert main(["--root", str(pipeline_root), "all", "--test-size", "0.3"]) == 0
    params_text = (pipeline_root / "params.yaml").read_text(encoding="utf-8")
    assert "test_size: 0.3" in params_text


# --- Контракт CLI ----------------------------------------------------------
# Требование: без обязательной подкоманды argparse завершает процесс.


@pytest.mark.e2e
def test_cli_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        main([])
