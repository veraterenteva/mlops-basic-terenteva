# HW4: Тесты для пайплайна с измерением покрытия

## Что требовалось

1. Написать тесты для всего пайплайна: блоки, модель, даталоадер, тренировка.
2. Реализовать код так, чтобы функции были полноценными реализациями и было что проверить.
3. Прогнать тесты на всех уровнях.

## Как запускать

```bash
# из корня репозитория
uv sync
uv run pytest
```

По умолчанию `pytest` сам показывает покрытие (`--cov=mlops --cov-report=term-missing`)
и падает, если покрытие ниже 85% (`--cov-fail-under=85`).

Прогон отдельных этапов:

```bash
# только юнит-тесты
uv run pytest -m unit

# только интеграционные
uv run pytest -m integration

# только e2e
uv run pytest -m e2e

# без порога покрытия, чтобы посмотреть цифры
uv run pytest --no-cov

# HTML-отчёт покрытия
uv run pytest --cov-report=html
```

## Что сделано. Краткое описание покрытия тестами

### Уровни тестов

| Файл | Уровень | Что проверяет |
| --- | --- | --- |
| `tests/test_blocks.py` | unit | `dot`, `mean`, `variance`, `standard_deviation`, `solve_linear_system` |
| `tests/test_metrics.py` | unit | `mse`, `rmse`, `mae`, `r2`, `all_metrics` |
| `tests/test_data.py` | unit | `load_csv`, `clean`, `train_test_split`, `StandardScaler`, сериализация |
| `tests/test_model.py` | unit | `RidgeRegression.fit/predict`, восстановление истинных весов, JSON roundtrip |
| `tests/test_config.py` | unit | `Paths`, `load_params`, валидация `params.yaml` |
| `tests/test_generate.py` | unit + integration | генерация строк, `write_csv`, `generate_dataset` |
| `tests/test_stages.py` | integration | каждая стадия и полный прогон `run_all` |
| `tests/test_e2e.py` | e2e | полный прогон через CLI и все артефакты на диске |

### Использованные возможности pytest

Продемонстрировали различные фишки и возможности pytest. 

**Фикстуры** — `tests/conftest.py`:

| Фикстура | Что даёт |
| --- | --- |
| `params` | набор параметров с уменьшенным `n_samples` для быстрых тестов |
| `params_file` | `params.yaml` на диске |
| `raw_csv` | реальный сгенерированный CSV |
| `dataset` | загруженный `Dataset` |
| `clean_dataset` | очищенный датасет |
| `splits` | train/test разбиение |
| `scaled_splits` | масштабированные сплиты вместе со скейлером |
| `trained_model` | обученная модель |
| `model_file` | модель, сохранённая на диск |
| `pipeline_root` | готовый рабочий каталог пайплайна с `params.yaml` |
| `csv_writer` | фабрика для записи произвольных CSV в негативных тестах |

**Параметризация** — `@pytest.mark.parametrize` используется в 15+ тестах:
граничные значения `test_size`, разные наборы векторов для `dot`, разные
варианты `alpha` для модели, разные `n_samples` для генератора и т.д.

**Проверка исключений** — `pytest.raises` с `match=` для 30+ негативных сценариев.

**Приближённые сравнения** — `pytest.approx` для всех вычислений с плавающей точкой.

**Маркеры** — `unit`, `integration`, `e2e` объявлены в `pyproject.toml`
и позволяют запускать нужный уровень отдельно.

**Временные каталоги** — `tmp_path` во всех тестах, которые пишут файлы;
тесты не зависят от состояния рабочей папки.

**Фабрика фикстур** — `csv_writer` возвращает функцию, что позволяет
конструировать произвольные входные данные прямо в тесте.

### Покрытие

Фактический результат прогона:

```text
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
src\mlops\__init__.py       2      0   100%
src\mlops\__main__.py       1      1     0%   3
src\mlops\blocks.py        39      0   100%
src\mlops\cli.py           67      3    96%   68, 115-116
src\mlops\config.py        52      0   100%
src\mlops\data.py         130      1    99%   108
src\mlops\evaluate.py      13      0   100%
src\mlops\generate.py      36      0   100%
src\mlops\metrics.py       24      0   100%
src\mlops\model.py         67      0   100%
src\mlops\train.py          9      0   100%
-----------------------------------------------------
TOTAL                     440      5    99%
Required test coverage of 85% reached. Total coverage: 98.86%

105 passed in 1.88s
```

**Итог: 105 тестов, покрытие 98.86%.**

## Полный прогон проверок

```bash
uv sync
uv run pytest              # 105 passed, покрытие 98.86%
uv run ruff check .        # All checks passed!
uv run ruff format --check . # 26 files already formatted
uv run ty check            # All checks passed!
uv run dvc repro           # пайплайн воспроизводится
```
