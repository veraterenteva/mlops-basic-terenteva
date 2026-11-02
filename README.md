# mlops-basic-terenteva

### Студентка — Терентьева Вера Васильевна

Домашние задания по дисциплине **MLOps**.

## Домашние задания

| № | Задание | Описание | Документация | Готово |
| --- | --- | --- | --- | --- |
| 1 | Python Project Setup | Инициализация проекта с `uv`, Ruff и ty, тесты на pytest | [`hw1/README.md`](hw1/README.md) | ☑ |
| 2 | Git Branching | Ветки и Pull Request по Gitflow | [`hw2/README.md`](hw2/README.md) | ☑ |
| 3 | DVC | Установка DVC, remote, пайплайн и данные | [`hw3/README.md`](hw3/README.md) | ☑ |
| 4 | Testing | Тесты пайплайна от юнит до e2e, покрытие в pytest | [`hw4/README.md`](hw4/README.md) | ☑ |

## Структура репозитория

```text
.
├── README.md
├── pyproject.toml              # корневой uv-проект (ДЗ3, ДЗ4)
├── uv.lock
├── .python-version
├── params.yaml                 # параметры пайплайна (DVC)
├── dvc.yaml                    # описание стадий
├── dvc.lock
├── .dvc/                       # кэш и конфигурация DVC
├── src/
│   └── mlops/                  # пайплайн: blocks, config, data, model, metrics,
│                               #           generate, train, evaluate, cli
├── tests/                      # тесты ДЗ4 (unit, integration, e2e)
│   ├── conftest.py
│   ├── test_blocks.py
│   ├── test_metrics.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_config.py
│   ├── test_generate.py
│   ├── test_stages.py
│   └── test_e2e.py
├── data/                       # артефакты пайплайна (под DVC)
├── hw1/                        # ДЗ1 — отдельный uv-проект
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── src/mlops_hw1/
│   └── tests/test_stats.py
├── hw2/README.md               # ДЗ2 — описание веток и PR
├── hw3/README.md               # ДЗ3 — DVC
└── hw4/README.md               # ДЗ4 — тесты
```

## Два uv-проекта: что и откуда запускать

В репозитории два независимых uv-проекта. Они не конфликтуют: `uv` всегда
использует **ближайший** `pyproject.toml` вверх по дереву.

| Где запускать | Какой проект | Что запускается |
| --- | --- | --- |
| `cd hw1` | `hw1/pyproject.toml`, `hw1/uv.lock`, `hw1/.venv` | ДЗ1: `uv sync`, `uv run pytest`, `uv run python -m mlops_hw1`, `uv run ruff check .`, `uv run ty check` |
| корень | корневой `pyproject.toml`, `uv.lock`, `.venv` | ДЗ3 и ДЗ4: `uv sync`, `uv run dvc repro`, `uv run pytest`, `uv run ruff check .`, `uv run ty check` |

Корневой `pytest` собирает только `tests/` (`testpaths = ["tests"]`),
поэтому тесты ДЗ1 из корня не запускаются — это сделано намеренно.

## ДЗ1: быстрый старт

```bash
cd hw1
uv sync
uv run python -m mlops_hw1
uv run pytest
```

Подробности — в [`hw1/README.md`](hw1/README.md).

## ДЗ2: что сделано с ветками

Подробное описание с SHA коммитов и списком открытых PR —
в [`hw2/README.md`](hw2/README.md). Кратко:

| Ветка | SHA | Коммит |
| --- | --- | --- |
| `main` | `81e9846` | оставлен один uv-проект в hw1 |
| `develop` | `0ddc16b` | docs: link created pull requests |
| `feature/add-rms` | `f56886e` | feat: add rms function |
| `feature/add-mode` | `5247218` | feat: add mode function |
| `release/0.2.0` | `5920942` | Merge branch 'develop' into release/0.2.0 |
| `hotfix/0.2.1` | `4c977e1` | fix: handle empty input |

Открыты Pull Request'ы
[#1](https://github.com/veraterenteva/mlops-basic-terenteva/pull/1),
[#2](https://github.com/veraterenteva/mlops-basic-terenteva/pull/2),
[#3](https://github.com/veraterenteva/mlops-basic-terenteva/pull/3),
[#4](https://github.com/veraterenteva/mlops-basic-terenteva/pull/4).

## ДЗ3: пайплайн и DVC

```bash
uv sync
uv run dvc repro
```

Стадии: `generate → prepare → train → evaluate`. Результат прогона:

```text
mse: 0.297656
rmse: 0.545579
mae: 0.428315
r2: 0.994599
```

Данные версионируются через DVC, remote — локальная папка вне репозитория.
Подробности и объяснение, что такое локальный remote, —
в [`hw3/README.md`](hw3/README.md).

## ДЗ4: тесты и покрытие

```bash
uv run pytest
```

```text
105 passed
TOTAL  440 stmts  5 miss  99%
Required test coverage of 85% reached. Total coverage: 98.86%
```

Тесты трёх уровней — `unit`, `integration`, `e2e`; используются фикстуры,
параметризация, `pytest.raises`, `pytest.approx`, `tmp_path` и фабрика фикстур.
Подробности — в [`hw4/README.md`](hw4/README.md).

## Полная проверка репозитория

```bash
# ДЗ3 и ДЗ4
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run dvc repro

# ДЗ1 (отдельный проект)
cd hw1
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```
