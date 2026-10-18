# mlops-basic-terenteva

Домашние задания по дисциплине **MLOps**.

## Структура репозитория

```text
.
├── README.md
├── pyproject.toml
├── hw1/                      # ДЗ1: uv + Ruff + ty
│   ├── .python-version
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── src/mlops_hw1/
│   └── tests/test_stats.py
└── hw2/                      # ДЗ2: ветки и Pull Request (Gitflow)
    └── README.md
```

## Домашние задания

| № | Задание              | Описание | Готово |
| --- |----------------------| --- | --- |
| 1 | Python Project Setup | Инициализация проекта с `uv`, Ruff и ty. | ☑ |
| 2 | Git Branching        | Работа с ветками и Pull Request по Gitflow. | ☑ |
| 3 | Не выдана            | Не выдана | ☐ |

## ДЗ1: быстрый старт

`hw1` — самостоятельный uv-проект:

```bash
cd hw1
uv sync
uv run python -m mlops_hw1
```

Пакет `mlops_hw1` содержит две функции: `rms` (среднеквадратичное значение)
и `mode` (мода).

## ДЗ1: проверки

Из папки `hw1`:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Либо из корня репозитория (корневой `pyproject.toml` — агрегатор проверок для `hw1`):

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## ДЗ2: схема ветвления

Подробный сценарий с командами и чек-лист — в [`hw2/README.md`](hw2/README.md).
Кратко: `main` → `develop` → `feature/*` (PR) → `release/*` → тег `vX.Y.Z`,
плюс `hotfix/*` из `main`.
