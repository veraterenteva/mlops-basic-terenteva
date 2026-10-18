# mlops-basic-terenteva
### Студентка - Терентьева Вера Васильевна
Домашние задания по дисциплине **MLOps**.

## Структура репозитория

Единственный uv-проект — `hw1/` (там же `pyproject.toml` и `uv.lock`).
Корневых `pyproject.toml`/`uv.lock` нет намеренно, чтобы не держать два lock-файла.

```text
.
├── README.md
├── hw1/                      # ДЗ1: uv + Ruff + ty (uv-проект)
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

Все команды выполняются из папки `hw1` — это единственный uv-проект в репозитории:

```bash
cd hw1
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Результат: `7 passed`, `All checks passed!`, `6 files already formatted`,
`All checks passed!` (ty).

## ДЗ2: схема ветвления

Подробный сценарий с командами и чек-лист — в [`hw2/README.md`](hw2/README.md).
Кратко: `main` → `develop` → `feature/*` (PR) → `release/*` → тег `vX.Y.Z`,
плюс `hotfix/*` из `main`.
