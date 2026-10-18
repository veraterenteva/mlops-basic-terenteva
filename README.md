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

## ДЗ2: Gitflow

Подробный сценарий с командами и чек-лист — в [`hw2/README.md`](hw2/README.md).

### План работы по Gitflow

1. `develop` — постоянная ветка интеграции, создаётся от `main`.
2. `feature/*` — по одной ветке на изменение, создаются от `develop`,
   вливаются обратно в `develop` только через Pull Request.
3. `release/*` — подготовка релиза от `develop`, вливается в `main` и `develop`,
   на `main` ставится аннотированный тег `vX.Y.Z`.
4. `hotfix/*` — срочные исправления от `main`, вливаются в `main` и `develop`
   с новым тегом.

### Что сделано в этом репозитории

Ветки созданы и запушены в `origin`, в каждой есть коммит:

| Ветка | Коммит | Назначение |
| --- | --- | --- |
| `main` | `81e9846` | стабильная ветка |
| `develop` | `chore: init develop branch for gitflow` | интеграция изменений |
| `feature/add-rms` | `feat: add rms function` | функциональность |
| `feature/add-mode` | `feat: add mode function` | функциональность |
| `release/0.2.0` | `chore(release): bump version to 0.2.0` | подготовка релиза |
| `hotfix/0.2.1` | `fix: handle empty input` | срочное исправление |

Открытые Pull Request'ы:

| PR | Из | В | Ссылка |
| --- | --- | --- | --- |
| #1 | `feature/add-rms` | `develop` | [#1](https://github.com/veraterenteva/mlops-basic-terenteva/pull/1) |
| #2 | `feature/add-mode` | `develop` | [#2](https://github.com/veraterenteva/mlops-basic-terenteva/pull/2) |
| #3 | `release/0.2.0` | `main` | [#3](https://github.com/veraterenteva/mlops-basic-terenteva/pull/3) |
| #4 | `hotfix/0.2.1` | `main` | [#4](https://github.com/veraterenteva/mlops-basic-terenteva/pull/4) |

> Ветки и коммиты в них созданы намеренно минимальными: цель задания —
> отработать сам процесс ветвления и Pull Request, а не добавить новую логику.
