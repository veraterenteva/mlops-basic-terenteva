# HW1: uv + Ruff + ty

Минимальный Python-проект для дисциплины MLOps: менеджер проектов
[uv](https://docs.astral.sh/uv/), линтер и форматтер
[Ruff](https://docs.astral.sh/ruff/), проверка типов
[ty](https://docs.astral.sh/ty/), тесты на pytest.

## Структура

```text
hw1/
├── .python-version
├── pyproject.toml
├── uv.lock
├── src/
│   └── mlops_hw1/
│       ├── __init__.py
│       └── py.typed
└── tests/
    └── test_mean.py
```

## Запуск

```bash
uv sync
uv run python -m mlops_hw1
```

## Проверки

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## Результат

- `pytest` — 3 теста проходят;
- `ruff check` / `ruff format --check` — без замечаний;
- `ty check` — без ошибок типов.
