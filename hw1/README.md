# HW1: Python Project Setup (uv + Ruff + ty)

## Что требовалось

Создать минимальный Python-проект с современным инструментарием:

1. инициализация проекта через [uv](https://docs.astral.sh/uv/) — файлы
   `pyproject.toml`, `uv.lock`, `.python-version`;
2. линтер и форматтер [Ruff](https://docs.astral.sh/ruff/);
3. проверка типов [ty](https://docs.astral.sh/ty/);
4. тесты на `pytest` хотя бы для одной функции;
5. README с командами запуска и проверок.

## Как запускать

`hw1/` — самостоятельный uv-проект. Все команды выполняются из этой папки:

```bash
cd hw1
uv sync
uv run python -m mlops_hw1
```

Ожидаемый вывод:

```text
RMS: 3.696845502136472
Mode: 4.0
```

Проверки:

```bash
cd hw1
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## Что сделано

### Структура

```text
hw1/
├── .python-version
├── pyproject.toml
├── uv.lock
├── src/
│   └── mlops_hw1/
│       ├── __init__.py
│       ├── __main__.py
│       └── py.typed
└── tests/
    └── test_stats.py
```

### Функции пакета

| Функция | Описание |
| --- | --- |
| `rms(values)` | среднеквадратичное значение (root mean square) |
| `mode(values)` | мода; при равенстве частот возвращается меньшее значение |

Обе функции выбрасывают `ValueError` на пустом списке. Модуль `__main__.py`
позволяет запускать пакет командой `uv run python -m mlops_hw1`.

### Тесты (7 штук)

| Тест | Что проверяет |
| --- | --- |
| `test_rms` | `rms([3, 4]) == sqrt(12.5)` |
| `test_rms_single_value` | `rms([5]) == 5` |
| `test_rms_empty` | `ValueError` на пустом списке |
| `test_mode` | `mode([1, 2, 2, 3]) == 2` |
| `test_mode_tie_returns_smallest` | при равенстве частот берётся меньшее |
| `test_mode_single_value` | `mode([4]) == 4` |
| `test_mode_empty` | `ValueError` на пустом списке |

### Результаты проверок

```text
uv run pytest               -> 7 passed
uv run ruff check .         -> All checks passed!
uv run ruff format --check  -> 4 files already formatted
uv run ty check             -> All checks passed!
```

## Про два uv-проекта в репозитории

В репозитории два независимых uv-проекта: этот (`hw1/`) и корневой
(для ДЗ3 и ДЗ4 с пайплайном и DVC). Они не конфликтуют, потому что `uv`
всегда поднимается до **ближайшего** `pyproject.toml`:

- команда, запущенная из `hw1/`, использует `hw1/pyproject.toml`, `hw1/uv.lock`
  и создаёт `hw1/.venv`;
- команда, запущенная из корня, использует корневые `pyproject.toml`,
  `uv.lock` и `.venv`.

Корневой `pytest` собирает только `tests/` (параметр `testpaths`), поэтому
тесты этой работы из корня не запускаются — это сделано намеренно, чтобы
результаты ДЗ1 и ДЗ3/ДЗ4 не смешивались.
