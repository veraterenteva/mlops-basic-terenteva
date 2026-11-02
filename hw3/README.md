# HW3: DVC и версионирование пайплайна

## Что требовалось

1. Установить DVC.
2. Настроить remote для хранения данных.
3. Создать пайплайн и данные (минимальные, но рабочие).

## Как запускать

Все команды выполняются из корня репозитория.

```bash
# 1. Установить зависимости, включая DVC
uv sync

# 2. Прогнать пайплайн целиком
uv run dvc repro

# 3. Посмотреть метрики
type data\metrics.json
```

Отдельные стадии:

```bash
uv run python -m mlops.cli generate
uv run python -m mlops.cli prepare
uv run python -m mlops.cli train
uv run python -m mlops.cli evaluate
```

Проверить состояние пайплайна:

```bash
uv run dvc status
```

## Что сделано

### Пайплайн

Код написан на стандартной библиотеке, без numpy и pandas — так его можно
полноценно покрыть тестами в ДЗ4.

| Модуль | Роль | Что внутри |
| --- | --- | --- |
| `src/mlops/blocks.py` | числовые примитивы | `dot`, `mean`, `variance`, `standard_deviation`, `solve_linear_system` (Гаусс с выбором главного элемента) |
| `src/mlops/config.py` | параметры и пути | `Params`, `DEFAULT_PARAMS`, `Paths`, `load_params` |
| `src/mlops/generate.py` | генерация данных | синтетический CSV по `seed`, `n_samples`, `n_features`, `noise` |
| `src/mlops/data.py` | подготовка данных | `load_csv`, `clean`, `train_test_split`, `StandardScaler`, `prepare_dataset` |
| `src/mlops/model.py` | модель | `RidgeRegression` на нормальных уравнениях: `fit`, `predict`, сохранение в JSON |
| `src/mlops/metrics.py` | метрики | `mse`, `rmse`, `mae`, `r2`, `all_metrics` |
| `src/mlops/train.py` | обучение | `train_model` |
| `src/mlops/evaluate.py` | оценка | `evaluate_model` |
| `src/mlops/cli.py` | CLI | подкоманды `generate`, `prepare`, `train`, `evaluate`, `all` |

### Стадии DVC

`dvc.yaml` описывает четыре стадии со связями `deps` → `outs`:

| Стадия | Читает | Пишет |
| --- | --- | --- |
| `generate` | `params.yaml`, код генератора | `data/raw.csv` |
| `prepare` | `data/raw.csv`, код подготовки | `data/train.csv`, `data/test.csv`, `data/scaler.json` |
| `train` | `data/train.csv`, код модели | `data/model.json` |
| `evaluate` | `data/test.csv`, `data/model.json`, код метрик | `data/metrics.json` |

Параметры в `params.yaml`:

| Параметр | Значение | Смысл |
| --- | --- | --- |
| `seed` | 42 | воспроизводимость генерации и сплита |
| `n_samples` | 200 | размер датасета |
| `n_features` | 3 | число признаков |
| `noise` | 0.5 | шум в целевой переменной |
| `test_size` | 0.2 | доля тестовой выборки |
| `ridge_alpha` | 1e-06 | коэффициент регуляризации |
| `target_column` | target | имя целевой колонки |

### Результат прогона

```text
uv run dvc repro
Running stage 'generate': ...
Running stage 'prepare': ...
Running stage 'train': ...
Running stage 'evaluate':
mse: 0.297656
rmse: 0.545579
mae: 0.428315
r2: 0.994599
```

Качество высокое ожидаемо: данные сгенерированы линейной зависимостью
`y = 2.5*f0 - 1.5*f1 + 0.75*f2 + шум`, поэтому линейная модель восстанавливает
её почти точно.

### DVC remote

Используется **локальный** remote вместо S3 — обычная папка на диске:

```bash
uv run dvc remote add -d localstorage C:/Users/vterenteva/mlops-dvc-storage
uv run dvc push
```

Результат:

```text
Setting 'localstorage' as a default remote.
5 files pushed
```

#### Почему сделали локальный remote и зачем он вне репозитория

**Remote** — второе хранилище, куда DVC копирует содержимое своего кэша,
чтобы данные можно было перенести на другую машину. Обычно это S3, GCS или
SSH-сервер. **Локальный remote** — тот же механизм, но хранилищем служит
обычная папка на диске.

Цикл работы:

```text
dvc repro  ->  артефакты попадают в .dvc/cache
dvc push   ->  копии уезжают в mlops-dvc-storage
dvc pull   ->  данные возвращаются обратно
```

Папка должна быть **вне репозитория**, потому что:

- внутри проекта это был бы дубль `.dvc/cache`, и кэш попал бы в git;
- `dvc pull` перестал бы что-либо проверять: второе хранилище лежало бы
  в том же репозитории, что и первое.

Ограничение: такая папка существует только на одной машине. Это имитация
S3, а не настоящее общее хранилище. 
В целом, для переносимости достаточно заменить одну команду:

```bash
uv run dvc remote add -d storage s3://bucket/path
uv run dvc push
```

#### Как пересоздать remote

```bash
mkdir C:\Users\<user>\mlops-dvc-storage
uv run dvc remote add -d localstorage C:/Users/<user>/mlops-dvc-storage
uv run dvc push
```

### Версионирование

- в git лежат `dvc.yaml`, `dvc.lock`, `params.yaml`, `data/.gitignore`
  и файлы-указатели `*.dvc`;
- сами данные и модель в git не попадают, они находятся в кэше DVC и в remote;
- `.dvcignore` исключает из пайплайна `hw1/`, `hw2/`, `hw4/`, `tests/`
  и кэши инструментов, чтобы DVC не считал их входными данными.

## Проверка результата

```bash
uv run dvc status
uv run dvc repro
uv run dvc metrics show
```
