"""Unit tests for the ridge regression model.

Block under test
----------------
``mlops.model.RidgeRegression`` — ядро пайплайна. Реализован вручную
через нормальные уравнения ``(X^T X + alpha * I) w = X^T y`` и решатель
из ``mlops.blocks``. Здесь проверяется и математика, и контракт API.

Requirements
------------
1. Модель восстанавливает истинные коэффициенты на чистых данных.
2. На данных с шумом ошибка остаётся в разумных пределах.
3. ``predict`` и ``predict_one`` согласованы между собой.
4. До ``fit`` модель непригодна к использованию и сообщает об этом.
5. Некорректные размеры и отрицательный ``alpha`` отвергаются.
6. Сериализация в JSON сохраняет веса и свободный член.
"""

import pytest

from mlops.generate import TRUE_WEIGHTS
from mlops.model import RidgeRegression, load_model, save_model

# --- Блок: обучение --------------------------------------------------------
# Требование: на данных без шума модель обязана найти истинные веса.
# При alpha = 0 это точное решение МНК; при заметной alpha веса слегка
# смещаются к нулю — это ожидаемая регуляризация, а не ошибка.


@pytest.mark.unit
@pytest.mark.parametrize(
    ("alpha", "tolerance"),
    [
        # Без штрафа замкнутая формула воспроизводит истину точно.
        (0.0, 1e-6),
        (1e-9, 1e-6),
        # Заметный штраф сжимает веса, поэтому допуск шире.
        (1e-3, 5e-3),
    ],
)
def test_model_recovers_true_weights(alpha: float, tolerance: float) -> None:
    features = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    targets = [
        sum(weight * value for weight, value in zip(TRUE_WEIGHTS, row, strict=True))
        for row in features
    ]
    model = RidgeRegression(alpha=alpha).fit(features, targets)
    assert model.weights is not None
    assert model.weights == pytest.approx(list(TRUE_WEIGHTS), abs=tolerance)


@pytest.mark.unit
def test_model_fits_noisy_data(trained_model: RidgeRegression, splits) -> None:
    # Требование: на train-части с шумом 0.5 ошибка каждой точки < 1.0.
    # Порог не «магический»: он отражает шум генератора.
    predictions = trained_model.predict(splits.train.features)
    expected = splits.train.targets
    errors = [abs(a - b) for a, b in zip(expected, predictions, strict=True)]
    assert max(errors) < 1.0


# --- Блок: предсказание ----------------------------------------------------
# Требование: батчевый и одиночный интерфейсы дают одно и то же.


@pytest.mark.unit
def test_predict_one_matches_predict(trained_model: RidgeRegression, splits) -> None:
    batch = trained_model.predict(splits.test.features)
    single = [trained_model.predict_one(row) for row in splits.test.features]
    assert batch == pytest.approx(single)


# --- Контракт и валидация --------------------------------------------------
# Требование: необученная модель и неверные размеры дают понятные ошибки,
# а не тихий мусор в предсказаниях.


@pytest.mark.unit
def test_predict_without_fit_raises() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        RidgeRegression().predict_one([1.0])


@pytest.mark.unit
def test_predict_rejects_wrong_feature_count(trained_model: RidgeRegression) -> None:
    # Модель обучена на 3 признаках, вектор из одного — ошибка.
    with pytest.raises(ValueError, match="does not match the model"):
        trained_model.predict_one([1.0])


@pytest.mark.unit
def test_fit_rejects_wrong_target_count() -> None:
    with pytest.raises(ValueError, match="same length"):
        RidgeRegression().fit([[1.0]], [1.0, 2.0])


@pytest.mark.unit
def test_fit_rejects_negative_alpha() -> None:
    # Отрицательный штраф сделал бы матрицу не положительно определённой.
    with pytest.raises(ValueError, match="non-negative"):
        RidgeRegression(alpha=-1.0).fit([[1.0]], [1.0])


@pytest.mark.unit
def test_fit_rejects_empty_features() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        RidgeRegression().fit([], [])


@pytest.mark.unit
def test_fit_rejects_empty_feature_vector() -> None:
    with pytest.raises(ValueError, match="feature vectors must not be empty"):
        RidgeRegression().fit([[]], [1.0])


@pytest.mark.unit
def test_fit_rejects_ragged_features() -> None:
    # «Рваные» строки разной длины не должны проходить в матрицу Грама.
    with pytest.raises(ValueError, match="same length"):
        RidgeRegression().fit([[1.0, 2.0], [1.0]], [1.0, 2.0])


# --- Блок: сериализация ----------------------------------------------------
# Требование: модель переживает сохранение в JSON и чтение обратно,
# иначе стадии train и evaluate не смогут обменяться артефактом.


@pytest.mark.unit
def test_model_roundtrip(tmp_path, trained_model: RidgeRegression) -> None:
    path = tmp_path / "model.json"
    save_model(path, trained_model)
    restored = load_model(path)
    assert restored.weights == pytest.approx(trained_model.weights)
    assert restored.intercept == pytest.approx(trained_model.intercept)


@pytest.mark.unit
def test_load_model_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.json")


@pytest.mark.unit
def test_from_dict_rejects_bad_weights() -> None:
    # Требование: битый артефакт не должен давать «модель без весов».
    with pytest.raises(ValueError, match="weights must be a list"):
        RidgeRegression.from_dict({"alpha": 1.0, "intercept": 0.0, "weights": "bad"})


@pytest.mark.unit
def test_to_dict_requires_fit() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        RidgeRegression().to_dict()
