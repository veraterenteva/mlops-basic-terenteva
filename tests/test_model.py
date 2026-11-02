"""Unit tests for the ridge regression model.

Block under test: ``mlops.model.RidgeRegression``

This is the core of the pipeline. It is implemented by hand with the
normal equations ``(X^T X + alpha * I) w = X^T y`` solved by the
``mlops.blocks`` solver, so both the mathematics and the API contract are
verified here.

What is verified

R1. The model recovers the true coefficients on noise-free data.
R2. On noisy data the prediction error stays within a sane bound.
R3. ``predict`` and ``predict_one`` agree with each other.
R4. The model is unusable before ``fit`` and says so explicitly.
R5. Wrong shapes and a negative ``alpha`` are rejected.
R6. JSON serialisation preserves the weights and the intercept.
"""

import pytest

from mlops.generate import TRUE_WEIGHTS
from mlops.model import RidgeRegression, load_model, save_model

# Training
# R1 and R2. On noise-free data the model must find the generating
# coefficients. With alpha = 0 this is the exact least-squares solution;
# with a noticeable alpha the weights shrink slightly towards zero, which is
# the intended regularisation rather than an error.


@pytest.mark.unit
@pytest.mark.parametrize(
    ("alpha", "tolerance"),
    [
        # No penalty: the closed-form solution reproduces the truth exactly.
        (0.0, 1e-6),
        # A negligible penalty keeps the same accuracy.
        (1e-9, 1e-6),
        # A visible penalty shrinks the weights, so the tolerance is wider.
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
    # R2: the generator adds noise with sigma = 0.5, so a per-point error
    # below 1.0 means the model captured the signal and not the noise.
    predictions = trained_model.predict(splits.train.features)
    expected = splits.train.targets
    errors = [abs(a - b) for a, b in zip(expected, predictions, strict=True)]
    assert max(errors) < 1.0


# Prediction
# R3: the batch and the single-row interfaces must produce identical values.


@pytest.mark.unit
def test_predict_one_matches_predict(trained_model: RidgeRegression, splits) -> None:
    batch = trained_model.predict(splits.test.features)
    single = [trained_model.predict_one(row) for row in splits.test.features]
    assert batch == pytest.approx(single)


# Contract and validation
# R4 and R5: an unfitted model and wrong shapes must raise clear errors
# instead of returning garbage predictions.


@pytest.mark.unit
def test_predict_without_fit_raises() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        RidgeRegression().predict_one([1.0])


@pytest.mark.unit
def test_predict_rejects_wrong_feature_count(trained_model: RidgeRegression) -> None:
    # The model was fitted on three features; a one-element vector is wrong.
    with pytest.raises(ValueError, match="does not match the model"):
        trained_model.predict_one([1.0])


@pytest.mark.unit
def test_fit_rejects_wrong_target_count() -> None:
    with pytest.raises(ValueError, match="same length"):
        RidgeRegression().fit([[1.0]], [1.0, 2.0])


@pytest.mark.unit
def test_fit_rejects_negative_alpha() -> None:
    # A negative penalty would make the Gram matrix non positive-definite.
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
    # Rows of different length cannot be assembled into a Gram matrix.
    with pytest.raises(ValueError, match="same length"):
        RidgeRegression().fit([[1.0, 2.0], [1.0]], [1.0, 2.0])


# Serialisation
# R6: the train and evaluate stages exchange the model through a JSON file,
# so a lossy round trip would silently change the reported metrics.


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
    # R6: a corrupted artefact must not become a model without weights.
    with pytest.raises(ValueError, match="weights must be a list"):
        RidgeRegression.from_dict({"alpha": 1.0, "intercept": 0.0, "weights": "bad"})


@pytest.mark.unit
def test_to_dict_requires_fit() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        RidgeRegression().to_dict()
