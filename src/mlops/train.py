"""Training stage: fit the model on the prepared train split."""

from pathlib import Path

from mlops.data import load_csv
from mlops.model import RidgeRegression, save_model


def train_model(train_path: Path, model_path: Path, alpha: float) -> RidgeRegression:
    """Load the train split, fit the model and persist it."""
    dataset = load_csv(train_path)
    model = RidgeRegression(alpha=alpha)
    model.fit(dataset.features, dataset.targets)
    save_model(model_path, model)
    return model
