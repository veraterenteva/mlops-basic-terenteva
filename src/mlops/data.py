"""Data loading, cleaning, splitting and feature scaling."""

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    """A tabular dataset: feature rows, targets and column names."""

    features: list[list[float]]
    targets: list[float]
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.features)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def load_csv(path: Path, target_column: str = "target") -> Dataset:
    """Read a CSV file produced by the generator stage."""
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError("dataset must contain a header and at least one row")

    header = rows[0]
    if target_column not in header:
        raise ValueError(f"target column not found: {target_column}")

    target_index = header.index(target_column)
    feature_indices = [index for index in range(len(header)) if index != target_index]
    feature_names = [header[index] for index in feature_indices]

    features: list[list[float]] = []
    targets: list[float] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(
                f"row {line_number} has {len(row)} values, expected {len(header)}"
            )
        features.append([float(row[index]) for index in feature_indices])
        targets.append(float(row[target_index]))

    return Dataset(features=features, targets=targets, feature_names=feature_names)


def clean(dataset: Dataset) -> Dataset:
    """Drop rows with missing or non-finite values and de-duplicate."""
    if not dataset.features:
        raise ValueError("dataset must not be empty")

    clean_features: list[list[float]] = []
    clean_targets: list[float] = []
    seen: set[tuple[float, ...]] = set()

    for features, target in zip(dataset.features, dataset.targets, strict=True):
        key = (*features, target)
        if key in seen:
            continue
        if any(value != value for value in key):  # NaN check without math.isnan
            continue
        seen.add(key)
        clean_features.append(list(features))
        clean_targets.append(target)

    if not clean_features:
        raise ValueError("no rows left after cleaning")

    return Dataset(
        features=clean_features,
        targets=clean_targets,
        feature_names=list(dataset.feature_names),
    )


@dataclass(frozen=True)
class Splits:
    """Train and test parts of a dataset."""

    train: Dataset
    test: Dataset


def train_test_split(dataset: Dataset, test_size: float, seed: int) -> Splits:
    """Shuffle the dataset and split it into train and test parts."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if len(dataset) < 2:
        raise ValueError("dataset must contain at least two rows")

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    test_count = max(1, int(round(len(indices) * test_size)))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    if not train_indices:
        raise ValueError("train part is empty, decrease test_size")

    def take(chosen: list[int]) -> Dataset:
        return Dataset(
            features=[list(dataset.features[index]) for index in chosen],
            targets=[dataset.targets[index] for index in chosen],
            feature_names=list(dataset.feature_names),
        )

    return Splits(train=take(train_indices), test=take(test_indices))


@dataclass(frozen=True)
class StandardScaler:
    """Zero-mean, unit-variance feature scaling."""

    means: list[float]
    stds: list[float]

    @classmethod
    def fit(cls, dataset: Dataset) -> "StandardScaler":
        """Compute per-feature mean and standard deviation."""
        if not dataset.features:
            raise ValueError("dataset must not be empty")

        n_features = dataset.n_features
        columns = [
            [row[column] for row in dataset.features] for column in range(n_features)
        ]
        means = [sum(column) / len(column) for column in columns]
        stds: list[float] = []
        for column, mu in zip(columns, means, strict=True):
            variance = sum((value - mu) ** 2 for value in column) / len(column)
            stds.append(variance**0.5 if variance > 0 else 1.0)
        return cls(means=means, stds=stds)

    def transform(self, dataset: Dataset) -> Dataset:
        """Apply the stored statistics to a dataset."""
        if dataset.n_features != len(self.means):
            raise ValueError("dataset feature count does not match the scaler")

        features = [
            [
                (value - mean) / std
                for value, mean, std in zip(row, self.means, self.stds, strict=True)
            ]
            for row in dataset.features
        ]
        return Dataset(
            features=features,
            targets=list(dataset.targets),
            feature_names=list(dataset.feature_names),
        )

    def to_dict(self) -> dict[str, list[float]]:
        """Serialise the scaler to a JSON-compatible mapping."""
        return {"means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "StandardScaler":
        """Restore a scaler from its JSON representation."""
        means = [float(value) for value in payload["means"]]
        stds = [float(value) for value in payload["stds"]]
        if len(means) != len(stds):
            raise ValueError("means and stds must have the same length")
        return cls(means=means, stds=stds)


def write_dataset(path: Path, dataset: Dataset) -> None:
    """Write a dataset back to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*dataset.feature_names, "target"])
        for features, target in zip(dataset.features, dataset.targets, strict=True):
            writer.writerow([*features, target])


def save_scaler(path: Path, scaler: StandardScaler) -> None:
    """Persist a scaler as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")


def load_scaler(path: Path) -> StandardScaler:
    """Load a scaler from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"scaler not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return StandardScaler.from_dict(payload)


def prepare_dataset(
    raw_path: Path, params_seed: int, test_size: float, data_dir: Path
) -> tuple[Path, Path, Path]:
    """Clean the raw dataset, split it and persist train, test and scaler files."""
    dataset = clean(load_csv(raw_path))
    splits = train_test_split(dataset, test_size=test_size, seed=params_seed)
    scaler = StandardScaler.fit(splits.train)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    scaler_path = data_dir / "scaler.json"

    write_dataset(train_path, scaler.transform(splits.train))
    write_dataset(test_path, scaler.transform(splits.test))
    save_scaler(scaler_path, scaler)

    return train_path, test_path, scaler_path
