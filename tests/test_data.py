"""Unit tests for the data block: loading, cleaning, splitting, scaling.

Block under test: ``mlops.data``

This layer sits between the generator and the model. It reads the CSV,
removes garbage, splits the dataset and standardises the features. A
mistake here is invisible at runtime but corrupts both training and
evaluation, which is why it gets the most detailed coverage.

What is verified

R1. ``load_csv`` parses the header, separates the target column and reports
    every malformed file shape (no file, no target column, a short row,
    a header-only file).
R2. ``clean`` drops duplicates and NaN rows but refuses to return an empty
    dataset, because an empty result would hide the real problem.
R3. ``train_test_split`` is deterministic for a fixed seed, changes for a
    different seed, preserves the total row count and computes the test
    size as round(n * test_size) with a minimum of one row.
R4. ``StandardScaler`` produces zero mean and unit variance, survives a
    constant column without dividing by zero, rejects a mismatched feature
    count and round-trips through JSON without losing its statistics.
"""

import pytest

from mlops.data import (
    Dataset,
    StandardScaler,
    clean,
    load_csv,
    load_scaler,
    save_scaler,
    train_test_split,
    write_dataset,
)

# Loading CSV
# R1: parse the file produced by the generate stage.


@pytest.mark.unit
def test_load_csv_reads_generated_file(dataset: Dataset) -> None:
    # The fixture dataset holds 40 rows with three features named f0..f2,
    # taken from the CSV header.
    assert len(dataset) == 40
    assert dataset.n_features == 3
    assert dataset.feature_names == ["f0", "f1", "f2"]


@pytest.mark.unit
def test_load_csv_missing_file(tmp_path) -> None:
    # R1: a missing file must raise instead of producing an empty dataset
    # that would fail much later inside the model.
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")


@pytest.mark.unit
def test_load_csv_rejects_unknown_target(tmp_path, csv_writer) -> None:
    # R1: if the target column does not exist there is nothing to learn
    # from, so the loader stops immediately.
    path = csv_writer(tmp_path / "x.csv", ["f0", "y"], [[1.0, 2.0]])
    with pytest.raises(ValueError, match="target column not found"):
        load_csv(path, target_column="target")


@pytest.mark.unit
def test_load_csv_rejects_short_row(tmp_path, csv_writer) -> None:
    # R1: a truncated row must not silently become a row with a different
    # number of features. The message names the expectation.
    path = csv_writer(tmp_path / "x.csv", ["f0", "target"], [[1.0]])
    with pytest.raises(ValueError, match="expected 2"):
        load_csv(path)


@pytest.mark.unit
def test_load_csv_rejects_header_only(tmp_path, csv_writer) -> None:
    # R1: a table without data rows is useless for training.
    path = csv_writer(tmp_path / "x.csv", ["f0", "target"], [])
    with pytest.raises(ValueError, match="at least one row"):
        load_csv(path)


# Cleaning
# R2: remove noise, but never hide the fact that everything was removed.


@pytest.mark.unit
def test_clean_removes_duplicates() -> None:
    # Two identical rows collapse into one; the unique row is kept.
    dataset = Dataset(
        features=[[1.0], [1.0], [2.0]],
        targets=[1.0, 1.0, 2.0],
        feature_names=["f0"],
    )
    cleaned = clean(dataset)
    assert len(cleaned) == 2


@pytest.mark.unit
def test_clean_removes_nan_rows() -> None:
    # R2: the row containing NaN is dropped and the valid row is untouched.
    dataset = Dataset(
        features=[[float("nan")], [2.0]],
        targets=[1.0, 2.0],
        feature_names=["f0"],
    )
    cleaned = clean(dataset)
    assert len(cleaned) == 1
    assert cleaned.features == [[2.0]]


@pytest.mark.unit
def test_clean_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        clean(Dataset(features=[], targets=[], feature_names=[]))


@pytest.mark.unit
def test_clean_rejects_all_rows_dropped() -> None:
    # R2: if cleaning removes every row the function must fail here, not
    # let the next stage crash with an unrelated error.
    dataset = Dataset(
        features=[[float("nan")]],
        targets=[1.0],
        feature_names=["f0"],
    )
    with pytest.raises(ValueError, match="no rows left"):
        clean(dataset)


# Splitting
# R3: a reproducible split with a predictable test size.


@pytest.mark.unit
@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_bad_test_size(clean_dataset: Dataset, test_size: float) -> None:
    # Boundaries are exclusive: 0 and 1 are invalid, so are negative values
    # and anything above one.
    with pytest.raises(ValueError, match="between 0 and 1"):
        train_test_split(clean_dataset, test_size=test_size, seed=1)


@pytest.mark.unit
def test_split_rejects_single_row() -> None:
    # A single row cannot fill both sides of the split.
    dataset = Dataset(features=[[1.0]], targets=[1.0], feature_names=["f0"])
    with pytest.raises(ValueError, match="at least two rows"):
        train_test_split(dataset, test_size=0.5, seed=1)


@pytest.mark.unit
@pytest.mark.parametrize("test_size", [0.1, 0.25, 0.5])
def test_split_sizes(clean_dataset: Dataset, test_size: float) -> None:
    # R3: no row is lost or duplicated, and the test part is
    # round(n * test_size), never smaller than one row.
    splits = train_test_split(clean_dataset, test_size=test_size, seed=3)
    total = len(splits.train) + len(splits.test)
    assert total == len(clean_dataset)
    assert len(splits.test) == max(1, round(len(clean_dataset) * test_size))


@pytest.mark.unit
def test_split_is_deterministic(clean_dataset: Dataset) -> None:
    # R3: the same seed produces the same split, which is what makes the
    # DVC pipeline reproducible.
    first = train_test_split(clean_dataset, test_size=0.25, seed=5)
    second = train_test_split(clean_dataset, test_size=0.25, seed=5)
    assert first.train.features == second.train.features
    assert first.test.features == second.test.features


@pytest.mark.unit
def test_split_differs_for_different_seeds(clean_dataset: Dataset) -> None:
    # R3: the seed must actually be used, not ignored.
    first = train_test_split(clean_dataset, test_size=0.25, seed=5)
    second = train_test_split(clean_dataset, test_size=0.25, seed=6)
    assert first.train.features != second.train.features


# Scaling
# R4: fit on the train split, then apply the same statistics everywhere.
# Fitting on the full dataset would leak test information into training.


@pytest.mark.unit
def test_scaler_fit_produces_zero_mean_unit_std(splits) -> None:
    # The defining property: after transform the mean is 0 and the
    # variance is 1 for every feature.
    scaler = StandardScaler.fit(splits.train)
    scaled = scaler.transform(splits.train)
    for column in range(scaled.n_features):
        values = [row[column] for row in scaled.features]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        assert mean == pytest.approx(0.0, abs=1e-9)
        assert variance == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_scaler_handles_constant_column() -> None:
    # R4: a constant feature has zero standard deviation. Dividing by it
    # would raise, so the scaler substitutes 1.0 and every value becomes 0.
    dataset = Dataset(
        features=[[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]],
        targets=[0.0, 1.0, 2.0],
        feature_names=["f0", "f1"],
    )
    scaler = StandardScaler.fit(dataset)
    assert scaler.stds[0] == 1.0
    scaled = scaler.transform(dataset)
    assert [row[0] for row in scaled.features] == [0.0, 0.0, 0.0]


@pytest.mark.unit
def test_scaler_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        StandardScaler.fit(Dataset(features=[], targets=[], feature_names=[]))


@pytest.mark.unit
def test_scaler_rejects_feature_mismatch(splits) -> None:
    # R4: applying a scaler with a different number of features means the
    # artefacts came from different runs, which must not go unnoticed.
    scaler = StandardScaler(means=[0.0], stds=[1.0])
    with pytest.raises(ValueError, match="does not match the scaler"):
        scaler.transform(splits.train)


@pytest.mark.unit
def test_scaler_roundtrip(tmp_path, splits) -> None:
    # R4: save and load must preserve the statistics exactly, otherwise the
    # train and evaluate stages would disagree about the scaling.
    scaler = StandardScaler.fit(splits.train)
    path = tmp_path / "scaler.json"
    save_scaler(path, scaler)
    restored = load_scaler(path)
    assert restored.means == pytest.approx(scaler.means)
    assert restored.stds == pytest.approx(scaler.stds)


@pytest.mark.unit
def test_load_scaler_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_scaler(tmp_path / "missing.json")


@pytest.mark.unit
def test_scaler_from_dict_validates_length() -> None:
    # R4: a corrupted JSON file must not produce a scaler with ragged lists.
    with pytest.raises(ValueError, match="same length"):
        StandardScaler.from_dict({"means": [0.0, 1.0], "stds": [1.0]})


# Writing CSV
# R1 in reverse: what the pipeline writes must be readable by the pipeline.


@pytest.mark.unit
def test_write_dataset_roundtrip(tmp_path, clean_dataset: Dataset) -> None:
    path = tmp_path / "out.csv"
    write_dataset(path, clean_dataset)
    restored = load_csv(path)
    assert len(restored) == len(clean_dataset)
    assert restored.feature_names == clean_dataset.feature_names
