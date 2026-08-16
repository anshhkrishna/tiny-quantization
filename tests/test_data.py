import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import load_splits


def test_shapes_and_total_count():
    splits = load_splits(seed=0)
    n_train = splits["X_train"].shape[0]
    n_cal = splits["X_cal"].shape[0]
    n_test = splits["X_test"].shape[0]
    assert splits["X_train"].shape[1] == 64
    assert splits["X_cal"].shape[1] == 64
    assert splits["X_test"].shape[1] == 64
    assert splits["y_train"].shape == (n_train,)
    assert splits["y_cal"].shape == (n_cal,)
    assert splits["y_test"].shape == (n_test,)
    assert n_train + n_cal + n_test == 1797


def test_pixel_value_range():
    splits = load_splits(seed=0)
    for key in ("X_train", "X_cal", "X_test"):
        X = splits[key]
        assert X.min() >= 0.0
        assert X.max() <= 1.0
        assert X.max() > 0.5


def test_label_range():
    splits = load_splits(seed=0)
    for key in ("y_train", "y_cal", "y_test"):
        y = splits[key]
        assert y.min() == 0
        assert y.max() == 9
        assert np.issubdtype(y.dtype, np.integer)


def test_no_overlap_between_splits():
    splits = load_splits(seed=0)
    train_rows = {tuple(row) for row in splits["X_train"]}
    cal_rows = {tuple(row) for row in splits["X_cal"]}
    test_rows = {tuple(row) for row in splits["X_test"]}
    assert train_rows.isdisjoint(cal_rows)
    assert train_rows.isdisjoint(test_rows)
    assert cal_rows.isdisjoint(test_rows)


def test_seed_reproducibility():
    a = load_splits(seed=0)
    b = load_splits(seed=0)
    assert np.array_equal(a["X_train"], b["X_train"])
    assert np.array_equal(a["y_train"], b["y_train"])
    assert np.array_equal(a["X_test"], b["X_test"])
    assert np.array_equal(a["y_test"], b["y_test"])


def test_seed_variation_changes_split():
    a = load_splits(seed=0)
    b = load_splits(seed=1)
    assert not np.array_equal(a["X_train"], b["X_train"])


def test_split_proportions_approximate_target():
    splits = load_splits(seed=0, cal_size=0.15, test_size=0.15)
    total = 1797
    n_train = splits["X_train"].shape[0]
    n_cal = splits["X_cal"].shape[0]
    n_test = splits["X_test"].shape[0]
    assert abs(n_train / total - 0.70) < 0.02
    assert abs(n_cal / total - 0.15) < 0.02
    assert abs(n_test / total - 0.15) < 0.02
