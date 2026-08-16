"""Load and split the digits dataset used to train the model being quantized."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 0


def load_splits(seed=SEED, cal_size=0.15, test_size=0.15):
    """Load sklearn's bundled digits dataset and split into train/calibration/test.

    Pixel values (0-16 in the raw data) are normalized to [0, 1]. The calibration
    split is drawn separately from train, not as a subset of it, so that GPTQ-style
    calibration activations never overlap with the model's own training data. The
    test split is held out from everything until the final accuracy readout.
    """
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)

    train_size = 1.0 - cal_size - test_size
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=train_size, random_state=seed, stratify=y
    )
    rest_test_fraction = test_size / (cal_size + test_size)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest, y_rest, test_size=rest_test_fraction, random_state=seed, stratify=y_rest
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "X_test": X_test,
        "y_test": y_test,
    }
