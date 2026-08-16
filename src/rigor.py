"""Repeats training and quantization across multiple seeds to report mean and
standard deviation of test accuracy per quantization variant, and checks that GPTQ's
calibration activations never draw from the training or test splits.

The data split (train/calibration/test rows) is held fixed across seeds via
data_seed, so only weight initialization and minibatch order vary from seed to seed --
this isolates the effect of quantization scheme from the effect of which rows landed
in which split. model_seed and train_rng_seed are both varied together per seed, kept
offset from each other so weight init and minibatch shuffling never share a stream.
"""
import numpy as np

from data import load_splits
from experiment import VARIANTS, evaluate_all
from train import train_model

SEEDS = (0, 1, 2, 3, 4)
DATA_SEED = 0
TRAIN_RNG_OFFSET = 100


def check_no_leakage(splits):
    """Asserts the calibration split shares no rows with train or test -- the
    boundary GPTQ's calibration activations must never cross.
    """
    def as_row_set(X):
        return {tuple(row) for row in X}

    cal_rows = as_row_set(splits["X_cal"])
    train_rows = as_row_set(splits["X_train"])
    test_rows = as_row_set(splits["X_test"])
    assert not (cal_rows & train_rows), "calibration split overlaps training split"
    assert not (cal_rows & test_rows), "calibration split overlaps test split"


def run_all_seeds(seeds=SEEDS, data_seed=DATA_SEED, verbose=True):
    """Trains and evaluates all five variants once per seed, holding the data split
    fixed across seeds. Returns {variant: list of per-seed accuracies}, in seed order.
    """
    splits = load_splits(seed=data_seed)
    check_no_leakage(splits)

    per_variant = {name: [] for name in VARIANTS}
    for seed in seeds:
        params, losses = train_model(
            splits["X_train"], splits["y_train"],
            model_seed=seed, train_rng_seed=seed + TRAIN_RNG_OFFSET, log_every=10 ** 9,
        )
        results = evaluate_all(params, splits)
        for name in VARIANTS:
            acc, _ = results[name]
            per_variant[name].append(acc)
        if verbose:
            summary = "  ".join(f"{name}={results[name][0] * 100:.2f}%" for name in VARIANTS)
            print(f"seed {seed}: final loss={losses[-1]:.4f}  {summary}")
    return per_variant


def main():
    per_variant = run_all_seeds()
    print()
    print("calibration/train/test leakage check: passed, no row overlap")
    print()
    for name in VARIANTS:
        accs = np.array(per_variant[name])
        print(f"{name:17s}  mean = {accs.mean() * 100:6.2f}%   std = {accs.std() * 100:5.2f}%"
              f"   (n={len(accs)})")


if __name__ == "__main__":
    main()
