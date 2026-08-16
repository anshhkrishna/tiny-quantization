"""Evaluates all five quantization variants on a single trained model and reports
test accuracy plus weight mean-squared quantization error for each. Real stdout is
committed verbatim to results/run.log.
"""
import numpy as np

from data import load_splits
from model import accuracy
from quantize import gptq_int4, naive_int4, per_channel_int8, per_tensor_int8
from train import train_model

VARIANTS = ("float32", "naive_int4", "per_tensor_int8", "per_channel_int8", "gptq_int4")


def weight_mse(params_a, params_b):
    """Mean squared error between two params' weight matrices, pooled across every
    layer's weights (flattened and concatenated, not averaged per-layer first, so
    layers with more parameters contribute proportionally more).
    """
    diffs = [
        (a["W"] - b["W"]).ravel()
        for a, b in zip(params_a["layers"], params_b["layers"])
    ]
    return float(np.mean(np.concatenate(diffs) ** 2))


def evaluate_all(params, splits):
    """Returns {variant: (test_accuracy, weight_mse)} for all five variants. Weight
    MSE is against the float32 weights themselves, so float32's own entry is 0.0.
    """
    quantized = {
        "naive_int4": naive_int4(params),
        "per_tensor_int8": per_tensor_int8(params),
        "per_channel_int8": per_channel_int8(params),
        "gptq_int4": gptq_int4(params, splits["X_cal"]),
    }
    results = {"float32": (accuracy(params, splits["X_test"], splits["y_test"]), 0.0)}
    for name, q_params in quantized.items():
        acc = accuracy(q_params, splits["X_test"], splits["y_test"])
        mse = weight_mse(q_params, params)
        results[name] = (acc, mse)
    return results


def main():
    splits = load_splits(seed=0)
    params, losses = train_model(splits["X_train"], splits["y_train"])
    print(f"train={splits['X_train'].shape[0]} cal={splits['X_cal'].shape[0]} "
          f"test={splits['X_test'].shape[0]}  final training loss={losses[-1]:.4f}")

    results = evaluate_all(params, splits)
    for name in VARIANTS:
        acc, mse = results[name]
        print(f"{name:17s}  test accuracy = {acc * 100:6.2f}%   weight MSE = {mse:.6e}")


if __name__ == "__main__":
    main()
