"""Naive round-to-nearest int4 weight quantization: one symmetric scale per weight
matrix (per layer, not per output row), no rounding-error compensation. This is the
reference floor every other scheme (per-tensor int8, per-channel int8, GPTQ-style
int4) gets measured against.

Only weight matrices are quantized; biases stay float32. Biases are a tiny fraction of
a model's parameters, so quantizing them buys negligible compression -- weight-only
quantization is what the int4/int8 literature this project follows actually measures.
"""
import numpy as np

INT4_MIN, INT4_MAX = -8, 7


def quantize_symmetric(W, level_min, level_max):
    """Round-to-nearest symmetric quantization of a single weight matrix with one
    scale for the whole matrix. Returns (W_dequantized, scale): the dequantized
    (still float) reconstruction used for inference, and the scale factor applied.
    """
    max_abs = np.abs(W).max()
    scale = max_abs / level_max if max_abs > 0 else 1.0
    q = np.clip(np.round(W / scale), level_min, level_max)
    return q * scale, scale


def naive_int4(params):
    """Applies quantize_symmetric independently to each layer's weight matrix (one
    scale per layer), leaving biases untouched.
    """
    layers = []
    for layer in params["layers"]:
        W_q, _ = quantize_symmetric(layer["W"], INT4_MIN, INT4_MAX)
        layers.append({"W": W_q, "b": layer["b"]})
    return {"layers": layers}
