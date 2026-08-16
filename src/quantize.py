"""Weight quantization schemes for the trained MLP, from naive round-to-nearest up to
a GPTQ-style sequential error-compensating quantizer. Every scheme here quantizes only
weight matrices, leaving biases float32 -- biases are a tiny fraction of a model's
parameters, so quantizing them buys negligible compression, which is why the int4/int8
literature this project follows measures weight-only quantization.

Weight matrices are laid out (fan_in, fan_out): forward() computes `a @ W`, so each
*output channel* is a column of W (axis 1), not a row -- this is the layout every
per-channel/per-column operation below assumes.
"""
import numpy as np

INT4_MIN, INT4_MAX = -8, 7
INT8_MIN, INT8_MAX = -128, 127


def quantize_symmetric(W, level_min, level_max):
    """Round-to-nearest symmetric quantization of an array with one scale for the
    whole array. Returns (W_dequantized, scale): the dequantized (still float)
    reconstruction used for inference, and the scale factor applied. Works on a full
    weight matrix (per-tensor) or a single row/column (per-channel) alike.
    """
    max_abs = np.abs(W).max()
    scale = max_abs / level_max if max_abs > 0 else 1.0
    q = np.clip(np.round(W / scale), level_min, level_max)
    return q * scale, scale


def _quantize_every_layer(params, level_min, level_max):
    """Applies quantize_symmetric independently to each layer's weight matrix (one
    scale per layer), leaving biases untouched.
    """
    layers = []
    for layer in params["layers"]:
        W_q, _ = quantize_symmetric(layer["W"], level_min, level_max)
        layers.append({"W": W_q, "b": layer["b"]})
    return {"layers": layers}


def naive_int4(params):
    """Single per-tensor scale, symmetric, no error compensation."""
    return _quantize_every_layer(params, INT4_MIN, INT4_MAX)


def per_tensor_int8(params):
    """Single per-tensor scale, symmetric, no error compensation -- same shape as
    naive_int4 but with int8's wider level range.
    """
    return _quantize_every_layer(params, INT8_MIN, INT8_MAX)


def per_channel_int8(params):
    """One int8 scale per output channel (column of W), rather than one scale for
    the whole matrix. Lets each output channel use the full int8 range regardless of
    how its weight magnitude compares to other channels in the same layer.
    """
    layers = []
    for layer in params["layers"]:
        W = layer["W"]
        W_q = np.empty_like(W)
        for j in range(W.shape[1]):
            W_q[:, j], _ = quantize_symmetric(W[:, j], INT8_MIN, INT8_MAX)
        layers.append({"W": W_q, "b": layer["b"]})
    return {"layers": layers}


def gptq_int4(params, X_cal, level_min=INT4_MIN, level_max=INT4_MAX, damp=1e-2):
    """GPTQ/OBQ-style sequential quantization: within each layer, quantize one input
    feature's row of W at a time (round-to-nearest at a fixed per-layer scale), then
    push the rounding error into the not-yet-quantized rows using the inverse of a
    Hessian built from that layer's calibration-set input activations, following the
    classical optimal-brain-surgeon weight-update rule OBQ/GPTQ both use. Because W is
    laid out (fan_in, fan_out) here, "row" (axis 0, input features) plays the role
    GPTQ's "column" plays for its (d_out, d_in) layout -- the Hessian is shared across
    every output channel either way, since it only depends on the layer's inputs.

    Calibration activations are threaded through each layer's *already-quantized*
    predecessors as they are computed, so later layers' Hessians reflect the
    compounding effect of upstream quantization, the same as GPTQ processing a
    network's layers in order. Weight matrices here are small enough (at most 64
    rows) that inverting each layer's Hessian directly is cheap, so this skips GPTQ's
    Cholesky-update trick, which exists to avoid that inversion at LLM scale.
    """
    layers = []
    a = X_cal
    n_layers = len(params["layers"])
    for i, layer in enumerate(params["layers"]):
        W = layer["W"].copy()
        fan_in = W.shape[0]

        H = (2.0 / a.shape[0]) * (a.T @ a)
        H += damp * np.mean(np.diag(H)) * np.eye(fan_in)
        H_inv = np.linalg.inv(H)

        max_abs = np.abs(W).max()
        scale = max_abs / level_max if max_abs > 0 else 1.0

        for r in range(fan_in):
            w_row = W[r, :]
            q_row = np.clip(np.round(w_row / scale), level_min, level_max) * scale
            error = w_row - q_row
            W[r, :] = q_row
            if r + 1 < fan_in:
                coeff = H_inv[r, r + 1:] / H_inv[r, r]
                W[r + 1:, :] -= np.outer(coeff, error)

        z = a @ W + layer["b"]
        a = z if i == n_layers - 1 else np.maximum(z, 0.0)
        layers.append({"W": W, "b": layer["b"]})
    return {"layers": layers}
