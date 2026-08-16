"""From-scratch NumPy multi-layer perceptron for digit classification -- the model
whose trained weights get quantized by src/quantize.py.

A stack of Linear -> ReLU hidden layers followed by a linear output layer, softmax
cross-entropy loss. Forward pass, backward pass, and the loss are all hand-derived on
plain numpy arrays, with no autograd framework involved.
"""
import numpy as np


def init_params(in_dim, hidden_dims, out_dim, seed):
    """He-initialized weights for each Linear layer, biases at zero. Parameters are a
    list of {"W", "b"} dicts, one per layer, in forward order -- this is the shape
    quantize.py and train.py both operate on.
    """
    rng = np.random.default_rng(seed)
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    layers = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        W = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        b = np.zeros(fan_out)
        layers.append({"W": W, "b": b})
    return {"layers": layers}


def forward(params, X, cache=False):
    """ReLU on every hidden layer, linear (no activation) on the output layer.

    Returns (logits, trace). trace is the list of (layer_input, pre_activation) pairs
    backward() needs to reconstruct the chain rule, or None if cache=False.
    """
    a = X
    trace = [] if cache else None
    for i, layer in enumerate(params["layers"]):
        z = a @ layer["W"] + layer["b"]
        is_output = i == len(params["layers"]) - 1
        if cache:
            trace.append((a, z))
        a = z if is_output else np.maximum(z, 0.0)
    return a, trace


def softmax_cross_entropy(logits, y):
    """Mean softmax cross-entropy loss over a batch, and the softmax probabilities
    (returned so backward() can reuse them without recomputing the softmax).
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    log_likelihood = -np.log(probs[np.arange(n), y] + 1e-12)
    return log_likelihood.mean(), probs


def backward(params, y, probs, trace):
    """Hand-derived gradients: dL/dlogits = (probs - onehot) / n for softmax cross-
    entropy, then standard backprop through each Linear -> ReLU layer using the
    (input, pre_activation) pairs recorded by forward()'s trace.
    """
    layers = params["layers"]
    n = probs.shape[0]
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y] = 1.0
    dz = (probs - onehot) / n

    grads = [None] * len(layers)
    for i in reversed(range(len(layers))):
        a_in, _ = trace[i]
        grads[i] = {"W": a_in.T @ dz, "b": dz.sum(axis=0)}
        if i > 0:
            da_in = dz @ layers[i]["W"].T
            _, z_prev = trace[i - 1]
            dz = da_in * (z_prev > 0)
    return grads


def predict(params, X):
    logits, _ = forward(params, X, cache=False)
    return np.argmax(logits, axis=1)


def accuracy(params, X, y):
    return float(np.mean(predict(params, X) == y))
