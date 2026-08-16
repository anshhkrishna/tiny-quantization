"""Adam training loop for the digit-classification MLP, and a CLI entry point that
trains the float32 model and reports its test accuracy alongside the naive-int4
quantized version's. Run as `python train.py` from within `src/`; real stdout is
committed verbatim to `results/baseline.log`.
"""
import numpy as np

from data import load_splits
from model import accuracy, backward, forward, init_params, softmax_cross_entropy
from quantize import naive_int4

HIDDEN_DIMS = (32, 32)
OUT_DIM = 10
MODEL_SEED = 0
TRAIN_RNG_SEED = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 60

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


def _zeros_like(params):
    return {"layers": [{"W": np.zeros_like(l["W"]), "b": np.zeros_like(l["b"])}
                        for l in params["layers"]]}


def _adam_update(arr, g, m, v, t, lr):
    m[:] = ADAM_BETA1 * m + (1 - ADAM_BETA1) * g
    v[:] = ADAM_BETA2 * v + (1 - ADAM_BETA2) * (g * g)
    m_hat = m / (1 - ADAM_BETA1 ** t)
    v_hat = v / (1 - ADAM_BETA2 ** t)
    arr -= lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)


def train_model(X_train, y_train, hidden_dims=HIDDEN_DIMS, num_epochs=NUM_EPOCHS,
                 batch_size=BATCH_SIZE, lr=LEARNING_RATE, model_seed=MODEL_SEED,
                 train_rng_seed=TRAIN_RNG_SEED, log_every=10):
    """Trains a fresh MLP with minibatch Adam, minibatches reshuffled every epoch from
    an RNG independent of the weight-init seed. Returns (params, epoch_losses).
    """
    params = init_params(X_train.shape[1], hidden_dims, OUT_DIM, seed=model_seed)
    m = _zeros_like(params)
    v = _zeros_like(params)
    t = 0
    rng = np.random.default_rng(train_rng_seed)
    n = X_train.shape[0]
    epoch_losses = []
    for epoch in range(1, num_epochs + 1):
        perm = rng.permutation(n)
        batch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits, trace = forward(params, X_train[idx], cache=True)
            loss, probs = softmax_cross_entropy(logits, y_train[idx])
            grads = backward(params, y_train[idx], probs, trace)
            t += 1
            for layer, g, ml, vl in zip(params["layers"], grads, m["layers"], v["layers"]):
                _adam_update(layer["W"], g["W"], ml["W"], vl["W"], t, lr)
                _adam_update(layer["b"], g["b"], ml["b"], vl["b"], t, lr)
            batch_losses.append(loss)
        epoch_losses.append(float(np.mean(batch_losses)))
        if epoch == 1 or epoch % log_every == 0 or epoch == num_epochs:
            print(f"  epoch {epoch:3d}/{num_epochs}  loss={epoch_losses[-1]:.4f}")
    return params, epoch_losses


def main():
    splits = load_splits(seed=0)
    print(f"train={splits['X_train'].shape[0]} cal={splits['X_cal'].shape[0]} "
          f"test={splits['X_test'].shape[0]}  hidden_dims={HIDDEN_DIMS} "
          f"model_seed={MODEL_SEED} train_rng_seed={TRAIN_RNG_SEED} "
          f"batch_size={BATCH_SIZE} lr={LEARNING_RATE} epochs={NUM_EPOCHS}")

    params, losses = train_model(splits["X_train"], splits["y_train"])
    assert np.isfinite(losses).all(), "training loss went non-finite"
    assert losses[-1] < losses[0], "training loss did not decrease"

    float32_acc = accuracy(params, splits["X_test"], splits["y_test"])
    print(f"float32 test accuracy = {float32_acc * 100:.2f}%")

    quantized = naive_int4(params)
    int4_acc = accuracy(quantized, splits["X_test"], splits["y_test"])
    print(f"naive int4 test accuracy = {int4_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
