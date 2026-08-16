# tiny-quantization

> post training quantization from scratch in numpy, measuring how much gptq style compensation recovers from int4 rounding

## Claim

Per-channel int8 post-training quantization of a small trained classifier is nearly
lossless relative to the full-precision model. Naive round-to-nearest int4 costs real,
measurable accuracy. A GPTQ-style sequential error-compensation scheme recovers a
specific, measurable fraction of that int4 loss, but not all of it.

## Baseline

The full-precision (float32) trained model, and naive round-to-nearest int4
quantization (single per-tensor scale, symmetric, no error compensation) applied to
that same model's weights. Both are the reference points every other scheme in this
project is measured against.

## Result

A from-scratch NumPy MLP (64 -> 32 -> 32 -> 10, no autograd framework) is trained on
scikit-learn's digits dataset, then quantized five ways and evaluated on a held-out
test split, repeated across 5 training seeds (`results/rigor.log`):

| variant | test accuracy (mean +/- std, 5 seeds) |
|---|---|
| float32 | 97.56% +/- 0.18% |
| naive int4 | 95.41% +/- 1.04% |
| per-tensor int8 | 97.56% +/- 0.18% |
| per-channel int8 | 97.63% +/- 0.18% |
| gptq-style int4 | 96.52% +/- 0.95% |

Per-channel int8 is statistically indistinguishable from float32, confirming the
"nearly lossless" part of the claim. Naive int4 trails float32 by 0.74 to 4.08 points
depending on seed, a real drop on every one of the 5 seeds, not just on average.
GPTQ-style int4 (sequential quantize-and-compensate, following the OBQ/GPTQ line of
work, with a directly inverted per-layer Hessian rather than GPTQ's Cholesky trick,
tractable at this weight-matrix size) beats naive int4 on every seed too, by 0.37 to
2.60 points, but never fully closes the gap to float32: a 1.04-point residual gap on
average. This matches the expected mechanism: GPTQ compensates each layer's own
rounding error but not how that error compounds through the layers downstream of it.

![test accuracy per quantization variant, with seed-variance error bars and float32/naive-int4 reference lines](results/headline.png)

The recovery is also more seed-dependent than a single run would suggest: one seed's
GPTQ model tied float32 exactly, while another recovered only about a quarter of naive
int4's gap. See `results/FINDING.md` for the short writeup and `results/baseline.log`,
`results/run.log`, `results/rigor.log` for the full numbers.

## Reproduce

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py            # trains the float32 model, prints baseline + naive-int4 accuracy -> results/baseline.log
python src/experiment.py       # single-seed evaluation of all five variants -> results/run.log
python src/rigor.py            # 5-seed evaluation, leakage check -> results/rigor.log
python src/plot_headline.py    # regenerates results/headline.png from results/rigor.log
pytest tests/
```
