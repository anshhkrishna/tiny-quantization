# tiny-quantization

> post-training quantization from scratch in numpy, from int8 down to a gptq-style int4

## Claim

Per-channel int8 post-training quantization of a small trained classifier is nearly
lossless relative to the full-precision model. Naive round-to-nearest int4 costs real,
measurable accuracy. A GPTQ-style sequential error-compensation scheme recovers a
specific, measurable fraction of that int4 loss.

## Baseline

The full-precision (float32) trained model, and naive round-to-nearest int4 quantization
(single scale, no error compensation) applied to the same model's weights.

## Status

Planning complete. Implementation in progress.
