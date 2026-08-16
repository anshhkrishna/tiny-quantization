A small NumPy-only MLP (64 to 32 to 32 to 10, trained from scratch on scikit-learn's
digits dataset) was quantized five ways and evaluated across 5 training seeds. Naive
round-to-nearest int4 costs a real, measurable amount of test accuracy relative to the
float32 model: 97.56% mean vs. 95.41% mean, a drop that held on every individual seed
(0.74 to 4.08 points). Per-channel int8 was statistically indistinguishable from
float32 (97.63% vs. 97.56%), confirming the "nearly lossless" part of the claim. A
GPTQ-style sequential error-compensation scheme, applied to int4, recovered part of
naive int4's loss but not all of it: 96.52% mean, beating naive int4 on every seed
(0.37 to 2.60 points) while never matching float32. The genuinely surprising part is
how seed-dependent that recovery was: one seed's GPTQ model tied float32 exactly,
while another recovered barely a quarter of naive int4's gap, so a single-seed run
would have reported either "GPTQ closes the gap completely" or "GPTQ barely helps,"
both wrong as a general claim.
