"""Tests the core quantization claims across multiple training seeds: per-channel
int8 tracks float32 closely, naive int4 loses real accuracy relative to float32, and
GPTQ-style int4 recovers a real, measurable fraction of that naive-int4 loss --
alongside a check that GPTQ's calibration activations never leak from train or test.
"""
import numpy as np
import pytest

from data import load_splits
from rigor import SEEDS, check_no_leakage, run_all_seeds

PER_CHANNEL_TOLERANCE = 0.02
NAIVE_BELOW_FLOAT32_MARGIN = 0.003
GPTQ_ABOVE_NAIVE_MARGIN = 0.002


@pytest.fixture(scope="module")
def per_variant():
    return run_all_seeds(seeds=SEEDS, verbose=False)


def test_per_channel_int8_close_to_float32(per_variant):
    float32 = np.array(per_variant["float32"])
    per_channel = np.array(per_variant["per_channel_int8"])
    assert np.all(np.abs(float32 - per_channel) < PER_CHANNEL_TOLERANCE)


def test_naive_int4_measurably_below_float32(per_variant):
    float32 = np.array(per_variant["float32"])
    naive = np.array(per_variant["naive_int4"])
    assert np.all(float32 - naive > NAIVE_BELOW_FLOAT32_MARGIN)


def test_gptq_int4_measurably_above_naive_int4(per_variant):
    naive = np.array(per_variant["naive_int4"])
    gptq = np.array(per_variant["gptq_int4"])
    assert np.all(gptq - naive > GPTQ_ABOVE_NAIVE_MARGIN)


def test_calibration_split_never_overlaps_train_or_test():
    splits = load_splits(seed=0)
    check_no_leakage(splits)
