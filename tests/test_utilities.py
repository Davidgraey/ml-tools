"""
Shared utilities and distance metrics.
"""

import numpy as np
import pytest
from polyergalio import distances
from polyergalio.utilities import rolling_windows_nd, standardize_data

METRICS = (
    distances.euclidian_distance,
    distances.manhattan_distance,
)


@pytest.mark.parametrize(
    "length,window,overlap",
    ((100, 10, 0), (100, 10, 3), (100, 7, 6), (50, 50, 0)),
)
def test_rolling_windows_shape(length, window, overlap):
    data = np.arange(length, dtype=float)
    windows = rolling_windows_nd(data, window_size=window, num_overlap=overlap)
    stride = window - overlap
    expected = (length - window) // stride + 1
    assert windows.shape == (expected, window)


def test_standardize_centres_and_scales():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=7.0, scale=3.0, size=(500, 4))
    standardized = standardize_data(data, axis=0)
    assert np.allclose(standardized.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(standardized.std(axis=0), 1.0, atol=1e-6)


@pytest.mark.parametrize("metric", METRICS)
def test_distance_is_a_symmetric_metric(metric):
    rng = np.random.default_rng(0)
    left = rng.normal(size=(4, 5))
    right = rng.normal(size=(4, 5))
    assert np.allclose(metric(left, left), 0.0, atol=1e-10)
    assert np.allclose(metric(left, right), metric(right, left))
