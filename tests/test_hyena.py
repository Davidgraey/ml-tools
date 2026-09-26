"""
HyenaOperator and its parts: exact gradients, causality, and the FFT convolution
against its time-domain definition.
"""

import numpy as np
import pytest
from conftest import numeric_gradient, relative_error
from polyergalio.models.layers.hyena_layers import (
    HyenaOperator,
    causal_convolution,
)

HIDDEN = 4

CONFIGURATIONS = {
    "order_1": {"order": 1},
    "order_2": {"order": 2},
    "order_3_odd_length": {"order": 3, "sequence_length": 7},
    "long_kernel": {"short_kernel": 5},
}


def build(config: dict):
    config = dict(config)
    sequence = config.pop("sequence_length", 8)
    layer = HyenaOperator(sequence, HIDDEN, filter_features=6, positional_bands=3, **config)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, sequence, HIDDEN))
    upstream = rng.normal(size=x.shape)
    mask = np.ones((2, sequence))
    mask[1, -2:] = 0
    return layer, x, upstream, mask


@pytest.mark.parametrize("config", CONFIGURATIONS.values(), ids=CONFIGURATIONS.keys())
def test_gradients_match_finite_differences(config):
    layer, x, upstream, mask = build(config)

    def loss():
        return float(np.sum(layer.forward(x, mask=mask) * upstream))

    loss()
    layer.zero_gradients()
    dx = layer.backward(upstream)
    gradients = layer.get_gradients()
    assert relative_error(dx, numeric_gradient(loss, x)) < 1e-6

    for owner, sublayer in layer.owned_layers().items():
        for name, analytic in gradients[owner].items():
            parameter = getattr(sublayer, name.removeprefix("gradient_"))
            assert relative_error(analytic, numeric_gradient(loss, parameter)) < 1e-6, f"{owner}.{name}"


def test_causal_convolution_matches_direct_sum():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=(2, 9, 3))
    filters = rng.normal(size=(9, 3))
    direct = np.stack([sum(filters[t - j] * signal[:, j] for j in range(t + 1)) for t in range(9)], axis=1)
    assert np.allclose(causal_convolution(signal, filters), direct)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_output_ignores_future_tokens(order):
    layer, x, _, _ = build({"order": order})
    before = layer.forward(x).copy()
    x[:, 5] += 1.0
    after = layer.forward(x)
    assert np.allclose(after[:, :5], before[:, :5])
    assert not np.allclose(after[:, 5:], before[:, 5:])


def test_padding_does_not_reach_later_tokens():
    layer, x, _, _ = build({})
    mask = np.ones((2, 8))
    mask[:, :2] = 0
    before = layer.forward(x, mask=mask).copy()
    x[:, 0] += 5.0
    after = layer.forward(x, mask=mask)
    assert np.allclose(after[:, 3:], before[:, 3:])
