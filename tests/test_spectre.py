"""
SpectreAttention and SpectreDecoderAttention: exact gradients in every
configuration, and the decoder's causal training forward against its
time-domain definition and against prefill / decode_step.
"""

import numpy as np
import pytest
from conftest import numeric_gradient, relative_error
from polyergalio.models.layers.spectre_layers import SpectreAttention, SpectreDecoderAttention
from polyergalio.models.optimizers import Adam

SEQUENCE = 8
HIDDEN = 8

CONFIGURATIONS = {
    "plain": {},
    "memory": {"memory_tokens": 3},
    "band": {"band_radius": 1},
    "band_and_memory": {"band_radius": 2, "memory_tokens": 2},
    "odd_length": {"sequence_length": 7},
}


def build(layer_class, config: dict):
    config = dict(config)
    sequence = config.pop("sequence_length", SEQUENCE)
    layer = layer_class(sequence, HIDDEN, num_heads=2, **config)
    rng = np.random.default_rng(0)
    if layer.band_radius:
        layer.band_taps = 0.3 * (rng.normal(size=layer.band_taps.shape) + 1j * rng.normal(size=layer.band_taps.shape))
    layer.head_gate.weights_2 = rng.normal(size=layer.head_gate.weights_2.shape)
    x = rng.normal(size=(3, sequence, HIDDEN))
    upstream = rng.normal(size=x.shape)
    mask = np.ones((3, sequence))
    mask[:, -2:] = 0
    return layer, x, upstream, mask


def complex_numeric_gradient(loss, tensor, step=1e-6):
    """dL/dRe + i dL/dIm, the convention the layer's complex gradients follow"""
    gradient = np.zeros_like(tensor)
    for index in np.ndindex(tensor.shape):
        original = tensor[index]
        for direction in (1, 1j):
            tensor[index] = original + step * direction
            up = loss()
            tensor[index] = original - step * direction
            down = loss()
            tensor[index] = original
            gradient[index] += direction * (up - down) / (2 * step)
    return gradient


@pytest.mark.parametrize("layer_class", [SpectreAttention, SpectreDecoderAttention], ids=lambda c: c.__name__)
@pytest.mark.parametrize("config", CONFIGURATIONS.values(), ids=CONFIGURATIONS.keys())
def test_gradients_match_finite_differences(layer_class, config):
    layer, x, upstream, mask = build(layer_class, config)

    def loss():
        return float(np.sum(layer.forward(x, mask=mask, training_now=False) * upstream))

    def analytic():
        loss()
        layer.zero_gradients()
        dx = layer.backward(upstream)
        return dx, layer.get_gradients()

    dx, gradients = analytic()
    assert relative_error(dx, numeric_gradient(loss, x)) < 1e-6

    parameters = {
        "gradient_query_weights": layer.query_weights,
        "gradient_values_weights": layer.values_weights,
        "gradient_query_bias": layer.query_bias,
        "gradient_bias": layer.activation_bias,
    }
    for name, tensor in parameters.items():
        assert relative_error(gradients[name], numeric_gradient(loss, tensor)) < 1e-6, name
    for name in ("gamma", "beta", "weights_1", "bias_1", "weights_2", "bias_2"):
        analytic_gate = gradients["head_gate"]["gradient_" + name]
        assert relative_error(analytic_gate, numeric_gradient(loss, getattr(layer.head_gate, name))) < 1e-6, name
    if layer.memory_tokens:
        memory = gradients["persistent_memory"]["gradient_memory"]
        assert relative_error(memory, numeric_gradient(loss, layer.memory.memory)) < 1e-6
    if layer.band_radius:
        assert relative_error(gradients["gradient_band"], complex_numeric_gradient(loss, layer.band_taps)) < 1e-6


def test_wrm_gradients_match_finite_differences():
    layer, x, upstream, mask = build(SpectreAttention, {"use_wrm": True})

    def loss():
        return float(np.sum(layer.forward(x, mask=mask, training_now=False) * upstream))

    loss()
    layer.zero_gradients()
    assert relative_error(layer.backward(upstream), numeric_gradient(loss, x)) < 1e-6


def test_the_gate_starts_near_identity():
    """bias holds every gate at 1 + 0j; the shrunken projection only nudges it with content"""
    layer = SpectreAttention(SEQUENCE, HIDDEN, num_heads=2)
    layer.head_gate.weights_2[...] = 0.0
    x = np.random.default_rng(1).normal(size=(2, SEQUENCE, HIDDEN))
    values = layer._project_heads(x, layer.values_weights, layer.values_bias)
    np.testing.assert_allclose(layer.forward(x, training_now=False), values, atol=1e-12)

    fresh = SpectreAttention(SEQUENCE, HIDDEN, num_heads=2)
    fresh.forward(x, training_now=False)
    assert np.abs(fresh.gate - 1).mean() < 0.3


def test_each_head_gates_from_its_own_queries_only():
    layer = SpectreAttention(SEQUENCE, HIDDEN, num_heads=2)
    x = np.random.default_rng(3).normal(size=(2, SEQUENCE, HIDDEN))
    layer.forward(x, training_now=False)
    before = layer.gate.copy()
    layer.query_weights[0] += np.random.default_rng(4).normal(size=layer.query_weights[0].shape)
    layer.forward(x, training_now=False)
    assert not np.allclose(layer.gate[:, 0], before[:, 0])
    np.testing.assert_allclose(layer.gate[:, 1], before[:, 1])


def test_the_layer_learns_token_mixing():
    """each position copies its predecessor -- a target only the spectral gate can express"""
    rng = np.random.default_rng(0)
    layer = SpectreAttention(SEQUENCE, HIDDEN, num_heads=2)
    x = rng.normal(size=(16, SEQUENCE, HIDDEN))
    target = np.roll(x, 1, axis=1)
    optimizer = Adam(1e-2)
    for _ in range(200):
        layer.zero_gradients()
        output = layer.forward(x, training_now=True)
        layer.backward(2 * (output - target) / output.size)
        optimizer.step([layer])
    held_out = rng.normal(size=(16, SEQUENCE, HIDDEN))
    error = np.mean((layer.forward(held_out, training_now=False) - np.roll(held_out, 1, axis=1)) ** 2)
    assert error < 0.1


# -------------    decoding    ----------------------------
def causal_reference(layer, x, mask=None):
    """
    The decoder's definition, written directly in time: position p mixes values at positions <= p
    with the filter from its chunk anchor's pooled query.
    """
    memory, batch = layer.memory_tokens, x.shape[0]
    mask = np.ones(x.shape[:2]) if mask is None else mask
    combined = x
    if memory:
        bank = np.broadcast_to(layer.memory.get_memory()[None], (batch, memory, layer.hidden_dim))
        combined = np.concatenate([bank, x], axis=1)
    queries = layer._project_heads(combined, layer.query_weights, layer.query_bias)
    values = layer._project_heads(combined, layer.values_weights, layer.values_bias)
    values[:, memory:] *= mask[..., None]

    output = np.zeros_like(x)
    for position in range(x.shape[1]):
        anchor = (position // layer.chunk_size) * layer.chunk_size
        query_sum = queries[:, :memory].sum(axis=1) + (queries[:, memory: memory + anchor + 1] * mask[:, : anchor + 1, None]).sum(axis=1)
        counts = np.maximum(mask[:, : anchor + 1].sum(axis=1, keepdims=True), 1.0) + memory
        filters = np.fft.irfft(layer.gate_from_pooled_sum(query_sum, counts), n=combined.shape[1], axis=1)
        row = memory + position
        output[:, position] = sum(filters[:, row - j] * values[:, j] for j in range(row + 1))
    return output


CHUNK_SIZES = [1, 3, SEQUENCE]

DECODER_CONFIGURATIONS = {
    "plain": {},
    "memory": {"memory_tokens": 2},
    "band_and_memory": {"memory_tokens": 2, "band_radius": 1},
    "full_reconstruction": {"memory_tokens": 2, "use_positional_phase": False},
}


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("config", DECODER_CONFIGURATIONS.values(), ids=DECODER_CONFIGURATIONS.keys())
def test_the_training_forward_matches_the_causal_definition(config, chunk_size):
    layer, x, _, mask = build(SpectreDecoderAttention, {**config, "chunk_size": chunk_size})
    np.testing.assert_allclose(layer.forward(x, mask=mask), causal_reference(layer, x, mask), atol=1e-10)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_the_training_forward_never_reads_the_future(chunk_size):
    layer, x, _, mask = build(SpectreDecoderAttention, {"memory_tokens": 2, "chunk_size": chunk_size})
    before = layer.forward(x, mask=mask)
    for position in range(1, SEQUENCE):
        changed = x.copy()
        changed[:, position:] += 1.0
        np.testing.assert_allclose(layer.forward(changed, mask=mask)[:, :position], before[:, :position], atol=1e-12)


@pytest.mark.parametrize("chunk_size", [3, SEQUENCE])
def test_chunked_gradients_match_finite_differences(chunk_size):
    layer, x, upstream, mask = build(SpectreDecoderAttention, {"memory_tokens": 2, "band_radius": 1, "chunk_size": chunk_size})

    def loss():
        return float(np.sum(layer.forward(x, mask=mask) * upstream))

    loss()
    layer.zero_gradients()
    assert relative_error(layer.backward(upstream), numeric_gradient(loss, x)) < 1e-6
    gradients = layer.get_gradients()
    assert relative_error(gradients["gradient_query_weights"], numeric_gradient(loss, layer.query_weights)) < 1e-6
    assert relative_error(gradients["head_gate"]["gradient_weights_1"], numeric_gradient(loss, layer.head_gate.weights_1)) < 1e-6
    assert relative_error(gradients["persistent_memory"]["gradient_memory"], numeric_gradient(loss, layer.memory.memory)) < 1e-6


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("config", DECODER_CONFIGURATIONS.values(), ids=DECODER_CONFIGURATIONS.keys())
def test_prefill_and_decode_match_the_training_forward(config, chunk_size):
    layer, x, _, _ = build(SpectreDecoderAttention, {**config, "chunk_size": chunk_size})
    trained = layer.forward(x)
    for prompt in (1, 2, 4):
        outputs = [layer.prefill(x[:, :prompt])] + [layer.decode_step(x[:, t]) for t in range(prompt, SEQUENCE)]
        for position, output in zip(range(prompt - 1, SEQUENCE), outputs):
            np.testing.assert_allclose(output, trained[:, position], atol=1e-10)


def test_a_padded_prompt_decodes_like_the_masked_training_forward():
    layer, x, _, _ = build(SpectreDecoderAttention, {"memory_tokens": 2, "chunk_size": 3})
    mask = np.ones((x.shape[0], SEQUENCE))
    mask[:, 1] = 0
    trained = layer.forward(x, mask=mask)
    outputs = [layer.prefill(x[:, :4], mask=mask[:, :4])] + [layer.decode_step(x[:, t]) for t in range(4, SEQUENCE)]
    for position, output in zip(range(3, SEQUENCE), outputs):
        np.testing.assert_allclose(output, trained[:, position], atol=1e-10)


@pytest.mark.parametrize("config", DECODER_CONFIGURATIONS.values(), ids=DECODER_CONFIGURATIONS.keys())
def test_decoding_past_the_window_matches_a_forward_over_the_last_window(config):
    """chunk_size=1 only: larger chunks' anchors stop lining up with a fresh window once it slides"""
    layer, _, _, _ = build(SpectreDecoderAttention, config)
    x = np.random.default_rng(2).normal(size=(2, 2 * SEQUENCE + 3, HIDDEN))
    outputs = [layer.prefill(x[:, :1])] + [layer.decode_step(x[:, t]) for t in range(1, x.shape[1])]
    for t in range(SEQUENCE - 1, x.shape[1]):
        window = x[:, t - SEQUENCE + 1: t + 1]
        np.testing.assert_allclose(outputs[t], layer.forward(window, training_now=False)[:, -1], atol=1e-10)