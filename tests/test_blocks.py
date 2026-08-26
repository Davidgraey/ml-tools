"""
Composite blocks: the FNet-style Fourier mixer and the SPECTRE mixer.

Two things get checked. The FFT adjoint helpers are verified directly against
finite differences at both sequence-length parities, because the Hermitian
symmetry correction differs between them and getting it wrong costs tens of
percent. Then every learnable parameter in each block is gradient checked,
including the complex Toeplitz taps.
"""

import numpy as np
import pytest

from ml_tools.models.blocks import (
    FourierAttention,
    SpectreAttention,
    irfft_adjoint,
    rfft_adjoint,
)
from ml_tools.models.model_loss import MSELoss
from ml_tools.models.optimizers import SGD
from conftest import (
    GRADIENT_TOLERANCE,
    numeric_gradient,
    numeric_gradient_complex,
    relative_error,
)


PARITIES = (6, 7, 8, 9, 16, 17)


def promote(block):
    """float64 for every parameter, so finite differences have the precision"""
    for name in ("fc_query", "fc_values", "fc_1", "fc_2", "fc"):
        layer = getattr(block, name, None)
        if layer is not None:
            layer.weights = layer.weights.astype(np.float64)
            layer.bias = layer.bias.astype(np.float64)
    if hasattr(block, "activation_bias"):
        block.activation_bias = block.activation_bias.astype(np.float64)
    if getattr(block, "norm_b", None) is not None and block.norm_b.shift_scale:
        block.norm_b.scale_gamma = block.norm_b.scale_gamma.astype(np.float64)
        block.norm_b.shift_beta = block.norm_b.shift_beta.astype(np.float64)
    return block


def block_parameter_error(block, x_data, targets, parameter, gradient_reader) -> float:
    def scalar():
        prediction = block.forward(x_data)
        return float(((prediction - targets) ** 2).mean())

    scalar()
    prediction = block.forward(x_data)
    block.backward(2 * (prediction - targets) / prediction.size)
    analytic = gradient_reader()

    numeric = numeric_gradient(scalar, parameter)
    return relative_error(analytic, numeric)


# -------------    the FFT adjoints    -----------------------------
@pytest.mark.slow
@pytest.mark.parametrize("sequence", PARITIES)
def test_irfft_adjoint(sequence):
    """
    Paired bins carry double the energy. Nyquist only exists for even lengths,
    so for odd lengths the final bin is paired and must double too -- the case
    that was silently wrong for every odd sequence.
    """
    rng = np.random.default_rng(0)
    bins = sequence // 2 + 1
    spectrum = rng.normal(size=(1, bins, 3)) + 1j * rng.normal(size=(1, bins, 3))
    upstream = rng.normal(size=(1, sequence, 3))

    def scalar():
        return float(
            (np.fft.irfft(spectrum, n=sequence, axis=1) * upstream).sum()
        )

    analytic = irfft_adjoint(upstream, sequence, axis=1)
    numeric = numeric_gradient_complex(scalar, spectrum)
    assert relative_error(analytic, numeric) < GRADIENT_TOLERANCE


@pytest.mark.slow
@pytest.mark.parametrize("sequence", PARITIES)
def test_rfft_adjoint(sequence):
    rng = np.random.default_rng(0)
    bins = sequence // 2 + 1
    signal = rng.normal(size=(1, sequence, 3))
    cotangent = rng.normal(size=(1, bins, 3)) + 1j * rng.normal(size=(1, bins, 3))

    def scalar():
        spectrum = np.fft.rfft(signal, axis=1)
        return float(
            (spectrum.real * cotangent.real).sum()
            + (spectrum.imag * cotangent.imag).sum()
        )

    analytic = rfft_adjoint(cotangent, sequence, axis=1)
    numeric = numeric_gradient(scalar, signal)
    assert relative_error(analytic, numeric) < GRADIENT_TOLERANCE


# -------------    FourierAttention    -----------------------------
def test_fourier_attention_preserves_shape(sequence_batch):
    block = FourierAttention(ni=4, no=4, use_2d=True)
    assert block.forward(sequence_batch).shape == sequence_batch.shape


def test_fourier_attention_requires_matching_widths():
    with pytest.raises(AssertionError):
        FourierAttention(ni=8, no=4)


def test_fourier_attention_rejects_two_dimensional_input_when_2d():
    """
    With use_2d the transform mixes the last two axes. On (batch, hidden) that
    means mixing across the batch, which leaks between independent samples.
    """
    block = FourierAttention(ni=4, no=4, use_2d=True)
    with pytest.raises(AssertionError):
        block.forward(np.zeros((3, 4)))


def test_fourier_attention_does_not_leak_across_the_batch(sequence_batch):
    block = FourierAttention(ni=4, no=4, use_2d=True)
    baseline = block.forward(sequence_batch.copy())
    perturbed = sequence_batch.copy()
    perturbed[0] += 5.0
    assert np.allclose(block.forward(perturbed)[1:], baseline[1:])


@pytest.mark.slow
def test_fourier_attention_weight_gradient(sequence_batch):
    rng = np.random.default_rng(0)
    block = promote(FourierAttention(ni=4, no=4))
    targets = rng.normal(size=sequence_batch.shape)
    error = block_parameter_error(
        block, sequence_batch, targets, block.fc.weights,
        lambda: block.fc.gradient_weights,
    )
    assert error < GRADIENT_TOLERANCE


@pytest.mark.slow
def test_fourier_attention_norm_gradients(sequence_batch):
    rng = np.random.default_rng(0)
    targets = rng.normal(size=sequence_batch.shape)
    for parameter_name, gradient_name in (
        ("scale_gamma", "gradient_gamma"),
        ("shift_beta", "gradient_beta"),
    ):
        block = promote(FourierAttention(ni=4, no=4))
        error = block_parameter_error(
            block, sequence_batch, targets,
            getattr(block.norm_b, parameter_name),
            lambda block=block, name=gradient_name: getattr(block.norm_b, name),
        )
        assert error < GRADIENT_TOLERANCE, parameter_name


def test_fourier_attention_interface(sequence_batch):
    block = FourierAttention(ni=4, no=4)
    output = block.forward(sequence_batch)
    block.backward(np.ones_like(output))

    assert block.num_parameters > 0
    assert len(block.get_weights()) == 3
    assert set(block.get_gradients()) == {"norm_a", "fc", "norm_b"}
    block.zero_gradients()
    block.purge()


# -------------    SpectreAttention    -----------------------------
@pytest.mark.slow
@pytest.mark.parametrize("sequence", PARITIES)
@pytest.mark.parametrize("num_heads", (1, 2))
def test_spectre_parameter_gradients(sequence, num_heads):
    """every projection plus the learnable per-head modReLU bias"""
    rng = np.random.default_rng(0)
    hidden = 4
    x_data = rng.normal(size=(3, sequence, hidden))
    targets = rng.normal(size=x_data.shape)

    def make():
        return promote(
            SpectreAttention(
                sequence_length=sequence, hidden_dim=hidden, num_heads=num_heads
            )
        )

    for parameter_name, reader in (
        ("fc_query", lambda b: b.fc_query.gradient_weights),
        ("fc_values", lambda b: b.fc_values.gradient_weights),
        ("fc_1", lambda b: b.fc_1.gradient_weights),
        ("fc_2", lambda b: b.fc_2.gradient_weights),
    ):
        block = make()
        error = block_parameter_error(
            block, x_data, targets,
            getattr(block, parameter_name).weights,
            lambda block=block, reader=reader: reader(block),
        )
        assert error < GRADIENT_TOLERANCE, parameter_name

    block = make()
    error = block_parameter_error(
        block, x_data, targets, block.activation_bias,
        lambda: block.gradient_bias,
    )
    assert error < GRADIENT_TOLERANCE, "activation_bias"


@pytest.mark.slow
@pytest.mark.parametrize("sequence", (7, 8))
@pytest.mark.parametrize("num_heads", (1, 2))
def test_spectre_band_tap_gradient(sequence, num_heads):
    """
    The optional Toeplitz gate, whose taps are complex. The convolution is
    depth-wise, so each head carries its own tap vector and the gradient has to
    stay separated by head.
    """
    rng = np.random.default_rng(0)
    hidden = 4
    x_data = rng.normal(size=(3, sequence, hidden))
    targets = rng.normal(size=x_data.shape)

    block = promote(
        SpectreAttention(
            sequence_length=sequence, hidden_dim=hidden,
            num_heads=num_heads, band_radius=2,
        )
    )
    shape = (num_heads, 5)
    block.band_taps = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * 0.3

    def scalar():
        prediction = block.forward(x_data)
        return float(((prediction - targets) ** 2).mean())

    scalar()
    prediction = block.forward(x_data)
    block.backward(2 * (prediction - targets) / prediction.size)

    numeric = numeric_gradient_complex(scalar, block.band_taps)
    assert relative_error(block.gradient_band, numeric) < GRADIENT_TOLERANCE


@pytest.mark.slow
def test_spectre_input_gradient(sequence_batch):
    rng = np.random.default_rng(0)
    block = promote(SpectreAttention(sequence_length=8, hidden_dim=4))
    upstream = rng.normal(size=sequence_batch.shape)
    x_data = np.array(sequence_batch, dtype=np.float64)

    block.forward(x_data)
    analytic = block.backward(upstream.copy())
    numeric = numeric_gradient(
        lambda: float((block.forward(x_data) * upstream).sum()), x_data
    )
    assert relative_error(analytic, numeric) < GRADIENT_TOLERANCE


@pytest.mark.parametrize("num_heads", (1, 2, 4))
def test_spectre_preserves_shape(sequence_batch, num_heads):
    block = SpectreAttention(sequence_length=8, hidden_dim=4, num_heads=num_heads)
    assert block.forward(sequence_batch).shape == sequence_batch.shape


@pytest.mark.parametrize("num_heads", (1, 2, 4))
def test_spectre_gate_is_diagonal_across_channels(num_heads):
    """
    The gate holds one value per head per frequency, broadcast over that head's
    channels. A gate with a channel axis would be a different model.
    """
    block = SpectreAttention(sequence_length=8, hidden_dim=4, num_heads=num_heads)
    block.forward(np.random.default_rng(0).normal(size=(3, 8, 4)))
    assert block.gate.shape == (3, num_heads, block.num_frequencies)


def test_spectre_single_head_is_unchanged_by_the_head_axis():
    """
    The default has to stay the layer it was before heads existed: one gate,
    every channel gated by it.
    """
    block = SpectreAttention(sequence_length=8, hidden_dim=4)
    x_data = np.random.default_rng(0).normal(size=(3, 8, 4))
    output = block.forward(x_data)

    expected = np.fft.irfft(
        block.value_transform * block.gate[:, 0, :, None], n=8, axis=1
    )
    assert np.allclose(output, expected)


def test_spectre_rejects_heads_that_do_not_divide_the_channels():
    with pytest.raises(AssertionError):
        SpectreAttention(sequence_length=8, hidden_dim=6, num_heads=4)

    with pytest.raises(AssertionError):
        SpectreAttention(sequence_length=8, hidden_dim=4, num_heads=0)


def test_spectre_heads_do_not_mix_channels():
    """
    Heads partition the channel axis, so zeroing one head's gate must silence
    exactly that head's channels and leave the rest untouched.
    """
    num_heads = 2
    block = SpectreAttention(sequence_length=8, hidden_dim=4, num_heads=num_heads)
    x_data = np.random.default_rng(0).normal(size=(3, 8, 4))

    baseline = block.forward(x_data)
    gate = block.gate.copy()

    gate[:, 0, :] = 0.0
    block.gate = gate
    silenced = np.fft.irfft(
        block._merge_heads(
            block._split_heads(block.value_transform) * block._align_gate(gate)
        ),
        n=8, axis=1,
    )

    head_dim = block.head_dim
    assert np.allclose(silenced[..., :head_dim], 0.0)
    assert np.allclose(silenced[..., head_dim:], baseline[..., head_dim:])


@pytest.mark.parametrize(
    "shape,reason",
    (
        ((2, 6, 4), "sequence length differs from construction"),
        ((2, 8, 5), "hidden dim differs from construction"),
        ((2, 8), "missing the sequence axis"),
    ),
)
def test_spectre_rejects_mismatched_input(shape, reason):
    block = SpectreAttention(sequence_length=8, hidden_dim=4)
    with pytest.raises(AssertionError):
        block.forward(np.zeros(shape))


@pytest.mark.parametrize("num_heads", (1, 2))
@pytest.mark.parametrize("band_radius", (0, 2))
def test_spectre_interface(band_radius, num_heads):
    block = SpectreAttention(
        sequence_length=8, hidden_dim=4,
        num_heads=num_heads, band_radius=band_radius,
    )
    assert block.num_parameters > 0, "must be reportable before any backward pass"

    output = block.forward(np.random.default_rng(0).normal(size=(2, 8, 4)))
    block.backward(np.ones_like(output))

    expected = {"gradient_bias", "fc_query", "fc_values", "fc_1", "fc_2"}
    if band_radius:
        expected.add("gradient_band")
    assert set(block.get_gradients()) == expected

    block.zero_gradients()
    block.purge()


def test_spectre_rejects_a_degenerate_spread():
    for radius in (-1,):
        with pytest.raises(Exception):
            SpectreAttention(sequence_length=8, hidden_dim=4, band_radius=radius)


# -------------    training through the optimizer    ---------------
@pytest.mark.slow
@pytest.mark.parametrize(
    "make_block",
    (
        lambda: FourierAttention(ni=8, no=8),
        lambda: SpectreAttention(sequence_length=9, hidden_dim=8),
        lambda: SpectreAttention(sequence_length=9, hidden_dim=8, band_radius=2),
        lambda: SpectreAttention(sequence_length=9, hidden_dim=8, num_heads=4),
        lambda: SpectreAttention(
            sequence_length=9, hidden_dim=8, num_heads=4, band_radius=2
        ),
    ),
)
def test_block_reduces_loss(make_block):
    rng = np.random.default_rng(0)
    block = make_block()
    sequence = getattr(block, "sequence_length", 9)
    x_data = rng.normal(size=(8, sequence, 8))
    targets = np.sin(x_data * 2)

    loss = MSELoss()
    optimizer = SGD(0.02)
    first = None
    for _ in range(150):
        prediction = block.forward(x_data)
        value = loss(prediction, targets)
        if first is None:
            first = value
        block.backward(loss.backward())
        optimizer.step([block])

    assert value < first, f"loss did not fall: {first} -> {value}"
