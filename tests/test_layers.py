"""
Layers: forward shapes, backward correctness, and the Layer interface.

The gradient tests here compare each layer's backward pass against finite
differences of its own forward pass. Interface tests cover the accessors that
an optimizer calls, which is where several layers used to raise AttributeError
on attribute names that had drifted.
"""

import numpy as np
import pytest

from ml_tools.models.layers.layers import (
    DropoutLayer,
    FourierLayer,
    FrequencyFFT,
    FullyConnectedLayer,
    InverseFourierLayer,
    NormalizeLayer,
    RMSNormLayer,
    hartley,
    hartley_2d,
    kaiming,
    shape_conflict,
    xavier,
)
from ml_tools.models.layers.operators import LatentStack
from conftest import (
    GRADIENT_TOLERANCE,
    as_float64,
    input_gradient_error,
    parameter_gradient_error,
    relative_error,
)


ACTIVATIONS = ("linear", "relu", "relu_leaky", "sigmoid", "tanh", "swish", "softmax")


# -------------    weight initialisation    ------------------------
@pytest.mark.parametrize("initialiser,expected", ((kaiming, 2.0), (xavier, 1.0)))
def test_initialiser_scale(initialiser, expected):
    """variance should track the fan-in rule each initialiser implements"""
    rng = np.random.RandomState(0)
    weights = initialiser(rng, ni=512, no=256)
    assert weights.std() == pytest.approx(np.sqrt(expected / 512), rel=0.1)


# -------------    FullyConnectedLayer    --------------------------
@pytest.mark.slow
@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_fully_connected_input_gradient(activation, small_matrix):
    layer = as_float64(FullyConnectedLayer(4, 3, activation))
    assert input_gradient_error(layer, small_matrix) < GRADIENT_TOLERANCE


@pytest.mark.slow
@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_fully_connected_weight_gradient(activation, small_matrix):
    layer = as_float64(FullyConnectedLayer(4, 3, activation))
    assert (
        parameter_gradient_error(layer, small_matrix, layer.weights, "gradient_weights")
        < GRADIENT_TOLERANCE
    )


@pytest.mark.slow
def test_fully_connected_bias_gradient(small_matrix):
    layer = as_float64(FullyConnectedLayer(4, 3, "tanh"))
    assert (
        parameter_gradient_error(layer, small_matrix, layer.bias, "gradient_bias")
        < GRADIENT_TOLERANCE
    )


@pytest.mark.slow
def test_fully_connected_handles_three_dimensional_input(sequence_batch):
    """(batch, sequence, hidden) must round trip through the 2D reshape"""
    layer = as_float64(FullyConnectedLayer(4, 3, "swish"))
    assert input_gradient_error(layer, sequence_batch) < GRADIENT_TOLERANCE


def test_fully_connected_output_shape(sequence_batch):
    layer = FullyConnectedLayer(4, 7, "relu")
    assert layer.forward(sequence_batch).shape == (3, 8, 7)


def test_fully_connected_rejects_mismatched_width():
    layer = FullyConnectedLayer(4, 3, "relu")
    with pytest.raises(AssertionError):
        layer.forward(np.zeros((5, 9)))


def test_forced_activation_resolves_a_derivative(small_matrix):
    """
    backward(forced_activation=...) must look the name up in the derivative
    registry, not the activation registry.
    """
    layer = FullyConnectedLayer(4, 3, "relu")
    layer.forward(small_matrix)
    delta = layer.backward(np.ones((6, 3)), forced_activation="linear")
    assert delta.shape == small_matrix.shape


def test_fully_connected_interface(small_matrix):
    layer = FullyConnectedLayer(4, 3, "relu")
    layer.forward(small_matrix)
    layer.backward(np.ones((6, 3)))

    assert layer.num_parameters == 4 * 3 + 3
    assert layer.get_weights().shape == (4 * 3 + 3,)
    assert set(layer.get_gradients()) == {"gradient_weights", "gradient_bias"}
    layer.zero_gradients()
    assert not layer.get_gradients()["gradient_weights"].any()
    layer.purge()
    assert layer.input is None


def test_gradients_preserve_float32(small_matrix):
    layer = FullyConnectedLayer(4, 3, "relu")
    layer.forward(small_matrix.astype(np.float32))
    delta = layer.backward(np.ones((6, 3), dtype=np.float32))
    assert delta.dtype == np.float32
    assert layer.gradient_weights.dtype == np.float32


def test_update_weights_moves_against_the_gradient(small_matrix):
    layer = FullyConnectedLayer(4, 3, "linear")
    layer.forward(small_matrix)
    layer.backward(np.ones((6, 3)))
    before = layer.weights.copy()
    layer.update_weights(
        gradient_bias=layer.gradient_bias * 0.1,
        gradient_weights=layer.gradient_weights * 0.1,
    )
    assert not np.allclose(before, layer.weights)


# -------------    NormalizeLayer    -------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("shift_scale", (True, False))
def test_normalize_input_gradient(shift_scale, small_matrix):
    layer = NormalizeLayer(4, shift_scale=shift_scale)
    if shift_scale:
        layer.scale_gamma = np.full((1, 4), 1.3)
        layer.shift_beta = np.full((1, 4), 0.2)
    assert input_gradient_error(layer, small_matrix) < GRADIENT_TOLERANCE


@pytest.mark.slow
@pytest.mark.parametrize(
    "parameter_name,gradient_name",
    (("scale_gamma", "gradient_gamma"), ("shift_beta", "gradient_beta")),
)
def test_normalize_parameter_gradients(parameter_name, gradient_name, small_matrix):
    layer = NormalizeLayer(4, shift_scale=True)
    layer.scale_gamma = np.full((1, 4), 1.3)
    layer.shift_beta = np.full((1, 4), 0.2)
    error = parameter_gradient_error(
        layer, small_matrix, getattr(layer, parameter_name), gradient_name
    )
    assert error < GRADIENT_TOLERANCE


def test_normalize_output_is_standardised(small_matrix):
    layer = NormalizeLayer(4, shift_scale=False)
    output = layer.forward(small_matrix)
    assert np.allclose(output.mean(axis=-1), 0.0, atol=1e-10)
    assert np.allclose(output.std(axis=-1), 1.0, atol=1e-3)


def test_normalize_initialises_to_identity():
    """gamma at 1 and beta at 0, otherwise the layer attenuates from the start"""
    layer = NormalizeLayer(6, shift_scale=True)
    assert np.allclose(layer.scale_gamma, 1.0)
    assert np.allclose(layer.shift_beta, 0.0)


def test_normalize_preserves_float32(small_matrix):
    layer = NormalizeLayer(4, shift_scale=True)
    assert layer.forward(small_matrix.astype(np.float32)).dtype == np.float32


@pytest.mark.parametrize("shift_scale", (True, False))
def test_normalize_interface(shift_scale, small_matrix):
    """
    All four accessors used to reference beta/gamma names that did not exist,
    raising on one branch or the other.
    """
    layer = NormalizeLayer(4, shift_scale=shift_scale)
    layer.forward(small_matrix)
    layer.backward(np.ones((6, 4)))

    assert layer.num_parameters == (8 if shift_scale else 0)
    assert len(layer.get_weights()) == 2
    assert bool(layer.get_gradients()) is shift_scale
    layer.zero_gradients()
    layer.purge()


def test_normalize_skipped_by_optimizer_when_parameterless(small_matrix):
    """an empty gradient dict is how a layer opts out of the update"""
    layer = NormalizeLayer(4, shift_scale=False)
    layer.forward(small_matrix)
    layer.backward(np.ones((6, 4)))
    assert layer.get_gradients() == {}


# -------------    RMSNormLayer    ---------------------------------
@pytest.mark.slow
def test_rms_norm_input_gradient(small_matrix):
    layer = RMSNormLayer(4)
    layer.scale_gamma = np.full((1, 4), 1.2)
    assert input_gradient_error(layer, small_matrix) < GRADIENT_TOLERANCE


@pytest.mark.slow
def test_rms_norm_gradient_three_dimensional(sequence_batch):
    layer = RMSNormLayer(4)
    assert input_gradient_error(layer, sequence_batch) < GRADIENT_TOLERANCE


@pytest.mark.slow
def test_rms_norm_gamma_gradient(small_matrix):
    layer = RMSNormLayer(4)
    layer.scale_gamma = np.full((1, 4), 1.2)
    error = parameter_gradient_error(
        layer, small_matrix, layer.scale_gamma, "gradient_gamma"
    )
    assert error < GRADIENT_TOLERANCE


def test_rms_norm_does_not_centre():
    """
    RMSNorm divides by root mean square and does not subtract the mean, which
    is the whole difference from LayerNorm.
    """
    x_data = np.full((3, 4), 5.0)
    output = RMSNormLayer(4).forward(x_data)
    assert np.allclose(output, 1.0)
    assert not np.allclose(output.mean(axis=-1), 0.0)


def test_rms_norm_interface(small_matrix):
    layer = RMSNormLayer(4)
    layer.forward(small_matrix)
    layer.backward(np.ones((6, 4)))
    assert layer.num_parameters == 4
    assert layer.get_weights().shape == (1, 4)
    assert set(layer.get_gradients()) == {"gradient_gamma"}
    layer.zero_gradients()
    layer.purge()


# -------------    DropoutLayer    ---------------------------------
def test_dropout_zeroes_roughly_the_requested_fraction():
    layer = DropoutLayer(dropout_prob=0.5, use_rescale=False)
    kept = layer.forward(np.ones((400, 250)), training_now=True)
    assert kept.mean() == pytest.approx(0.5, abs=0.02)


def test_dropout_inverted_scaling_preserves_the_mean():
    layer = DropoutLayer(dropout_prob=0.5, use_rescale=True)
    kept = layer.forward(np.ones((400, 250)), training_now=True)
    assert kept.mean() == pytest.approx(1.0, abs=0.02)


def test_dropout_passes_through_at_evaluation():
    layer = DropoutLayer(dropout_prob=0.5)
    x_data = np.ones((20, 10))
    assert np.array_equal(layer.forward(x_data, training_now=False), x_data)


def test_dropout_backward_applies_the_mask():
    layer = DropoutLayer(dropout_prob=0.5, use_rescale=False)
    layer.forward(np.ones((30, 20)), training_now=True)
    delta = layer.backward(np.ones((30, 20)))
    assert np.array_equal(delta.astype(bool), layer.mask.astype(bool))


def test_dropout_rejects_degenerate_probabilities():
    for probability in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(AssertionError):
            DropoutLayer(dropout_prob=probability)


def test_dropout_is_parameterless():
    layer = DropoutLayer(dropout_prob=0.1)
    assert layer.num_parameters == 0
    assert layer.get_gradients() == {}
    assert layer.get_weights() is None


@pytest.mark.xfail(
    reason="use_rescale=False compensates at neither train nor eval time, so "
    "inference is scaled by 1/keep_prob relative to training",
    strict=True,
)
def test_dropout_train_and_eval_agree_in_expectation():
    layer = DropoutLayer(dropout_prob=0.5, use_rescale=False)
    x_data = np.ones((400, 250))
    training = layer.forward(x_data, training_now=True).mean()
    evaluation = layer.forward(x_data, training_now=False).mean()
    assert training == pytest.approx(evaluation, abs=0.02)


@pytest.mark.xfail(
    reason="an evaluation forward leaves the previous training mask in place, "
    "so backward keeps applying it",
    strict=True,
)
def test_dropout_backward_after_eval_is_transparent():
    layer = DropoutLayer(dropout_prob=0.5, use_rescale=False)
    layer.forward(np.ones((30, 20)), training_now=True)
    layer.forward(np.ones((30, 20)), training_now=False)
    assert np.allclose(layer.backward(np.ones((30, 20))), 1.0)


# -------------    the Hartley transform helpers    ----------------
@pytest.mark.parametrize("length", (6, 7, 8, 9, 16, 17))
def test_hartley_is_its_own_inverse_up_to_scale(length):
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(3, length))
    assert np.allclose(hartley(hartley(x_data)) / length, x_data)


@pytest.mark.parametrize("shape", ((4, 6), (5, 5), (3, 8)))
def test_hartley_2d_is_its_own_inverse_up_to_scale(shape):
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=shape)
    scale = shape[0] * shape[1]
    assert np.allclose(hartley_2d(hartley_2d(x_data)) / scale, x_data)


def test_hartley_is_full_rank():
    """
    Re(fft) discards the circularly-odd component and is rank deficient.
    Hartley keeps everything, which is why the FFT layers use it.
    """
    length = 8
    basis = np.eye(length)
    columns = np.array([hartley(row) for row in basis])
    assert np.linalg.matrix_rank(columns) == length


def test_hartley_output_is_real():
    rng = np.random.default_rng(0)
    assert not np.iscomplexobj(hartley(rng.normal(size=(2, 8))))


# -------------    the FFT layers    -------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("use_2d", (True, False))
def test_fourier_layer_gradient(use_2d):
    rng = np.random.default_rng(0)
    layer = FourierLayer(use_2d=use_2d)
    assert input_gradient_error(layer, rng.normal(size=(3, 8, 4))) < GRADIENT_TOLERANCE


@pytest.mark.slow
@pytest.mark.parametrize("use_2d", (True, False))
def test_inverse_fourier_layer_gradient(use_2d):
    rng = np.random.default_rng(0)
    layer = InverseFourierLayer(use_2d=use_2d)
    assert input_gradient_error(layer, rng.normal(size=(3, 8, 4))) < GRADIENT_TOLERANCE


@pytest.mark.parametrize("use_2d", (True, False))
def test_fourier_pair_round_trips(use_2d):
    """
    Forward then inverse must reconstruct. With Re(fft) it did not even
    approximately, since both directions threw information away.
    """
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(3, 8, 4))
    forward = FourierLayer(use_2d=use_2d)
    inverse = InverseFourierLayer(use_2d=use_2d)
    assert np.allclose(inverse.forward(forward.forward(x_data)), x_data)


@pytest.mark.parametrize("use_2d", (True, False))
def test_fourier_layer_preserves_shape(use_2d):
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(3, 8, 4))
    assert FourierLayer(use_2d=use_2d).forward(x_data).shape == x_data.shape


def test_fourier_layers_are_parameterless():
    for layer in (FourierLayer(), InverseFourierLayer(), FrequencyFFT(8, 4)):
        assert layer.num_parameters == 0
        assert layer.get_gradients() == {}


@pytest.mark.slow
@pytest.mark.parametrize("window_size", (4, 5, 8, 9))
def test_frequency_fft_gradient(window_size):
    rng = np.random.default_rng(0)
    layer = FrequencyFFT(max_sequence_length=16, window_size=window_size)
    x_data = rng.normal(size=(6, window_size))
    assert input_gradient_error(layer, x_data) < GRADIENT_TOLERANCE


@pytest.mark.slow
def test_frequency_fft_gradient_three_dimensional():
    rng = np.random.default_rng(0)
    layer = FrequencyFFT(max_sequence_length=8, window_size=4)
    assert input_gradient_error(layer, rng.normal(size=(2, 5, 4))) < GRADIENT_TOLERANCE


def test_frequency_fft_preserves_window_width():
    layer = FrequencyFFT(max_sequence_length=8, window_size=4)
    assert layer.forward(np.ones((8, 4))).shape == (8, 4)


def test_frequency_fft_rejects_a_mismatched_window():
    layer = FrequencyFFT(max_sequence_length=8, window_size=4)
    with pytest.raises(AssertionError):
        layer.forward(np.ones((8, 6)))


def test_frequency_fft_rejects_too_many_windows():
    layer = FrequencyFFT(max_sequence_length=4, window_size=4)
    with pytest.raises(AssertionError):
        layer.forward(np.ones((9, 4)))


# -------------    LatentStack    ----------------------------------
@pytest.mark.parametrize(
    "shape", ((6,), (4, 3), (2, 3, 5))
)
def test_latent_stack_concatenates_on_the_last_axis(shape):
    rng = np.random.default_rng(0)
    left, right = rng.normal(size=shape), rng.normal(size=shape)
    stacked = LatentStack().forward(left, right)
    assert stacked is not None, "unsupported rank returned None"
    assert stacked.shape[-1] == shape[-1] * 2


def test_latent_stack_backward_splits_by_width():
    stacker = LatentStack()
    left = np.zeros((4, 3))
    right = np.zeros((4, 5))
    stacked = stacker.forward(left, right)
    assert stacked.shape == (4, 8)

    grad_left, grad_right = stacker.backward(np.ones((4, 8)))
    assert grad_left.shape == left.shape
    assert grad_right.shape == right.shape


def test_latent_stack_round_trips_values():
    stacker = LatentStack()
    left = np.arange(12, dtype=float).reshape(4, 3)
    right = np.arange(20, dtype=float).reshape(4, 5)
    stacker.forward(left, right)
    grad_left, grad_right = stacker.backward(
        np.hstack([left, right])
    )
    assert np.array_equal(grad_left, left)
    assert np.array_equal(grad_right, right)


@pytest.mark.xfail(
    reason="forward branches on ndim 1, 2 and 3 only, so rank 4 falls through "
    "and returns None",
    strict=True,
)
def test_latent_stack_handles_rank_four():
    rng = np.random.default_rng(0)
    left = right = rng.normal(size=(2, 3, 4, 5))
    assert LatentStack().forward(left, right) is not None


# -------------    the declared shapes    --------------------------
def test_shapes_reports_input_and_output():
    layer = FullyConnectedLayer(6, 12, "relu")
    assert layer.shapes == {"input": ((6,),), "output": ((12,),)}


def test_shapes_are_known_before_any_data_arrives():
    """the point of declaring them: nothing has been forwarded yet"""
    layer = NormalizeLayer(ni=9)
    assert layer.shapes["input"] == ((9,),)
    assert not hasattr(layer, "in_shape")


@pytest.mark.parametrize(
    "layer,expected",
    (
        (RMSNormLayer(ni=8), ((8,),)),
        (FrequencyFFT(max_sequence_length=20, window_size=64), ((None, 64),)),
        (DropoutLayer(0.25), ((None,),)),
        (FourierLayer(), ((None,),)),
        (InverseFourierLayer(use_2d=False), ((None,),)),
    ),
)
def test_layers_declare_their_trailing_axes(layer, expected):
    assert layer.shapes["input"] == expected


def test_a_merge_declares_one_shape_per_source():
    """two inputs to forward, so two declared input shapes"""
    assert len(LatentStack().shapes["input"]) == 2


def test_an_elementwise_layer_passes_its_input_shape_through():
    """
    Dropout has no width of its own to declare, so it reports the incoming one.
    Were it to report (None,) instead, every layer downstream of a dropout
    would go unchecked.
    """
    assert DropoutLayer(0.5).infer_output_shapes(((16,),)) == ((16,),)


def test_a_stack_infers_the_summed_width():
    assert LatentStack().infer_output_shapes(((12,), (4,))) == ((16,),)


def test_a_stack_stays_unknown_when_a_source_is():
    assert LatentStack().infer_output_shapes(((12,), (None,))) == ((None,),)


def test_a_fixed_width_layer_ignores_what_arrives():
    """its output is set by its weights, not by its input"""
    assert FullyConnectedLayer(6, 12, "relu").infer_output_shapes(((6,),)) == ((12,),)


# -------------    shape comparison    -----------------------------
@pytest.mark.parametrize(
    "produced,expected",
    (
        ((12,), (12,)),
        ((None,), (12,)),
        ((12,), (None,)),
        ((4, 12), (12,)),
        ((12,), (4, 12)),
        ((None, 64), (20, 64)),
    ),
)
def test_compatible_shapes_report_nothing(produced, expected):
    assert shape_conflict(produced, expected) is None


@pytest.mark.parametrize(
    "produced,expected",
    (((12,), (9,)), ((4, 12), (4, 9)), ((3, 12), (4, 12))),
)
def test_conflicting_shapes_name_the_axis(produced, expected):
    assert shape_conflict(produced, expected) is not None


def test_comparison_is_right_aligned():
    """(12,) describes the last axis, so a leading axis is not its business"""
    assert shape_conflict((5, 12), (12,)) is None
