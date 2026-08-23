"""
Activations and their derivatives.

Every derivative registered in derivative_dictionary is a vector-Jacobian
product: given (post-activation output, pre-activation z, upstream gradient) it
returns the finished delta. These tests check that contract against finite
differences of the activation itself, which is the only way to catch a
derivative that is plausible but wrong -- the failure mode that had relu,
relu_leaky, sigmoid and swish all returning incorrect gradients.
"""

import numpy as np
import pytest

from ml_tools.models import activations
from conftest import GRADIENT_TOLERANCE, numeric_gradient, relative_error


# mod_relu takes complex input and a bias vector, so it has its own contract
# and is exercised separately below
ELEMENTWISE = ("linear", "relu", "relu_leaky", "sigmoid", "tanh", "swish")
ALL_REGISTERED = tuple(activations.activation_dictionary)


def test_every_activation_has_a_derivative():
    missing = set(activations.activation_dictionary) - set(
        activations.derivative_dictionary
    )
    assert not missing, f"activations without a registered derivative: {missing}"


@pytest.mark.parametrize("name", ALL_REGISTERED)
def test_activation_preserves_shape(name):
    x_data = np.linspace(-3, 3, 24).reshape(4, 6)
    if name == "mod_relu":
        x_data = x_data + 1j * x_data[::-1]
        assert activations.activation_dictionary[name](x_data, -0.2).shape == x_data.shape
    else:
        assert activations.activation_dictionary[name](x_data).shape == x_data.shape


@pytest.mark.slow
@pytest.mark.parametrize("name", ELEMENTWISE)
def test_derivative_matches_finite_differences(name):
    """the VJP must equal d/dx of sum(upstream * activation(x))"""
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(5, 4)) * 2
    upstream = rng.normal(size=(5, 4))

    activation = activations.activation_dictionary[name]
    derivative = activations.derivative_dictionary[name]

    analytic = derivative(activation(x_data), x_data, upstream)
    numeric = numeric_gradient(
        lambda: float((activation(x_data) * upstream).sum()), x_data
    )

    assert relative_error(analytic, numeric) < GRADIENT_TOLERANCE


@pytest.mark.slow
def test_softmax_derivative_is_a_true_vjp():
    """
    softmax couples every element of a row, so its VJP cannot be a diagonal
    factor. This is the case that used to raise on any batched input.
    """
    rng = np.random.default_rng(1)
    x_data = rng.normal(size=(5, 4))
    upstream = rng.normal(size=(5, 4))

    analytic = activations.softmax_derivative(
        activations.softmax(x_data), x_data, upstream
    )
    numeric = numeric_gradient(
        lambda: float((activations.softmax(x_data) * upstream).sum()), x_data
    )

    assert relative_error(analytic, numeric) < GRADIENT_TOLERANCE


@pytest.mark.parametrize("name", ELEMENTWISE + ("softmax",))
def test_derivative_returns_upstream_shape(name):
    x_data = np.linspace(-2, 2, 12).reshape(3, 4)
    upstream = np.ones_like(x_data)
    activation = activations.activation_dictionary[name]
    delta = activations.derivative_dictionary[name](activation(x_data), x_data, upstream)
    assert delta.shape == upstream.shape


def test_relu_gates_on_the_pre_activation():
    """
    The bug this guards: reading the sign off the post-activation output, which
    is never negative, so the gate was always open and relu backpropagated as
    if it were linear.
    """
    z_values = np.array([[-2.0, -0.5, 0.5, 2.0]])
    output = activations.relu(z_values)
    upstream = np.ones_like(z_values)

    delta = activations.relu_derivative(output, z_values, upstream)

    assert delta[0, 0] == 0.0 and delta[0, 1] == 0.0
    assert delta[0, 2] == 1.0 and delta[0, 3] == 1.0


def test_relu_leaky_uses_alpha_below_zero():
    z_values = np.array([[-1.0, 1.0]])
    output = activations.relu_leaky(z_values)
    delta = activations.relu_leaky_derivative(output, z_values, np.ones_like(z_values))
    assert delta[0, 0] == pytest.approx(0.1)
    assert delta[0, 1] == pytest.approx(1.0)


def test_derivatives_preserve_float32():
    """a float32 network must not be silently promoted mid-backward-pass"""
    x_data = np.linspace(-2, 2, 12).astype(np.float32).reshape(3, 4)
    upstream = np.ones_like(x_data)
    for name in ELEMENTWISE:
        activation = activations.activation_dictionary[name]
        delta = activations.derivative_dictionary[name](
            activation(x_data), x_data, upstream
        )
        assert delta.dtype == np.float32, f"{name} promoted to {delta.dtype}"


def test_sigmoid_is_numerically_stable_at_extremes():
    extreme = np.array([[-800.0, 0.0, 800.0]])
    probabilities = activations.sigmoid(extreme)
    assert np.isfinite(probabilities).all()
    assert probabilities[0, 0] == pytest.approx(0.0)
    assert probabilities[0, 1] == pytest.approx(0.5)
    assert probabilities[0, 2] == pytest.approx(1.0)


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(2)
    probabilities = activations.softmax(rng.normal(size=(7, 5)) * 10)
    assert np.allclose(probabilities.sum(axis=-1), 1.0)
    assert (probabilities >= 0).all()


# -------------    modReLU, the complex activation    --------------
def test_mod_relu_preserves_phase():
    """modrelu scales magnitude and leaves the direction alone"""
    z_values = np.array([1.0 + 1.0j, -2.0 + 0.5j])
    activated = activations.mod_relu(z_values, 0.5)
    assert np.allclose(np.angle(activated), np.angle(z_values))


def test_mod_relu_zeroes_below_the_bias():
    z_values = np.array([0.1 + 0.0j])
    assert abs(activations.mod_relu(z_values, -1.0)[0]) == pytest.approx(0.0)


@pytest.mark.slow
def test_mod_relu_derivative_matches_finite_differences():
    """
    Both returned gradients are checked. d_bias was the one that silently
    dropped its activity mask, giving a 62 percent error on the learnable
    per-frequency bias SPECTRE trains.
    """
    from conftest import numeric_gradient_complex

    rng = np.random.default_rng(3)
    z_values = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    bias = rng.normal(size=5) * 0.5
    upstream = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))

    def scalar():
        activated = activations.mod_relu(z_values, bias)
        return float(
            (activated.real * upstream.real).sum()
            + (activated.imag * upstream.imag).sum()
        )

    analytic_bias, analytic_z = activations.mod_relu_derivative(
        z_values, bias, upstream
    )

    assert relative_error(analytic_bias, numeric_gradient(scalar, bias)) < 1e-6
    assert relative_error(analytic_z, numeric_gradient_complex(scalar, z_values)) < 1e-6


def test_derivative_registry_naming_convention():
    """the decorator strips the _derivative suffix, so the keys must line up"""
    for name, function in activations.derivative_dictionary.items():
        assert function.__name__ == f"{name}_derivative"
