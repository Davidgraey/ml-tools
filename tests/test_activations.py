"""
Activations: every derivative is the VJP of its activation.
"""

import numpy as np
import pytest
from conftest import (
    GRADIENT_TOLERANCE,
    numeric_gradient,
    numeric_gradient_complex,
    relative_error,
)
from polyergalio.models import activations

ELEMENTWISE = ("linear", "relu", "relu_leaky", "sigmoid", "tanh", "swish")


def test_every_activation_has_a_derivative():
    missing = set(activations.activation_dictionary) - set(
        activations.derivative_dictionary
    )
    assert not missing, f"activations without a registered derivative: {missing}"


@pytest.mark.parametrize("name", ELEMENTWISE + ("softmax",))
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


def test_mod_relu_derivative_matches_finite_differences():
    """both the complex input gradient and the per-frequency bias gradient"""
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


def test_sigmoid_is_numerically_stable_at_extremes():
    extreme = np.array([[-800.0, 0.0, 800.0]])
    probabilities = activations.sigmoid(extreme)
    assert np.isfinite(probabilities).all()
    assert probabilities[0, 0] == pytest.approx(0.0)
    assert probabilities[0, 1] == pytest.approx(0.5)
    assert probabilities[0, 2] == pytest.approx(1.0)
