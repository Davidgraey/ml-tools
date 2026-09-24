"""
Optimizers and the parameter-update contract.
"""

import numpy as np
import pytest
from ml_tools.models.layers.fft_layers import FourierAttention
from ml_tools.models.layers.basal_layers import (
    DropoutLayer,
    FullyConnectedLayer,
    NormalizeLayer,
    RMSNormLayer,
)
from ml_tools.models.model_loss import MSELoss
from ml_tools.models.neural_network import NeuralNetwork
from ml_tools.models.optimizers import SGD, Optimizer

PARAMETERISED_LAYERS = (
    lambda: FullyConnectedLayer(4, 3, "relu"),
    lambda: NormalizeLayer(4, shift_scale=True),
    lambda: RMSNormLayer(4),
)


def test_optimizer_is_abstract():
    with pytest.raises(TypeError):
        Optimizer()


@pytest.mark.parametrize("make_layer", PARAMETERISED_LAYERS)
def test_gradient_keys_match_update_signature(make_layer, small_matrix):
    """every key get_gradients emits must be a parameter update_weights takes"""
    import inspect

    layer = make_layer()
    output = layer.forward(small_matrix)
    layer.backward(np.ones_like(output))

    accepted = set(inspect.signature(layer.update_weights).parameters)
    emitted = set(layer.get_gradients())
    assert emitted <= accepted, f"{emitted - accepted} not accepted by update_weights"


@pytest.mark.parametrize("make_layer", PARAMETERISED_LAYERS)
def test_sgd_changes_parameters(make_layer, small_matrix):
    layer = make_layer()
    output = layer.forward(small_matrix)
    layer.backward(np.ones_like(output))

    before = {
        name: np.array(getattr(layer, name))
        for name in ("weights", "scale_gamma", "shift_beta")
        if isinstance(getattr(layer, name, None), np.ndarray)
    }
    SGD(0.1).step([layer])

    for name, original in before.items():
        assert not np.allclose(original, getattr(layer, name)), f"{name} unchanged"


def test_sgd_scales_by_the_learning_rate(small_matrix):
    """a step at rate r must move exactly r times the raw gradient"""
    layer = FullyConnectedLayer(4, 3, "linear")
    output = layer.forward(small_matrix)
    layer.backward(np.ones_like(output))

    gradient = layer.gradient_weights.copy()
    before = layer.weights.copy()
    SGD(0.05).step([layer])

    assert np.allclose(before - 0.05 * gradient, layer.weights)


def test_sgd_skips_parameterless_layers(small_matrix):
    """an empty or None gradient dict must not raise"""
    layers = [DropoutLayer(0.1), NormalizeLayer(4, shift_scale=False)]
    for layer in layers:
        output = layer.forward(small_matrix)
        layer.backward(np.ones_like(output))
    SGD(0.1).step(layers)


def test_zero_gradients_clears_every_layer(small_matrix):
    layer = FullyConnectedLayer(4, 3, "relu")
    output = layer.forward(small_matrix)
    layer.backward(np.ones_like(output))
    assert layer.gradient_weights.any()

    SGD(0.1).zero_gradients([layer])
    assert not layer.gradient_weights.any()


def test_sgd_handles_one_level_of_block_nesting(sequence_batch):
    """a block returns a dict of dicts, which step() flattens by one level"""
    block = FourierAttention(ni=4, no=4)
    output = block.forward(sequence_batch)
    block.backward(np.ones_like(output))
    SGD(0.01).step([block])


@pytest.mark.xfail(
    reason="step() scales one level of nesting, so a block containing a block "
    "produces a dict of dicts of dicts and the multiply fails",
    strict=True,
)
def test_sgd_handles_nested_blocks(sequence_batch):
    class Nested(FourierAttention):
        def __init__(self, width):
            super().__init__(width, width)
            self.inner = FourierAttention(width, width)

        def forward(self, x_data):
            return super().forward(self.inner.forward(x_data))

        def backward(self, gradient):
            return self.inner.backward(super().backward(gradient))

        def get_gradients(self):
            gradients = super().get_gradients()
            gradients["inner"] = self.inner.get_gradients()
            return gradients

    block = Nested(4)
    output = block.forward(sequence_batch)
    block.backward(np.ones_like(output))
    SGD(0.01).step([block])


# -------------    end to end descent    ---------------------------

def test_network_reduces_regression_loss(regression_dataset):
    x_data, y_data, _ = regression_dataset
    y_data = y_data.reshape(-1, 1)

    layers = [
        FullyConnectedLayer(3, 16, "relu"),
        NormalizeLayer(16, shift_scale=True),
        FullyConnectedLayer(16, 16, "swish"),
        FullyConnectedLayer(16, 1, "linear", is_output=True),
    ]
    network = NeuralNetwork(layers)
    loss = MSELoss()
    optimizer = SGD(0.01)

    first = None
    for _ in range(200):
        prediction = network.forward(x_data)
        value = loss(prediction, y_data)
        if first is None:
            first = value
        network.backward(loss.backward())
        optimizer.step(layers)

    assert value < first * 0.9, f"expected real progress, got {first} -> {value}"



def test_classifier_learns_a_separable_problem(multiclass_dataset):
    from ml_tools.generators.data_generators import to_onehot
    from ml_tools.models.constants import ClassificationTask
    from ml_tools.models.model_loss import CrossEntropyLoss

    x_data, y_data, _ = multiclass_dataset
    targets = to_onehot(y_data, 4)

    layers = [
        FullyConnectedLayer(5, 24, "relu"),
        FullyConnectedLayer(24, 4, "linear", is_output=True),
    ]
    network = NeuralNetwork(layers)
    loss = CrossEntropyLoss(ClassificationTask.MULTINOMIAL)
    optimizer = SGD(5.0)

    for _ in range(400):
        prediction = network.forward(x_data)
        loss(prediction, targets)
        network.backward(loss.backward())
        optimizer.step(layers)

    accuracy = (prediction.argmax(-1) == y_data).mean()
    assert accuracy > 0.6, f"accuracy {accuracy:.3f} is near chance"


@pytest.mark.slow
def test_multilabel_classifier_learns(multilabel_dataset):
    """
    multilabel_dataset was never wired into a real NeuralNetwork + optimizer
    loop -- test_supervised.py covers GradientDescent on it, this covers the
    layer/CrossEntropyLoss(MULTILABEL) path the same way the multinomial case
    above covers its own loss branch.
    """
    from ml_tools.models.activations import sigmoid
    from ml_tools.models.constants import ClassificationTask
    from ml_tools.models.model_loss import CrossEntropyLoss

    x_data, y_data, _ = multilabel_dataset

    layers = [
        FullyConnectedLayer(4, 24, "relu"),
        FullyConnectedLayer(24, 4, "linear", is_output=True),
    ]
    network = NeuralNetwork(layers)
    loss = CrossEntropyLoss(ClassificationTask.MULTILABEL)
    optimizer = SGD(2.0)

    for _ in range(400):
        prediction = network.forward(x_data)
        loss(prediction, y_data)
        network.backward(loss.backward())
        optimizer.step(layers)

    predicted = (sigmoid(prediction) >= 0.5).astype(int)
    accuracy = (predicted == y_data).mean()
    assert accuracy > 0.7, f"per-label accuracy {accuracy:.3f} is near chance"


def test_neural_network_orders_the_backward_pass(small_matrix):
    """layers must run in reverse for backward, forward order for forward"""
    layers = [
        FullyConnectedLayer(4, 6, "linear"),
        FullyConnectedLayer(6, 2, "linear"),
    ]
    network = NeuralNetwork(layers)
    output = network.forward(small_matrix)
    assert output.shape == (6, 2)
    assert network.backward(np.ones_like(output)).shape == small_matrix.shape
