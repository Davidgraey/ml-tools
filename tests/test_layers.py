from ml_tools.models.layers.layers import *
from ml_tools.models.model_loss import MSELoss
from ml_tools.models.optimizers import SGD

from conftest import regression_dataset


def test_layers(regression_dataset):
    x_regression, y_regression, meta = regression_dataset
    fc = FullyConnectedLayer(ni=3, no=10, activation_type="relu")
    nc1 = NormalizeLayer(ni=10, shift_scale=False)
    dp = DropoutLayer(dropout_prob=0.05)
    fc2 = FullyConnectedLayer(ni=10, no=10, activation_type="linear", is_output=False)
    nc2 = NormalizeLayer(ni=10, shift_scale=True)
    fc3 = FullyConnectedLayer(ni=10, no=10, activation_type="relu", is_output=False)
    fc4 = FullyConnectedLayer(ni=10, no=10, activation_type="sigmoid", is_output=False)
    fc5 = FullyConnectedLayer(ni=10, no=10, activation_type="relu_leaky", is_output=False)
    fc6 = FullyConnectedLayer(ni=10, no=1, activation_type="tanh", is_output=True)
    lossfc = MSELoss()
    optimizer = SGD(0.002)
    all_losses = []
    y_regression = y_regression.reshape(-1, 1)
    nnet_layers = [fc, nc1, dp, fc2, nc2, fc3, fc4, fc5, fc6]

    for _ in range(50):
        output = x_regression.copy()
        for layer in nnet_layers:
            output = layer.forward(output)
            print(f"forward pass: {layer} outs shaped {output.shape}")

        loss = lossfc(output, y_regression)
        all_losses.append(loss)
        grad_output = lossfc.backward()

        for layer in nnet_layers[::-1]:
            grad_output = layer.backward(grad_output)
            print(f"Backwards pass: {layer} gradients shaped {grad_output.shape}")
        optimizer.step(layers=nnet_layers)

