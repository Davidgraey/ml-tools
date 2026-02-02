import numpy as np
from numpy.typing import NDArray
from typing import Iterable
from ml_tools.models.layers.layers import Layer


class NeuralNetwork:
    def __init__(self, layers: Iterable[Layer]):
        self.layers: dict[int, Layer] = {
            l_idx: l_obj
            for l_idx, l_obj in enumerate(layers)
        }

        # flip around for easier backpropigation
        self.inverse_layers: dict[int, Layer] = dict(
            sorted(self.layers.items(), reverse=True)
        )

    def forward(self, x_data: NDArray):
        _x = x_data.copy()
        for l_idx, layer_object in self.layers.items():
            _x = layer_object.forward(_x)
        return _x

    def backward(self, incoming_gradient: NDArray):
        _grad = incoming_gradient.copy()
        for l_idx, layer_object in self.inverse_layers.items():
            _grad = layer_object.forward(_grad)
        return _grad

