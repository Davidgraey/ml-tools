import numpy as np
from numpy.typing import NDArray
from typing import Iterable
from ml_tools.models.layers.layers import Layer
from dataclasses import dataclass, field


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
            _grad = layer_object.backward(_grad)
            # try:
            #     _grad = layer_object.backward(_grad)
            # except:
            #     print(f"error in backprob during {layer_object} -> incoming shape {_grad.shape}")
        return _grad

    def compile(self):

        indegree = {
            layer: len(layer.upstream)
            for layer in self.layers
        }

        queue = [
            layer
            for layer, degree in indegree.items()
            if degree == 0
        ]

        self.forward_order = []

        while queue:

            layer = queue.pop(0)

            self.forward_order.append(layer)

            for child in layer.downstream:

                indegree[child] -= 1

                if indegree[child] == 0:
                    queue.append(child)

        self.backward_order = list(
            reversed(self.forward_order)
        )


################################################################################
################################################################################
################################################################################
################################################################################

@dataclass(slots=True)
class GradFlow:

    layer: Layer

    output: np.ndarray | None = None

    incoming_grads: dict[int, list[np.ndarray]] = field(default_factory=dict)

    executed: bool = False

def forward(self, x: np.ndarray) -> np.ndarray:

    flow = {
        layer: GradFlow(layer)
        for layer in self.layers
    }

    flow[self.input_layer].output = x

    for layer in self.forward_order:

        if layer is self.input_layer:
            continue

        inputs = [
            flow[parent].output
            for parent in layer.upstream
        ]

        if len(inputs) == 1:
            inputs = inputs[0]
        else:
            inputs = tuple(inputs)

        flow[layer].output = layer.forward(inputs)

    self.flow = flow

    return flow[self.output_layer].output

def forward(self, x: np.ndarray) -> np.ndarray:

    flow = {
        layer: GradFlow(layer)
        for layer in self.layers
    }

    flow[self.input_layer].output = x

    for layer in self.forward_order:

        if layer is self.input_layer:
            continue

        inputs = [
            flow[parent].output
            for parent in layer.upstream
        ]

        if len(inputs) == 1:
            inputs = inputs[0]
        else:
            inputs = tuple(inputs)

        flow[layer].output = layer.forward(inputs)

    self.flow = flow

    return flow[self.output_layer].output