import numpy as np
import ml_tools.models.activations as activations
import copy
from numpy.typing import NDArray
from abc import ABC, abstractmethod

EPSILON = 1e-14
# TODO: set the global float depth --
GLOBAL_DTYPE = np.float16


# -------------    weight initilization functions    ---------------
# ------------------------------------------------------------------
def xavier(ni: int, no: int) -> NDArray:
    return np.random.normal(loc=0.0, scale=1 / np.sqrt(ni), size=(ni + 1, no)).astype(
        dtype=GLOBAL_DTYPE
    )


def kaiming(ni: int, no: int) -> NDArray:
    """weight init function for linear / relu functions with zero bias"""
    w = np.random.normal(loc=0.0, scale=np.sqrt(2 / ni), size=(ni + 1, no)).astype(
        dtype=GLOBAL_DTYPE
    )
    w[0] = 0
    return w


weight_init = {
    "linear": kaiming,
    "relu": kaiming,
    "relu_leaky": kaiming,
    "sigmoid": xavier,
    "tanh": xavier,
    "softmax": xavier,
}

# ------------------------------------------------------------------


class Layer(ABC):
    def __init__(self):
        super().__init__()
        self.RNG = np.random.RandomState(42)

    @abstractmethod
    def forward(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def backward(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def update_weight(self):
        pass

    @abstractmethod
    def purge(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# TODO: build ENUMS fro activations
class FullyConnectedLayer(Layer):
    def __init__(self, ni: int, no: int, activation_type, is_output=False):
        """**********ARGUMENTS**********
        :param ni: number of input units
        :param no: number of output units
        :param activation_type: string isdentifying activation type, 'linear', 'sigmoid', 'tanh', etc.
        :param is_output: boolean flag designating if this is an output layer or hidden layer
        """
        super().__init__()
        self.activation = activation_type
        self._func_activation = activations.activation_dictionary[self.activation]
        self._func_derivative = activations.derivative_dictionary[self.activation]

        self.is_output = is_output
        self.weights = weight_init[activation_type](ni, no)
        self.shape = self.weights.shape

        # these values will be rewritten or updated on each pass
        self.output = np.empty(shape=(ni, no)).astype(dtype=GLOBAL_DTYPE)
        self.input = np.empty(shape=(1, ni)).astype(dtype=GLOBAL_DTYPE)
        self.gradient = np.empty(shape=(1, no)).astype(dtype=GLOBAL_DTYPE)

    def forward(self, incoming_x: NDArray, forced_activation=False) -> NDArray:
        """**********ARGUMENTS**********
        :param incoming_x: input data that is already standardized, if called for
        **********RETURNS**********
        :return: product of forward pass of incoming values
        """

        self.input = copy.copy(incoming_x)

        # bias units are self.weights[0:1, :]
        if forced_activation != False:
            outs = activations.activation_dictionary[forced_activation](
                incoming_x @ self.weights[1:, :] + self.weights[0:1, :]
            )
        else:
            outs = self._func_activation(
                incoming_x @ self.weights[1:, :] + self.weights[0:1, :]
            )
            self.output = outs

        return outs

    def backward(self, incoming_delta: NDArray) -> NDArray:
        """**********ARGUMENTS**********
        :param incoming_gradient: delta from previous step of backprop
        **********RETURNS**********
        :return: returns this layer's gradient contribution
        """

        activated_delta = incoming_delta * self._func_derivative(
            self.output, self.input
        )

        this_delta = self.input.T @ activated_delta
        bias_delta = np.sum(activated_delta, 0)
        self.gradient = np.vstack(
            (bias_delta, this_delta)
        )  # make gradient persistent in layer

        grad_contribution = incoming_delta @ self.weights[1:, ...].T

        # self.update_weight(self.gradient)

        return grad_contribution

    def update_weight(self, value: NDArray) -> None:
        """**********ARGUMENTS**********
        :param value: variable to update this layer's weights by with - this will already have Learning rate,
        depreication or momentum / other calculations addressed in the upper level.
        """
        self.weights += value
        pass

    def purge(self) -> None:
        """
        resets values tracked during training to zero - an inplace func
        """
        self.input = np.empty(shape=(1, 1)).astype(dtype=GLOBAL_DTYPE)
        self.output = np.empty(shape=(1, 1)).astype(dtype=GLOBAL_DTYPE)
        self.gradient = np.empty(shape=(1, 1)).astype(dtype=GLOBAL_DTYPE)

    def __str__(self):
        return f"Layer of {self.activation}, shaped {self.shape} -- output is {self.is_output}"

    def __repr__(self):
        return f"{self.shape}, using {self.activation}"


# TODO: FUTURE - add below layer types : As we add more layer types, we will have to change our cascade assembly
# construction slightly to better differentiate type.


class DropoutLayer(Layer):
    """Hinton style Dropout layer with scaling outputs"""

    def __init__(self, ni, no, dropout_prob=0.5):
        super().__init__()
        assert (dropout_prob > 0.0) and (dropout_prob < 1.0)
        self.ni = ni
        self.no = no
        self.shape = [self.ni, self.no]
        self.dropout_prob: float = dropout_prob
        self.keep_prob: float = 1 - dropout_prob

        self.generate_dropout()

        self.input = [0]
        self.output = [0]

    def generate_dropout(self) -> None:
        """here, weights"""
        self.weights = (
            self.RNG.binomial([np.ones((self.ni, self.no))], self.keep_prob)
            / self.keep_prob
        )
        # TODO: yes, this could loop infinitely. come back to this.
        if self.weights.sum() == 0.0:
            self.generate_dropout()

    def set_dropout_value(self, value):
        self.dropout_prob = value
        return

    def forward(
        self, incoming_x: NDArray, training_now: bool = True, forced_activation=False
    ):
        self.input = copy.copy(incoming_x)
        if training_now:
            self.generate_dropout()
            self.output = self.weights * incoming_x

        else:
            self.output = self.input

        return self.output

    def backward(self, incoming_delta):
        incoming_delta *= self.weights
        incoming_delta = self.input.T * incoming_delta

        self.gradient = incoming_delta

        return self.gradient

    def update_weight(self) -> None:
        pass

    def purge(self) -> None:
        """
        resets values tracked during training to zero - an inplace func
        """
        self.input = np.empty(shape=(1, 1)).astype(dtype=GLOBAL_DTYPE)
        self.output = np.empty(shape=(1, 1)).astype(dtype=GLOBAL_DTYPE)
        self.gradient = np.empty(shape=(1, 1)).astype(dtype=GLOBAL_DTYPE)

    def __str__(self):
        return f"Layer of Hinton Dropout currently at {self.dropout_prob}, shaped {self.shape}"

    def __repr__(self):
        return f"{self.shape}, using Hinton Dropout"


# numpy-ml ref:
# https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/layers/layers.py#L1634-L1803
class NormalizeLayer(Layer):
    def __init__(self, ni: int, shift_scale: bool = False):
        super().__init__()
        self.ni = ni
        self.shift_scale = shift_scale

        self.num_samples = 0
        # TODO: allow setting dtypes
        self.running_mean = np.empty(shape=(1, ni)).astype(dtype=GLOBAL_DTYPE)
        self.running_stds = np.empty(shape=(1, ni)).astype(dtype=GLOBAL_DTYPE)

        if self.shift_scale:
            self.shift_beta = np.zeros(shape=(1, ni)).astype(dtype=GLOBAL_DTYPE)
            self.scale_gamma = np.ones(shape=(1, ni)).astype(dtype=GLOBAL_DTYPE)
        else:
            self.shift_beta = None
            self.scale_gamma = None

    def forward(self, incoming_x: NDArray) -> NDArray:
        self.input = copy.copy(incoming_x)
        num_samples, _ = incoming_x.shape

        # TODO: update if > 2D
        _mean = np.mean(incoming_x, axis=0)
        _stds = np.std(incoming_x, axis=0)

        new_total = self.num_samples + num_samples
        w_old = self.num_samples / new_total
        w_new = num_samples / new_total

        # weight this batch's mean / stds and the prior by their sample size
        self._mu = (w_old * self.running_mean) + (w_new * _mean)
        self._theta = (w_old * self.running_stds) + (w_new * _stds)
        self._n = new_total

        self.x_norm = (incoming_x - self._mu) / (self._theta + EPSILON)

        if self.shift_scale is None:
            self.output = self.x_norm
        else:
            self.output = self.scale_gamma * (self.x_norm) + self.shift_beta

        return self.output

    def update_weight(self, delta_beta: NDArray, delta_gamma: NDArray) -> None:
        # persist the means, stds, and update the samples seen
        self.running_mean = self._mu
        self.running_stds = self._theta
        self.num_samples = self._n

        # update the shift & scale values based on gradient contributions
        self.scale_gamma += delta_gamma
        self.shift_beta += delta_beta

    def backward(self, incoming_delta: NDArray) -> NDArray:
        delta_beta = np.sum(incoming_delta, axis=0)
        delta_gamma = np.sum(incoming_delta * self.x_norm, axis=0)

        delta_norm = incoming_delta * self.scale_gamma

        grad_contribution = (
            self.ni * delta_norm
            - np.sum(delta_norm, axis=0, keepdims=True)  # stepped back through mean
            - self.x_norm * np.sum(delta_norm * self.x_norm, axis=0, keepdims=True)
        ) / (self.ni * self.running_stds + +EPSILON)

        # update trainable parameters -- shift & scale / beta and gamma
        # self.update_weight(delta_beta, delta_gamma)

        return grad_contribution

    def purge(self):
        self.input = np.empty(shape=(1, 1)).astype(dtype=GLOBAL_DTYPE)
        self.output = np.empty(shape=(1, 1)).astype(dtype=GLOBAL_DTYPE)
        self.gradient = np.empty(shape=(1, 1)).astype(dtype=GLOBAL_DTYPE)


class EmbeddingLayer(Layer):
    def __init__(
        self, ni: int, cardinality: int, embedding_dim: int, trainable: bool = True
    ):
        super().__init__()
        self.ni: int = ni
        self.cardinality: int = cardinality
        self.embedding_dim: int = embedding_dim

        self.projection = self.RNG.uniform(size=(cardinality, embedding_dim))
        self.gradient = np.zeros_like(self.projection)

    def forward(self, incoming_x: NDArray) -> NDArray:
        if incoming_x.dtype != np.int:
            _x = incoming_x.astype(int)
            
        assert _x.dtype == np.int_
        self.input = copy.copy(_x)

        # straight indexing
        outs = self.projection[_x]
        self.output = outs

        return outs

    def backward(self, incoming_delta: NDArray) -> NDArray:

        self.gradient = np.zeros_like(self.projection)

        # inplace operation - "assign" the gradients to their input variable
        np.add.at(self.gradient, self.input, incoming_delta)

        # self.update_weight(self.gradient)
        return self.input

    def update_weight(self, value: NDArray) -> None:

        self.projection += value

    @property
    def weights(self) -> NDArray:
        """alias"""
        return self.projection


if __name__ == "__main__":
    x = np.array([[1, 2, 3], [3, 5, 6], [7, 8, 9], [9, 8, 7], [100, 200, 300]])
    # y = np.random.normal(loc=x)

    fc = FullyConnectedLayer(ni=3, no=1, activation_type="linear", is_output=True)
    assert fc(x).shape == (5, 1)

    norm = NormalizeLayer(ni=3, shift_scale=True)
    assert norm(x).shape == (5, 3)

    dp = DropoutLayer(ni=5, no=3, dropout_prob=0.25)
    assert dp(x, training_now=True) == (5, 3)
    fc.forward(x)

    assert (fc(dp(norm(x))).shape == (5, 1))
