import numpy as np
import ml_tools.models.activations as activations
from ml_tools.models.model_loss import rmse_derivative
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Optional, Callable

# Availability


EPSILON = 1e-14
# TODO: set the global float depth --
GLOBAL_DTYPE = np.float16


# -------------    weight initilization functions    ---------------
# ------------------------------------------------------------------
def xavier(rng, ni: int, no: int) -> NDArray:
    return rng.normal(loc=0.0, scale=1 / np.sqrt(ni), size=(ni, no)).astype(
        dtype=GLOBAL_DTYPE
    )


def kaiming(rng, ni: int, no: int) -> NDArray:
    """weight init function for linear / relu functions with zero bias"""
    w = rng.normal(loc=0.0, scale=np.sqrt(2 / ni), size=(ni, no)).astype(
        dtype=GLOBAL_DTYPE
    )
    return w


weight_init = {
    "linear": kaiming,
    "relu": kaiming,
    "relu_leaky": kaiming,
    "swish": kaiming,
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


# TODO: build ENUMS for activations
class FullyConnectedLayer(Layer):
    def __init__(self, ni: int, no: int, activation_type: str, is_output: bool = False):
        """
        **********ARGUMENTS**********
        :param ni: number of input units
        :param no: number of output units
        :param activation_type: string isdentifying activation type, 'linear', 'sigmoid', 'tanh', etc.
        :param is_output: boolean flag designating if this is an output layer or hidden layer
        """
        super().__init__()
        self.activation: str = activation_type
        self._func_activation: Callable = activations.activation_dictionary[
            self.activation
        ]
        self._func_derivative: Callable = activations.derivative_dictionary[
            self.activation
        ]

        self.is_output: bool = is_output
        self.weights: NDArray = weight_init[activation_type](self.RNG, ni=ni, no=no)
        self.shape: tuple = self.weights.shape
        self.bias: NDArray = np.zeros((1, no), dtype=GLOBAL_DTYPE)

        # these values will be rewritten or updated on each pass
        self.output = np.empty(shape=(ni, no))
        self.input = np.empty(shape=(1, ni))
        self.gradient = np.empty(shape=(1, no))

    def forward(
        self, incoming_x: NDArray, forced_activation: Optional[str] = None
    ) -> NDArray:
        """

        Parameters
        ----------
        incoming_x : input data that is already standardized, if called for
        forced_activation :

        Returns
        -------
        Dot Product of forward pass of incoming values
        """
        assert incoming_x.shape[-1] == self.weights.shape[0], (
            f"weights and xs don't match -- x:{incoming_x.shape} "
            f"weights: {self.weights.shape}"
        )
        self.input = incoming_x  # do we need a ref?
        self.z = incoming_x @ self.weights + self.bias

        if forced_activation is None:  # standard layer activation
            this_activation: Callable = self._func_activation
        else:  # we call out the specific "forced" activation
            this_activation: Callable = activations.activation_dictionary[
                forced_activation
            ]

        self._used_activation = forced_activation or self.activation
        self.output = this_activation(self.z)

        return self.output

    def backward(
        self, incoming_grad: NDArray, forced_activation: Optional[str] = None
    ) -> NDArray:
        """
        Backward pass through this layer and it's activation
        Parameters
        ----------
        incoming_grad : the backward passed gradient from the "previous" step

        Returns
        -------
        returns this layer's gradient contribution
        """
        if forced_activation is None:
            this_derivative: Callable = self._func_derivative
        else:
            this_derivative: Callable = activations.activation_dictionary[
                forced_activation
            ]
        # reshape to 2D in case (batch, sequence, hidden)
        _x = self.input.reshape(-1, self.shape[-1])
        _delta = incoming_grad.reshape(-1, incoming_grad.shape[-1])

        delta = incoming_grad * this_derivative(self.output, self.z)

        self.gradient_weights = self.input.T @ delta
        self.gradient_bias = delta.sum(axis=0, keepdims=True)

        return delta @ self.weights.T

    def update_weight(self, weights_delta: NDArray, bias_delta: NDArray) -> None:
        """**********ARGUMENTS**********
        :param value: variable to update this layer's weights by with - this will already have Learning rate,
        depreication or momentum / other calculations addressed in the upper level.
        """
        self.weights -= weights_delta
        self.bias -= bias_delta

        pass

    def purge(self) -> None:
        """
        resets values tracked during training to zero - an inplace func
        """
        self.input = None
        self.output = None
        self.z = None
        self.gradient_weights = None
        self.gradient_bias = None

    def __str__(self):
        return f"Layer of {self.activation}, shaped {self.shape} -- output is {self.is_output}"

    def __repr__(self):
        return f"{self.shape}, using {self.activation}"


# TODO: FUTURE - add below layer types : As we add more layer types, we will have to change our cascade assembly
# construction slightly to better differentiate type.
class DropoutLayer(Layer):
    """Hinton style Dropout layer with scaling outputs"""

    def __init__(self, dropout_prob=0.5):
        super().__init__()
        assert (dropout_prob > 0.0) and (dropout_prob < 1.0)
        self.dropout_prob: float = dropout_prob
        self.keep_prob: float = 1 - dropout_prob

        self.mask = None
        self.input = None
        self.output = None

    # def set_dropout_value(self, value):
    #     self.dropout_prob = value
    #     return

    def forward(
        self, incoming_x: NDArray, training_now: bool = True, forced_activation=False
    ):
        self.input = incoming_x.copy()
        if training_now:
            self.mask = (
                self.RNG.binomial(1, self.keep_prob, size=x.shape)
                / self.keep_prob
            )
            return x * self.mask

        else:
            self.output = self.input

        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        if self.mask is None:
            return incoming_grad
        return incoming_grad * self.mask

        return self.gradient

    def update_weight(self) -> None:
        pass

    def purge(self) -> None:
        """
        resets values tracked during training to zero - an inplace func
        """
        self.input = None
        self.mask = None
        self.gradient = np.empty(shape=(1, 1))

    def __str__(self):
        return f"Layer of Hinton Dropout currently at {self.dropout_prob}, shaped {self.shape}"

    def __repr__(self):
        return f"{self.shape}, using Hinton Dropout"


# numpy-ml ref:
# https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/layers/layers.py#L1634-L1803
class NormalizeLayer(Layer):
    def __init__(self, ni: int, shift_scale: bool = True, eps: float = 1e-5):
        super().__init__()
        self.ni = ni
        self.shift_scale = shift_scale
        self.eps = eps

        if shift_scale:
            self.scale_gamma = np.ones((1, ni))
            self.shift_beta = np.zeros((1, ni))
        else:
            self.gamma = None
            self.beta = None

        # cached for backward
        self.x_norm = None
        self.std = None

    def forward(self, incoming_x: NDArray) -> NDArray:
        self.input = incoming_x
        num_samples, _ = incoming_x.shape

        _mean = np.mean(incoming_x,  axis=-1, keepdims=True)
        self.std = np.std(incoming_x,  axis=-1, keepdims=True)
        self.x_norm = (x - _mean) / self.std

        if self.shift_scale == True:
            return self.scale_gamma * (self.x_norm) + self.shift_beta

        return self.x_norm

    def update_weight(self, delta_beta: NDArray, delta_gamma: NDArray) -> None:
        # update the shift & scale values based on gradient contributions
        self.scale_gamma += delta_gamma
        self.shift_beta += delta_beta

    def backward(self, incoming_grad: NDArray) -> NDArray:
        if self.shift_scale:
            self.grad_beta = np.sum(incoming_grad, axis=0)
            self.grad_gamma = np.sum(incoming_grad * self.x_norm, axis=0)
            z = incoming_grad * self.gamma
        else:
            z = incoming_grad


        gradient = (1.0 / self.std) * (
                z
                - np.mean(z, axis=-1)
                - self.x_norm * np.mean(z * self.x_norm, axis=-1)
        )

        return gradient

    def purge(self):
        self.input = np.empty(shape=(1, 1))
        self.output = np.empty(shape=(1, 1))
        self.gradient = np.empty(shape=(1, 1))


class FrequencyFFT(Layer):
    def __init__(self, max_sequence_length: int, window_size: int):
        """
        Seting up a process for FFT transformations of windows of audio data - expecting data of 1 size batch,
        Preprocessing assumed: sliding windows or patches.
        shape -> (number_of_windows, samples per window)
        Parameters
        ----------
        max_sequence_length : the MAXIMUM supported sequence length - (windows)
        window_size : the number of samples in a window
        """
        super().__init__()
        self.max_sequence_length: int = max_sequence_length
        self.window_size: int = window_size

    def forward(self, windowed_data: NDArray) -> NDArray:
        """
        Forward process for the FFT transform
        Parameters
        ----------
        windowed_data : Numpy array of (number_of_windows, samples per window)

        Returns
        -------
        the FFT transform of windowed_data
        """
        num_windows, samples = windowed_data.shape

        assert num_windows <= self.max_sequence_length, f"Shapes don't match in {self}"
        assert samples <= self.window_size, f"Shapes don't match in {self}"

        self.input = windowed_data.copy()
        self.output = np.fft.rfft(windowed_data, axis=-1).real

        return self.output.copy()

    def backward(self, incoming_grad: NDArray) -> NDArray:
        self.gradient = np.fft.irfft(incoming_grad, axis=-1).real.copy()

        self.update_weight(self.gradient)
        return self.gradient

    def purge(self) -> None:
        self.input = np.empty(shape=(1, 1))
        self.output = np.empty(shape=(1, 1))
        self.gradient = np.empty(shape=(1, 1))

    def update_weight(self, *args):
        pass


# ====== ==================================================================
# TODO: WIP below


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
        self.input = _x.copy()

        # straight indexing
        outs = self.projection[_x]
        self.output = outs

        return outs

    def backward(self, incoming_grad: NDArray) -> NDArray:
        self.gradient = np.zeros_like(self.projection)

        # inplace operation - "assign" the gradients to their input variable
        np.add.at(self.gradient, self.input, incoming_grad)

        # self.update_weight(self.gradient)
        return self.input

    def update_weight(self, value: NDArray) -> None:
        self.projection += value

    @property
    def weights(self) -> NDArray:
        """alias"""
        return self.projection




class FourierLayer1D(Layer):
    #  https://ieeexplore.ieee.org/document/9616294
    def __init__(
        self, ni: int, no: int, window_count: int, sequence_length: int, hidden_dim: int
    ):
        super().__init__()
        self.ni = ni
        self.no = no

        # self.positional_weights = self.RNG.uniform(size=(sequence_length))
        # self.frequency_weights = self.RNG.uniform(size=(hidden_dim))

    def forward(self, x_data: NDArray):
        x_freq = np.fft.fft2(x_data, axes=(-2, -1)).real
        # scale, constrain or norm?

        # x_out = np.fft.ifft2(reweighted, axes=(-2, -1)).real

        # return normed(x_data + x_out)

        # x = x @ W_linear

    def backward(self, incoming_grad):
        complex_delta = incoming_grad.astype(np.complex128)

        self.gradient = np.fft.ifft(complex_delta, axis=self.axis).real

        self.update_weights()


if __name__ == "__main__":
    x = np.array([[1, 2, 3], [3, 5, 6], [7, 8, 9], [9, 8, 7], [100, 200, 300]])
    # y = np.random.normal(loc=x)

    fc = FullyConnectedLayer(ni=3, no=1, activation_type="linear", is_output=True)
    assert fc(x).shape == (5, 1)

    norm = NormalizeLayer(ni=3, shift_scale=True)
    assert norm(x).shape == (5, 3)

    dp = DropoutLayer(dropout_prob=0.25)
    assert dp(x, training_now=True).shape == (5, 3)
    fc.forward(x)

    assert fc(dp(norm(x))).shape == (5, 1)

    def train():
        from ml_tools.models.model_loss import MSELoss
        LR = 0.002
        fc = FullyConnectedLayer(ni=3, no=1, activation_type="linear")
        lossfc = MSELoss()
        all_losses = []
        y_regression = y_regression.reshape(-1, 1)
        for _ in range(100):
            output = fc.forward(x_regression)
            loss = lossfc(output, y_regression)
            all_losses.append(loss)
            grad_output = lossfc.backward()

            x_backprop = fc.backward(grad_output)

            fc.update_weight(LR * fc.gradient_weights, LR * fc.gradient_bias)


