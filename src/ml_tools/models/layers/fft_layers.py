import numpy as np
from ml_tools.models.layers import Layer
import ml_tools.models.activations as activations
from ml_tools.models.constants import GLOBAL_DTYPE, EPSILON, ANY_SHAPE
from numpy.typing import NDArray


def hartley(x_array: NDArray, axis: int = -1) -> NDArray:
    x_freq = np.fft.fft(x_array, axis=axis)
    return x_freq.real - x_freq.imag


def hartley_2d(x_array: NDArray, axes: tuple = (-2, -1)) -> NDArray:
    x_freq = np.fft.fft2(x_array, axes=axes)
    return x_freq.real - x_freq.imag

class FrequencyFFT(Layer):
    def __init__(self, max_sequence_length: int, window_size: int):
        """
        Seting up a process for FFT transformations of windows of audio data - expecting data of 1 size batch,
        Preprocessing assumed: sliding windows or patches.
        shape -> (number_of_windows, samples per window)

        We utilize a blackamn kernel to "ease in and out" -- frequency jumps from non-zero starts can give artifacts. Common for FFT / DFT application

        The transform is a discrete Hartley, not Re(fft). Re(fft) is blind to
        the circularly-odd part of the window, so phase information is lost
        and distinct windows collapse to the same output. Hartley is full rank
        and stays in real arithmetic, at the cost of a window_size wide output
        rather than window_size // 2 + 1.
        Parameters
        ----------
        max_sequence_length : the MAXIMUM supported sequence length - (windows)
        window_size : the number of samples in a window
        """
        super().__init__()
        self.max_sequence_length: int = max_sequence_length
        self.window_size: int = window_size
        self.window_kernel: NDArray = np.blackman(window_size)
        # (windows, samples per window), the window count left free since the
        # constructor only fixes its ceiling
        self.declare_shapes(
            inputs=((window_size,),), outputs=((window_size,),)
        )

    def forward(self, incoming_x: NDArray) -> NDArray:
        """
        Forward process for the Hartley transform
        Parameters
        ----------
        incoming_x : Numpy array of (number_of_windows, samples per window)

        Returns
        -------
        the Hartley transform of windowed_data, same width as the window
        """
        self.in_shape = incoming_x.shape

        assert self.in_shape[0] <= self.max_sequence_length, f"Shapes don't match in {self}"
        assert self.in_shape[-1] == self.window_size, (
            f"last axis must equal window_size {self.window_size}, "
            f"got {self.in_shape[-1]}"
        )

        self.input = incoming_x.reshape(-1, self.in_shape[-1])
        self.output = hartley(self.window_kernel * self.input, axis=-1)

        return self.output.reshape(self.in_shape)

    def backward(self, incoming_grad: NDArray) -> NDArray:
        """
        Hartley is its own adjoint, so the backward pass is the same
        transform followed by the same window weighting.
        """
        _grad = incoming_grad.reshape(-1, self.in_shape[-1])

        grad = self.window_kernel * hartley(_grad, axis=-1)
        return grad.reshape(self.in_shape)

    def purge(self) -> None:
        self.input = None
        self.output = None

    def get_weights(self, for_serialize: bool = False) -> NDArray:
        if for_serialize:
            return {}
        return None

    def set_weights(self, weights: dict) -> None:
        pass

    def get_gradients(self) -> dict[str, NDArray]:
        return {}

    def zero_gradients(self) -> None:
        pass

    def update_weights(self, **kwargs) -> None:
        pass

    @property
    def num_parameters(self) -> int:
        return 0


class FourierLayer(Layer):
    #  https://ieeexplore.ieee.org/document/9616294
    preserves_shape = True

    def __init__(self, use_2d:bool = True):
        super().__init__()
        self.use_2d = use_2d

        if use_2d == True:
            self.fft_axes = (-2, -1)
        else:
            self.fft_axes = -1

    def forward(self, incoming_x: NDArray) -> NDArray:
        """
        Hartley transform
        """
        self.input = incoming_x

        if not self.use_2d:
            self.output = hartley(incoming_x, axis=self.fft_axes)

        elif self.use_2d:
            self.output = hartley_2d(incoming_x, axes=self.fft_axes)

        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        """Hartley is symmetric, so the adjoint is the same transform."""
        if not self.use_2d:
            self.gradient = hartley(incoming_grad, axis=self.fft_axes)
        elif self.use_2d:
            self.gradient = hartley_2d(incoming_grad, axes=self.fft_axes)

        return self.gradient

    def purge(self) -> None:
        self.input = None
        self.output = None
        self.gradient = None

    def get_weights(self, for_serialize: bool = False) -> NDArray:
        if for_serialize:
            return {}
        return None

    def set_weights(self, weights: dict) -> None:
        pass

    def get_gradients(self) -> dict[str, NDArray]:
        return {} # TODO: return? self.gradient

    def zero_gradients(self) -> None:
        pass

    def update_weights(self, **kwargs) -> None:
        pass

    @property
    def num_parameters(self) -> int:
        return 0


class InverseFourierLayer(Layer):
    # https://arxiv.org/pdf/2502.18394
    preserves_shape = True

    def __init__(self, use_2d:bool = True):
        super().__init__()
        self.use_2d = use_2d

        if use_2d == True:
            self.fft_axes = (-2, -1)
        else:
            self.fft_axes = -1

    def forward(self, incoming_x: NDArray) -> NDArray:
        """
        Hartley is its own inverse up to 1/N, so the inverse direction is
        the same transform carrying that scale. Stacking this on top of
        FourierLayer reconstructs the input exactly.
        """
        self.input = incoming_x
        self.scale = self._scale(incoming_x.shape)

        if not self.use_2d:
            self.output = hartley(incoming_x, axis=self.fft_axes) / self.scale

        elif self.use_2d:
            self.output = hartley_2d(incoming_x, axes=self.fft_axes) / self.scale

        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        """Symmetric transform, so the adjoint carries the same 1/N."""
        if not self.use_2d:
            self.gradient = hartley(incoming_grad, axis=self.fft_axes) / self.scale
        elif self.use_2d:
            self.gradient = hartley_2d(incoming_grad, axes=self.fft_axes) / self.scale

        return self.gradient

    def _scale(self, shape: tuple) -> int:
        if self.use_2d:
            return shape[-2] * shape[-1]
        return shape[-1]

    def purge(self) -> None:
        self.input = None
        self.output = None
        self.gradient = None

    def get_weights(self, for_serialize: bool = False) -> NDArray:
        if for_serialize:
            return {}
        return None

    def set_weights(self, weights: dict) -> None:
        pass

    def get_gradients(self) -> dict[str, NDArray]:
        return {} # TODO: return? self.gradient

    def zero_gradients(self) -> None:
        pass

    def update_weights(self, **kwargs) -> None:
        pass

    @property
    def num_parameters(self) -> int:
        return 0