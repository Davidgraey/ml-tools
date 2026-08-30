import numpy as np
from numpy.typing import NDArray
from typing import Optional
from ml_tools.models.layers.layers import ANY_SHAPE, Layer


class LatentStack(Layer):
    """
    stack via latent dimension (-1) position
    """

    def __init__(self):
        super().__init__()
        self.declare_shapes(inputs=(ANY_SHAPE, ANY_SHAPE), outputs=(ANY_SHAPE,))
        self.split_dims = ()

    def infer_output_shapes(self, input_shapes: tuple[tuple, ...]) -> tuple[tuple, ...]:
        """
        The output width is the sum of the two incoming widths, so it is only
        knowable once they are. Adding it here rather than declaring it keeps
        the width alive for whatever consumes the merge -- otherwise every
        layer downstream of a concatenation goes unchecked.
        """
        widths = [shape[-1] for shape in input_shapes]
        if any(width is None for width in widths):
            return (ANY_SHAPE,)
        return ((sum(widths),),)

    def forward(self, array_a: NDArray, array_b: NDArray) -> NDArray:
        self.a_shape = array_a.shape
        self.b_shape = array_b.shape
        assert self.a_shape[0] == self.b_shape[0]

        self.split_dims = (self.a_shape[-1], self.b_shape[-1])

        if array_a.ndim == 1:
            return np.hstack((array_a.ravel(), array_b.ravel()))
        elif array_a.ndim == 2:
            return np.hstack((array_a, array_b))
        elif array_a.ndim == 3:
            return np.dstack((array_a, array_b))

    def backward(self, incoming_gradient: NDArray) -> tuple[NDArray, NDArray]:
        """
        # apportion the gradients to their correct inputs ("slices")
        Parameters
        ----------
        incoming_gradient : backpassed grad

        Returns
        -------
        the "split" gradients -> ordered in the same fashion as the inputs to the forward pass kwards.
        """
        a_dim, b_dim = self.split_dims

        grad_a = incoming_gradient[..., :a_dim]
        grad_b = incoming_gradient[..., a_dim:a_dim + b_dim]

        return grad_a, grad_b

    def purge(self):
        self.split_dims, self.a_shape, self.b_shape = None, None, None

    def update_weights(self, **kwargs) -> None:
        pass

    def zero_gradients(self) -> None:
        pass


class LatentSum(Layer):
    """
    sum two arrays together, element-wise

    """
    def __init__(self, broadcast_axis: Optional[int] = None):
        super().__init__()
        self.input_1: Optional[NDArray] = None
        self.input_2: Optional[NDArray] = None
        self.output: Optional[NDArray] = None

        self.broadcast_axis = broadcast_axis
        self._expanded_dim = None


        # Gradients calculated during backward pass
        self.gradient_input_1: Optional[NDArray] = None
        self.gradient_input_2: Optional[NDArray] = None

        self.is_output: bool = False

        self.declare_shapes(inputs=(ANY_SHAPE, ANY_SHAPE), outputs=(ANY_SHAPE,))


    @staticmethod
    def _reduce_to_shape(grad: NDArray, target_shape: tuple) -> NDArray:
        """ sum the gradients down to target_shape wherever we have to broadcas t"""
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def forward(self,
                input_1: NDArray,
                input_2: NDArray) -> NDArray:
        """
        Sums input_1 and input_2 element-wise.
        """
        self.in_shape_1 = input_1.shape
        self.in_shape_2 = input_2.shape
        ndim1, ndim2 = len(self.in_shape_1), len(self.in_shape_2)
        axis = self.broadcast_axis if self.broadcast_axis is not None else 1

        try:
            if ndim1 < ndim2:
                sum_array = np.expand_dims(input_1, axis) + input_2
                self._expanded_dim = (1, axis)
            elif ndim1 > ndim2:
                sum_array = input_1 + np.expand_dims(input_2, axis)
                self._expanded_dim = (2, axis)
            else:
                sum_array = input_1 + input_2
                self._expanded_dim = None
        except ValueError as e:
            raise RuntimeError(f"SummingLayer: Inputs not broadcastable. {e}")

        self.input_1 = input_1
        self.input_2 = input_2
        self.output = sum_array

        return self.output

    def backward(
            self, incoming_grad: NDArray
    ) -> tuple[NDArray, NDArray]:
        """
        Backward pass. Calculates gradients for input_1 and input_2.

        Parameters
        ----------
        incoming_grad : NDArray
            Gradient of the loss with respect to the output of this layer.

        Returns
        -------
        Tuple[NDArray, NDArray]
        """
        if self._expanded_dim is not None:
            which, axis = self._expanded_dim
            if which == 1:
                grad_1 = incoming_grad.sum(axis=axis)
                grad_2 = incoming_grad
            else:
                grad_1 = incoming_grad
                grad_2 = incoming_grad.sum(axis=axis)
        else:
            grad_1 = incoming_grad
            grad_2 = incoming_grad

        self.gradient_input_1 = self._reduce_to_shape(grad_1, self.in_shape_1)
        self.gradient_input_2 = self._reduce_to_shape(grad_2, self.in_shape_2)

        return self.gradient_input_1, self.gradient_input_2

    def update_weights(self, **kwargs) -> None:
        """
        Pass through -- no weights to update
        """
        pass

    def purge(self) -> None:
        self.input_1 = None
        self.input_2 = None
        self.output = None
        self.gradient_input_1 = None
        self.gradient_input_2 = None
        self.gradient = None
        self._expanded_dim = None

    def get_weights(self):
        return None

    def get_gradients(self) -> dict[str, NDArray]:
        """
        Returns a dictionary of gradients.
        For this layer, returns gradients of the inputs.
        """
        grads = {}
        if self.gradient_input_1 is not None:
            grads["grad_input_1"] = self.gradient_input_1
        if self.gradient_input_2 is not None:
            grads["grad_input_2"] = self.gradient_input_2
        return grads

    def zero_gradients(self) -> None:
        """
        Zeroes the stored gradient values.
        """
        self.gradient_input_1 = np.zeros_like(self.input_1)
        self.gradient_input_2 = np.zeros_like(self.input_2)
        self.gradient = None

    @property
    def num_parameters(self) -> int:
        return 0

    def __str__(self):
        return "SummingLayer"

    def __repr__(self):
        return f"{self}"


class ShiftRight(Layer):
    """
    Shift a sequence right by one position along the sequence axis, so
    position t sees position t-1's value instead of its own -- the standard
    teacher-forcing input for an autoregressive decoder.
    """

    preserves_shape = True

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.start_token: NDArray = np.zeros((1, 1, hidden_dim))
        self.declare_shapes(inputs=((hidden_dim,),), outputs=((hidden_dim,),))
        self.zero_gradients()

    def forward(self, input_data: NDArray) -> NDArray:
        """
        input_data : (batch, sequence, hidden)
        """
        assert input_data.ndim == 3, (
            f"expected (batch, sequence, hidden), got shape {input_data.shape}"
        )
        assert input_data.shape[-1] == self.hidden_dim, (
            f"built for hidden_dim {self.hidden_dim}, got {input_data.shape[-1]}"
        )
        batch = input_data.shape[0]
        start = np.broadcast_to(self.start_token, (batch, 1, self.hidden_dim))
        self.output = np.concatenate([start, input_data[:, :-1, :]], axis=1)
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        self.gradient_start_token = incoming_gradient[:, :1, :].sum(
            axis=0, keepdims=True
        )
        # -1?
        grad_input = np.zeros_like(incoming_gradient)
        grad_input[:, :-1, :] = incoming_gradient[:, 1:, :]
        return grad_input

    def update_weights(self, gradient_start_token: NDArray) -> None:
        self.start_token -= gradient_start_token

    def zero_gradients(self) -> None:
        self.gradient_start_token = np.zeros_like(self.start_token)

    def get_weights(self):
        return self.start_token

    def get_gradients(self) -> dict[str, NDArray]:
        return {"gradient_start_token": self.gradient_start_token}

    def purge(self) -> None:
        self.output = None

    @property
    def num_parameters(self) -> int:
        return self.start_token.size

    def __str__(self):
        return f"ShiftRight, hidden {self.hidden_dim}"

    def __repr__(self):
        return self.__str__()