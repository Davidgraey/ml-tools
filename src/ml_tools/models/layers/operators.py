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
    def __init__(self):
        super().__init__()
        self.input_1: Optional[NDArray] = None
        self.input_2: Optional[NDArray] = None
        self.output: Optional[NDArray] = None

        # Gradients calculated during backward pass
        self.gradient_input_1: Optional[NDArray] = None
        self.gradient_input_2: Optional[NDArray] = None

        self.is_output: bool = False

        self.declare_shapes(inputs=(ANY_SHAPE, ANY_SHAPE), outputs=(ANY_SHAPE,))

    def forward(self,
                input_1: NDArray,
                input_2: NDArray) -> NDArray:
        """
        Sums input_1 and input_2 element-wise.
        """
        self.in_shape_1 = input_1.shape
        self.in_shape_2 = input_2.shape

        # Assert that input shapes are broadcastable
        try:
            sum_array = input_1 + input_2
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
        # Derivative of sum function is 1 for all inputs.
        # Therefore, gradients flow back unchanged (identity).
        self.gradient_input_1 = incoming_grad
        self.gradient_input_2 = incoming_grad

        return self.gradient_input_1, self.gradient_input_2

    def update_weights(self, **kwargs) -> None:
        """
        Pass through -- no weights to update
        """
        pass

    def purge(self) -> None:
        """
        Resets all layer state.
        """
        self.input_1 = None
        self.input_2 = None
        self.output = None
        self.gradient_input_1 = None
        self.gradient_input_2 = None
        self.gradient = None

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




if __name__ == "__main__":
    a1 = np.array([1, 2, 3, 4, 5, 6])
    b1 = np.array([6, 5, 4, 3, 2, 1])
    a2 = np.array([[1, 2, 3], [4, 5, 6]])
    b2 = np.array([[6, 5, 4], [3, 2, 1]])
    a3 = np.array([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])
    b3 = np.array([[[6], [5], [4]], [[3], [2], [1]], [[0], [-1], [-2]]])

    stacker = LatentStack()

    outs = stacker.forward(a1, b1)
    [print(t.shape) for t in outs]
    print(f"1D output: {outs.shape}")

    stacker.purge()
    outs = stacker.forward(a2, b2)
    [print(t.shape) for t in outs]
    print(f"2D output: {outs.shape}")

    stacker.purge()
    outs = stacker.forward(a3, b3)
    [print(t.shape) for t in outs]
    print(f"3D output: {outs.shape}")
