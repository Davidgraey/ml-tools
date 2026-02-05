import numpy as np
from numpy.typing import NDArray
from ml_tools.models.layers.layers import Layer


class LatentStack(Layer):
    """
    stack via latent dimension (-1) position
    """

    def __init__(self):
        self.shapes = ()

    def forward(self, array_a: NDArray, array_b: NDArray) -> NDArray:
        self.a_shape = array_a.shape
        self.b_shape = array_b.shape
        assert self.a_shape[0] == self.b_shape[0]

        self.shapes = (self.a_shape[-1], self.b_shape[-1])

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
        a_dim, b_dim = self.shapes

        grad_a = incoming_gradient[..., :a_dim]
        grad_b = incoming_gradient[..., a_dim:a_dim + b_dim]

        return grad_a, grad_b

    def purge(self):
        self.shapes, self.a_shape, self.b_shape = None, None, None

    def update_weights(self, **kwargs) -> None:
        pass

    def zero_gradients(self) -> None:
        pass


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
