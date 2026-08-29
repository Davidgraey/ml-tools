import numpy as np
from numpy.typing import NDArray
from typing import Optional

from ml_tools.models.layers.layers import EPSILON, FullyConnectedLayer, Layer


class VotingBase(Layer):
    """
    Shared plumbing for the mixture voting layers --
    """

    # softmax votes already sum to one, so a top-k mask has to be renormalised
    # to stay a distribution. Independent sigmoid votes never summed to one and
    # must not be.
    renormalize: bool = False
    activation: str = "linear"

    def __init__(self, input_shape: int, num_experts: int, top_k: Optional[int] = None):
        super().__init__()
        assert top_k is None or 1 <= top_k <= num_experts, (
            f"top_k must fall in [1, {num_experts}], or be None for a dense "
            f"vote, got {top_k}"
        )
        assert not (self.renormalize and top_k == 1), (
            "top_k=1 on a renormalised vote is untrainable: the one surviving "
            "vote is divided by itself, so the layer emits a constant one and "
            "its gradient is exactly zero. Use top_k >= 2, or a VotingWeight, "
            "whose votes are independent and so are never renormalised."
        )
        self.input_shape = input_shape
        self.num_experts = num_experts
        self.top_k = top_k
        # the two stacked branches of the embed stage, one vote per expert out
        self.declare_shapes(
            inputs=((2, input_shape),), outputs=((num_experts,),)
        )

        self.stack: tuple[FullyConnectedLayer, ...] = ()
        self.in_shape = None
        self.mask = None
        self.route_sum = None
        self.output = None

    def forward(self, incoming_x: NDArray) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : (num_samples, 2, hidden_space), or anything that flattens
            to rows of 2 * input_shape

        Returns
        -------
        one vote per expert, (num_samples, num_experts)
        """
        self.in_shape = incoming_x.shape
        assert incoming_x.size % (2 * self.input_shape) == 0, (
            f"cannot read {self.in_shape} as rows of 2 * {self.input_shape}"
        )

        votes = incoming_x.reshape(-1, 2 * self.input_shape)
        for layer in self.stack:
            votes = layer(votes)

        self.output = self._route(votes)
        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        grad = self._route_backward(incoming_grad)
        for layer in self.stack[::-1]:
            grad = layer.backward(grad)

        return grad.reshape(self.in_shape)

    def _route(self, votes: NDArray) -> NDArray:
        """
        Optional sparse routing: only the top_k votes survive. argpartition
        keeps exactly k even when votes tie, which comparing against the k-th
        value would not.
        """
        if self.top_k is None:
            return votes

        keep = np.argpartition(votes, -self.top_k, axis=-1)[..., -self.top_k :]
        self.mask = np.zeros_like(votes, dtype=bool)
        np.put_along_axis(self.mask, keep, True, axis=-1)

        kept = votes * self.mask
        if not self.renormalize:
            return kept

        self.route_sum = np.sum(kept, axis=-1, keepdims=True) + EPSILON
        return kept / self.route_sum

    def _route_backward(self, incoming_grad: NDArray) -> NDArray:
        """
        VJP of _route. The mask is piecewise constant so it passes straight
        through; the renormalisation couples the surviving experts and takes
        the same form as the softmax VJP.
        """
        if self.top_k is None:
            return incoming_grad
        if not self.renormalize:
            return incoming_grad * self.mask

        dot = np.sum(incoming_grad * self.output, axis=-1, keepdims=True)
        return self.mask * (incoming_grad - dot) / self.route_sum

    def _named_stack(self) -> dict[str, FullyConnectedLayer]:
        """keys the optimizer round-trips through get_gradients/update_weights"""
        return {f"fc_{n}": layer for n, layer in enumerate(self.stack, start=1)}

    def get_weights(self) -> tuple[NDArray, ...]:
        return tuple(layer.get_weights() for layer in self.stack)

    def get_gradients(self) -> dict[str, dict[str, NDArray]]:
        return {
            name: layer.get_gradients() for name, layer in self._named_stack().items()
        }

    def update_weights(self, **gradients: dict[str, NDArray]) -> None:
        for name, layer in self._named_stack().items():
            layer.update_weights(**gradients[name])

    def zero_gradients(self) -> None:
        for layer in self.stack:
            layer.zero_gradients()

    def purge(self) -> None:
        for layer in self.stack:
            layer.purge()
        self.in_shape = None
        self.mask = None
        self.route_sum = None
        self.output = None

    @property
    def num_parameters(self) -> int:
        return sum(layer.num_parameters for layer in self.stack)

    def __str__(self):
        routing = f"top {self.top_k} of" if self.top_k else "dense over"
        return (
            f"{type(self).__name__}, {self.activation} {routing} "
            f"{self.num_experts} experts on 2 x {self.input_shape}"
        )

    def __repr__(self):
        return self.__str__()


class VotingWeight(VotingBase):
    def __init__(self, input_shape: int, num_experts: int, top_k: Optional[int] = None):
        """
        Independent per-expert weights in (0, 1), a single projection off the
        stacked embed output.

        Sigmoid rather than linear: a linear head cannot hold its output in
        (0, 1), and clipping one would kill the gradient outside the range.

        The votes do not compete. Each expert is judged on its own, so the
        weights need not sum to one and any number of them can sit near one at
        once. top_k therefore masks without renormalising, since renormalising
        would reimpose the sum-to-one that the sigmoid deliberately drops.

        Parameters
        ----------
        input_shape : hidden width of one branch of the embed stage output.
            The layer reads 2 * input_shape, both branches stacked.
        num_experts : number of experts to weight. Forward returns one weight
            per expert, and the network computes SUM(weight * expert_output).
        top_k : keep only the k largest weights, zeroing the rest. None keeps
            every expert.
        """
        super().__init__(input_shape, num_experts, top_k)
        self.activation = "sigmoid"
        self.stack = (
            FullyConnectedLayer(
                ni=2 * input_shape,
                no=num_experts,
                activation_type="sigmoid",
                is_output=True,
            ),
        )


class VotingGate(VotingBase):
    """determines the weighting of each expert in the final output"""

    renormalize = True

    def __init__(
        self,
        input_shape: int,
        hidden_size: int,
        num_experts: int,
        top_k: Optional[int] = None,
    ):
        """
        Softmax voting gate for the Fourier net.

        Two layers rather than one: the gate has to decide which expert suits a
        given input, and a single linear-plus-softmax head can only carve the
        stacked embedding into linear half-spaces. The relu hidden layer buys it
        a non-linear boundary.

        Unlike VotingWeight the experts compete here. The softmax makes the
        votes a distribution over experts, so raising one lowers the others.

        Parameters
        ----------
        input_shape : hidden width of one branch of the embed stage output.
            The layer reads 2 * input_shape, both branches stacked.
        hidden_size : width of the relu hidden layer.
        num_experts : number of experts to vote over. The softmaxed votes are
            used as weights, SUM(softmax(vote) * expert_output).
        top_k : keep only the k largest votes and renormalise them back to a
            distribution. None keeps every expert.
        """
        super().__init__(input_shape, num_experts, top_k)
        self.activation = "softmax"
        self.hidden_size = hidden_size
        self.stack = (
            FullyConnectedLayer(
                ni=2 * input_shape, no=hidden_size, activation_type="relu"
            ),
            FullyConnectedLayer(
                ni=hidden_size,
                no=num_experts,
                activation_type="softmax",
                is_output=True,
            ),
        )


class PoolingLayer(Layer):
    """
    Mean pooling layer to compute the average across the sequence dimension.

    Input shape: (batch, sequence, hidden)
    Output shape: (batch, 1, hidden)
    """
    preserves_shape = False

    def __init__(self):
        super().__init__()
        # Declare expected input/output shapes for the framework's pipeline
        self.declare_shapes(
            inputs=((None, None, None),),
            outputs=((None, 1, None),)
        )

    def forward(self, incoming_x: NDArray) -> NDArray:
        self.input = incoming_x  # Store for backward pass (matches framework pattern)
        # Average across axis 1 (sequence dimension), keepdims to preserve structure
        return incoming_x.mean(axis=1)

    def backward(self, incoming_grad: NDArray) -> NDArray:
        if self.input is None:
            return incoming_grad

        seq_len = self.input.shape[1]
        # Gradient of a mean operation is the incoming gradient divided by sequence length.
        return incoming_grad / seq_len

    def update_weights(self) -> None:
        pass

    def zero_gradients(self) -> None:
        pass

    def get_weights(self):
        return None

    def get_gradients(self) -> dict[str, NDArray]:
        return {}

    def purge(self):
        pass

    @property
    def num_parameters(self) -> int:
        return 0

    def __str__(self):
        return "Layer of Sequence Mean Pooling (avg over sequence)"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x = rng.normal(size=(5, 2, 6))
    upstream = rng.normal(size=(5, 4))

    for gate in (
        VotingWeight(input_shape=6, num_experts=4),
        VotingWeight(input_shape=6, num_experts=4, top_k=2),
        VotingGate(input_shape=6, hidden_size=8, num_experts=4),
        VotingGate(input_shape=6, hidden_size=8, num_experts=4, top_k=2),
    ):
        for layer in gate.stack:
            layer.weights = layer.weights.astype(np.float64)
            layer.bias = layer.bias.astype(np.float64)

        votes = gate.forward(x)
        analytic = gate.backward(upstream.copy())

        numeric = np.zeros_like(x)
        for index in np.ndindex(x.shape):
            original = x[index]
            x[index] = original + 1e-6
            plus = (gate.forward(x) * upstream).sum()
            x[index] = original - 1e-6
            minus = (gate.forward(x) * upstream).sum()
            x[index] = original
            numeric[index] = (plus - minus) / 2e-6

        print(
            f"{gate} | {gate.num_parameters} params | vote sums "
            f"{np.round(votes.sum(axis=-1), 3)} | input grad error "
            f"{np.abs(analytic - numeric).max():.2e}"
        )
