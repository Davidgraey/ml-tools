from typing import Optional

import numpy as np
from ml_tools.models.layers.layers import EPSILON, FullyConnectedLayer, Layer
from numpy.typing import NDArray


class VotingBase(Layer):
    renormalize: bool = False
    activation: str = "linear"

    def __init__(
        self,
        input_shape: int,
        num_experts: int,
        top_k: Optional[int] = None,
        bias_update_speed: float = 0.0,
    ):
        """
        Parameters
        ----------
        input_shape : width of the incoming hidden state, e.g. hidden_dim
        num_experts : one vote per expert out
        top_k : keep only the k largest votes, or None for a dense vote
        bias_update_speed : per-forward-pass adjustment to a routing bias added to each expert's vote before top-k
            SELECTION to order -- auxiliary-loss-free load-balancing mechanism
        """
        super().__init__()
        assert top_k is None or 1 <= top_k <= num_experts, (
            f"top_k must fall in [1, {num_experts}], or be None for a dense "
            f"vote, got {top_k}"
        )
        assert not (self.renormalize and top_k == 1), (
            "top_k=1 on a renormalised vote is untrainable, increase topk"
        )
        self.input_shape = input_shape
        self.num_experts = num_experts
        self.top_k = top_k

        # a slowly-adapting correction, not a per-batch scratch value. this is the auxiliary load-balancing approach
        self.bias_update_speed = bias_update_speed

        
        self.expert_bias = np.zeros(num_experts)
        # (input_shape,) > (num_experts,) layer
        self.declare_shapes(
            inputs=((input_shape,),), outputs=((num_experts,),)
        )

        self.stack: tuple[FullyConnectedLayer, ...] = ()
        self.in_shape = None
        self.mask = None
        self.token_mask = None
        self.route_sum = None
        self.output = None

        self.zero_gradients()

    def forward(self,
                incoming_x: NDArray,
                training_now: bool = True,
                mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : our incoming data, (..., input_shape) -- any number of leading batch/ sequence axes
        training_now : whether the expert-load bias is updated. False (inference) leaves expert_bias untouched
        mask : (...,) matching incoming_x's leading axes, 1 for a real token and 0 for padding

        Returns
        -------
        one vote per expert, (num_samples, num_experts)
        """
        self.in_shape = incoming_x.shape
        assert self.in_shape[-1] == self.input_shape, (
            f"cannot read trailing axis {self.in_shape[-1]} as input_shape "
            f"{self.input_shape}"
        )

        votes = incoming_x.reshape(-1, self.input_shape)
        for layer in self.stack:
            votes = layer(votes)

        if mask is not None:
            assert mask.shape == self.in_shape[:-1], (
                f"mask shape {mask.shape} must match incoming_x's leading "
                f"axes {self.in_shape[:-1]}"
            )
        self.token_mask = None if mask is None else mask.reshape(-1)

        self.output = self._route(votes, training_now=training_now)
        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        grad = self._route_backward(incoming_grad)
        for layer in self.stack[::-1]:
            grad = layer.backward(grad)

        return grad.reshape(self.in_shape)

    def _route(self, votes: NDArray, training_now: bool = True) -> NDArray:
        """
        Core Routing Process

        final selection considers votes + expert_bias.
        bias steers which experts fire without entering the combined output or its gradient

        Parameters
        ----------
        votes: the output / forward pass
        training_now: bool - if we're training vs inference (bias factored into the update or not)

        Returns
        -------

        """
        if self.top_k is None:
            return votes

        selection_scores = votes + self.expert_bias if self.bias_update_speed else votes
        keep = np.argpartition(selection_scores, -self.top_k, axis=-1)[..., -self.top_k :]
        self.mask = np.zeros_like(votes, dtype=bool)
        np.put_along_axis(self.mask, keep, True, axis=-1)

        if training_now and self.bias_update_speed:
            self.update_expert_bias()

        kept = votes * self.mask
        if not self.renormalize:
            return kept

        self.route_sum = np.sum(kept, axis=-1, keepdims=True) + EPSILON
        return kept / self.route_sum

    def update_expert_bias(self) -> None:
        """
        auxiliary-loss-free balancing update, similar to Deepseek v3 (entirely separate from the gradient)
        it's a load-balancing term to increase less-seen experts, while decreasing over-seen experts.

        padded rows (masks in Forward) are left out of the load average, otherwise they'd count as real usage
        """
        if self.token_mask is None:
            load = self.mask.mean(axis=0)
        else:
            valid = self.token_mask.astype(self.mask.dtype)[:, None]
            valid_count = np.maximum(valid.sum(), 1.0)
            load = (self.mask * valid).sum(axis=0) / valid_count

        fair_share = self.top_k / self.num_experts
        self.expert_bias += self.bias_update_speed * np.sign(fair_share - load)

    def _route_backward(self, incoming_grad: NDArray) -> NDArray:
        """
        backpass through the route
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

    def get_weights(self, for_serialize: bool = False):
        named = self._named_stack()
        if for_serialize:
            weights = {name: layer.get_weights(for_serialize=True) for name, layer in named.items()}
            weights["expert_bias"] = self.expert_bias
            return weights
        return tuple(layer.get_weights(for_serialize=False) for layer in self.stack) + (self.expert_bias,)

    def set_weights(self, weights: dict) -> None:
        if not weights:
            return
        for name, layer in self._named_stack().items():
            if name in weights:
                layer.set_weights(weights[name])
        if "expert_bias" in weights:
            self.expert_bias = np.asarray(weights["expert_bias"])

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
        self.token_mask = None
        self.route_sum = None
        self.output = None

    @property
    def num_parameters(self) -> int:
        return sum(layer.num_parameters for layer in self.stack)

    def __str__(self):
        routing = f"top {self.top_k} of" if self.top_k else "dense over"
        return (
            f"{type(self).__name__}, {self.activation} {routing} "
            f"{self.num_experts} experts on input width {self.input_shape}"
        )

    def __repr__(self):
        return self.__str__()


class VotingWeight(VotingBase):
    def __init__(
        self,
        input_shape: int,
        num_experts: int,
        top_k: Optional[int] = None,
        bias_update_speed: float = 0.0,
    ):
        """
        Independent per-expert weights between 0 and 1 as a single projection

        The votes do not compete. Each expert's activation is judged by the voting weights on its own, so the weights
        don't have to sum to 1 / be normalized. (independent consideration per expert)

        Outputs are weighted multiplicative elements. (expert 0 should be weighted (mult) by W[0] value

        Parameters
        ----------
        input_shape : hidden width of the incoming hidden state.
        num_experts : number of experts to weight. Forward returns one weight per expert
        top_k : keep only the k largest weights, zeroing the rest. None keeps
            every expert.
        bias_update_speed : see VotingBase class for more - this is the load-balancing mechanism
        """
        super().__init__(input_shape, num_experts, top_k, bias_update_speed=bias_update_speed)
        self.activation = "sigmoid"
        self.stack = (
            FullyConnectedLayer(
                ni=input_shape,
                no=num_experts,
                activation_type="sigmoid",
                is_output=True,
            ),
        )

        self.zero_gradients()


class VotingWeightBalanced(VotingBase):
    """determines the weighting of each expert in the final output"""

    renormalize = True

    def __init__(
        self,
        input_shape: int,
        hidden_size: int,
        num_experts: int,
        top_k: Optional[int] = None,
        gate_activation: str = "softmax",
        bias_update_speed: float = 0.0,
    ):
        """
        Unlike VotingWeight the experts compete here: top_k always re-norms; so raising one vote's weights lowers anothers

        Parameters
        ----------
        input_shape : hidden width of the incoming hidden state.
        hidden_size : width of the relu hidden layer.
        num_experts : number of experts to vote over. The gated votes are used as weights, SUM(gate(vote) * expert_output)
        top_k : keep only the k largest votes and renormalise them back to a distribution None keeps every expert.
        gate_activation : final projection's activation
            "softmax" (default) makes every expert compete for one fixed budget of weight.
            "sigmoid" scores each expert independently before top-k selection and renorm
        bias_update_speed : see VotingBase. 0.0 (default) here too, so this stays a plain competitive gate
        """
        super().__init__(
            input_shape, num_experts, top_k,
            bias_update_speed=bias_update_speed,
        )
        self.activation = gate_activation
        self.gate_activation = gate_activation
        self.hidden_size = hidden_size
        self.stack = (
            FullyConnectedLayer(
                ni=input_shape, no=hidden_size, activation_type="relu"
            ),
            FullyConnectedLayer(
                ni=hidden_size,
                no=num_experts,
                activation_type=gate_activation,
                is_output=True,
            ),
        )


class VotingGate(VotingBase):
    """boolean pass/no-pass gate -- top_k experts fire at full strength, everyone else is off. No reweighting."""

    def __init__(
        self,
        input_shape: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        bias_update_speed: float = 0.0,
    ):
        """
        Parameters
        ----------
        input_shape : hidden width of the incoming hidden state.
        hidden_size : width of the relu hidden layer.
        num_experts : number of experts to gate.
        top_k : how many experts pass per sample. Required -- a dense (top_k=None) boolean gate has nothing to gate.
        bias_update_speed : see VotingBase.
        """
        assert top_k is not None, (
            "VotingGate requires top_k -- a boolean gate with no top_k has "
            "nothing to gate"
        )
        super().__init__(
            input_shape, num_experts, top_k,
            bias_update_speed=bias_update_speed,
        )
        self.activation = "sigmoid"
        self.hidden_size = hidden_size
        self.stack = (
            FullyConnectedLayer(
                ni=input_shape, no=hidden_size, activation_type="relu"
            ),
            FullyConnectedLayer(
                ni=hidden_size,
                no=num_experts,
                activation_type="sigmoid",
                is_output=True,
            ),
        )

    def _route(self, votes: NDArray, training_now: bool = True) -> NDArray:
        """
        Straight-through gate: forward is the pure top_k boolean mask, no vote
        magnitude passes through.

        Parameters
        ----------
        votes : the output / forward pass
        training_now : bool - if we're training vs inference (bias factored into the update or not)

        Returns
        -------
        """
        selection_scores = votes + self.expert_bias if self.bias_update_speed else votes
        keep = np.argpartition(selection_scores, -self.top_k, axis=-1)[..., -self.top_k :]
        self.mask = np.zeros_like(votes, dtype=bool)
        np.put_along_axis(self.mask, keep, True, axis=-1)

        if training_now and self.bias_update_speed:
            self.update_expert_bias()

        return self.mask.astype(votes.dtype)

    def _route_backward(self, incoming_grad: NDArray) -> NDArray:
        """
        Straight-through estimator: reuses the non-renormalised
        """
        return incoming_grad * self.mask


class MixtureOfExperts(Layer):
    """
    DeepSeek-style mixture-of-experts block: a fixed pool of shared experts, always summed into the output, plus a
    larger pool of routed experts of which only the gate's top_k highest-weighted fire per sample.

    This framework has no sparse dispatch -- every routed expert still runs every forward pass -- but the gate's top-k
    mask zeroes out the unchosen experts' contribution to both the output and the gradient, so training still only ever
    reinforces the chosen experts
    """

    def __init__(
        self,
        hidden_dim: int,
        num_shared_experts: int,
        num_routed_experts: int,
        top_k: int,
        gate_hidden: int,
        activation_type: str = "relu",
        gate_activation: str = "softmax",
        bias_update_speed: float = 1e-3,
    ):
        """
        Parameters
        ----------
        hidden_dim : width of the incoming/outgoing hidden state
        num_shared_experts : experts summed into every output
        num_routed_experts : size of the routed expert pool the gate chooses top_k from
        top_k : count of routed experts to fire, 2 <= top_k <= num_routed_experts. The gate renormalises its weights
        gate_hidden : width of the gate's relu hidden layer, see VotingWeightBalanced
        activation_type : activation for every expert FullyConnectedLayer
        gate_activation : the gate's final-projection activation
            "softmax" (default) makes the routed experts compete for a fixed budget of weight
            "sigmoid" scores each independently before top-k selection and renormalisation
        bias_update_speed : the load-balancing mechnism
        """
        super().__init__()
        assert num_shared_experts >= 0, "num_shared_experts must be >= 0"
        assert 2 <= top_k <= num_routed_experts, f"top_k must fall in [2, {num_routed_experts}], got {top_k} -- "

        self.hidden_dim = hidden_dim
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.gate_hidden = gate_hidden
        self.activation_type = activation_type
        self.gate_activation = gate_activation
        self.bias_update_speed = bias_update_speed

        self.declare_shapes(inputs=((hidden_dim,),), outputs=((hidden_dim,),))

        self.shared_experts = tuple(
            FullyConnectedLayer(ni=hidden_dim, no=hidden_dim, activation_type=activation_type)
            for _ in range(num_shared_experts)
        )
        self.routed_experts = tuple(
            FullyConnectedLayer(ni=hidden_dim, no=hidden_dim, activation_type=activation_type)
            for _ in range(num_routed_experts)
        )

        self.gate = VotingWeightBalanced(
            input_shape=hidden_dim,
            hidden_size=gate_hidden,
            num_experts=num_routed_experts,
            top_k=top_k,
            gate_activation=gate_activation,
            bias_update_speed=bias_update_speed,
        )

        self._leading_shape = None
        self._routed_out = None
        self._gate_weights = None
        self.output = None

        self.zero_gradients()

    def forward(self,
                hidden_state: NDArray,
                training_now: bool = True,
                mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        hidden_state : (..., hidden_dim), any number of leading batch/sequence
            axes -- routing runs independently per trailing-axis row
        training_now : whether the gate's load-balancing bias (see the class
            docstring) is allowed to update from this pass. False (e.g.
            validation or inference) leaves it untouched.
        mask : (...,) matching hidden_state's leading axes, 1 for a real token
            and 0 for padding. Every expert still runs on every row
            regardless -- only the gate's load-balancing bias (see
            VotingBase._update_expert_bias) needs to know which rows are real,
            so this is passed straight through to it.

        Returns
        -------
        (..., hidden_dim): every shared expert's output, plus the top_k
        routed experts' output weighted by the gate, summed.
        """
        self._leading_shape = hidden_state.shape[:-1]

        shared_out = [expert(hidden_state) for expert in self.shared_experts]
        shared_sum = sum(shared_out) if shared_out else np.zeros_like(hidden_state)

        self._routed_out = np.stack([expert(hidden_state) for expert in self.routed_experts], axis=-2)

        gate_votes = self.gate(hidden_state, training_now=training_now, mask=mask)
        self._gate_weights = gate_votes.reshape(*self._leading_shape, self.num_routed_experts)

        weighted_routed = np.sum(self._routed_out * self._gate_weights[..., np.newaxis], axis=-2)

        self.output = shared_sum + weighted_routed
        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        grad_hidden = np.zeros(self._leading_shape + (self.hidden_dim,))

        for expert in self.shared_experts:
            grad_hidden = grad_hidden + expert.backward(incoming_grad)

        gate_grad = np.zeros(self._leading_shape + (self.num_routed_experts,))
        for e, expert in enumerate(self.routed_experts):
            expert_grad_out = incoming_grad * self._gate_weights[..., e : e + 1]
            grad_hidden = grad_hidden + expert.backward(expert_grad_out)
            gate_grad[..., e] = np.sum(incoming_grad * self._routed_out[..., e, :], axis=-1)

        grad_hidden = grad_hidden + self.gate.backward(
            gate_grad.reshape(-1, self.num_routed_experts)
        )

        return grad_hidden

    def _named_sublayers(self) -> dict[str, Layer]:
        """keys the optimizer round-trips through get_gradients/update_weights"""
        named = {f"shared_{n}": e for n, e in enumerate(self.shared_experts, start=1)}
        named.update({f"routed_{n}": e for n, e in enumerate(self.routed_experts, start=1)})
        named["gate"] = self.gate
        return named

    def get_weights(self, for_serialize: bool = False):
        named = self._named_sublayers()
        return {
            name: layer.get_weights(for_serialize=for_serialize)
            for name, layer in named.items()
        }

    def set_weights(self, weights: dict) -> None:
        if not weights:
            return
        for name, layer in self._named_sublayers().items():
            if name in weights:
                layer.set_weights(weights[name])

    def get_gradients(self) -> dict[str, dict]:
        return {name: layer.get_gradients() for name, layer in self._named_sublayers().items()}

    def update_weights(self, **gradients) -> None:
        named = self._named_sublayers()
        for name, layer in named.items():
            if name in gradients:
                layer.update_weights(**gradients[name])

    def zero_gradients(self) -> None:
        for layer in self._named_sublayers().values():
            layer.zero_gradients()

    def purge(self) -> None:
        for layer in self._named_sublayers().values():
            layer.purge()
        self._leading_shape = None
        self._routed_out = None
        self._gate_weights = None
        self.output = None

    @property
    def num_parameters(self) -> int:
        return sum(layer.num_parameters for layer in self._named_sublayers().values())

    def __str__(self):
        return (
            f"MixtureOfExperts, {self.num_shared_experts} shared + top "
            f"{self.top_k} of {self.num_routed_experts} routed experts, "
            f"hidden_dim {self.hidden_dim}"
        )

    def __repr__(self):
        return self.__str__()


class PoolingLayer(Layer):
    """
    Mean pooling layer to compute the average across the sequence dimension.

    Input shape: (batch, sequence, hidden)
    Output shape: (batch, 1, hidden)
    """
    preserves_shape = False

    def __init__(self):
        super().__init__()
        self.declare_shapes(
            inputs=((None, None, None),),
            outputs=((None, 1, None),)
        )

        self.input = None
        self.weights = None
        self.counts = None

        self.zero_gradients()

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        mask : (batch, sequence), 1 for a real position and 0 for padding
        """
        self.input = incoming_x
        if mask is None:
            self.weights = None
            self.counts = None
            return incoming_x.mean(axis=1, keepdims=True)

        assert mask.shape == incoming_x.shape[:2], (
            f"mask shape {mask.shape} must match incoming_x's (batch, "
            f"sequence) axes {incoming_x.shape[:2]}"
        )
        self.weights = mask[..., None].astype(incoming_x.dtype)
        self.counts = np.maximum(self.weights.sum(axis=1, keepdims=True), 1.0)
        return (incoming_x * self.weights).sum(axis=1, keepdims=True) / self.counts

    def backward(self, incoming_grad: NDArray) -> NDArray:
        if self.input is None:
            return incoming_grad

        if self.counts is None:
            seq_len = self.input.shape[1]
            return np.broadcast_to(incoming_grad / seq_len, self.input.shape).copy()

        return np.broadcast_to(incoming_grad / self.counts, self.input.shape) * self.weights

    def update_weights(self) -> None:
        pass

    def zero_gradients(self) -> None:
        pass

    def get_weights(self, for_serialize: bool = False):
        return {} if for_serialize else None

    def get_gradients(self) -> dict[str, NDArray]:
        return {}

    def purge(self):
        self.input = None
        self.weights = None
        self.counts = None

    @property
    def num_parameters(self) -> int:
        return 0

    def __str__(self):
        return "Layer of Sequence Mean Pooling (avg over sequence)"

    def __repr__(self):
        return self.__str__()
