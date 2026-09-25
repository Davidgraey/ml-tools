"""
The NeuralNetwork graph container (DAG)
"""

import inspect

import numpy as np
import pytest
from conftest import GRADIENT_TOLERANCE, numeric_gradient, relative_error
from polyergalio.models.layers.basal_layers import (
    ANY_SHAPE,
    DropoutLayer,
    FullyConnectedLayer,
    Layer,
    NormalizeLayer,
    shape_conflict,
)
from polyergalio.models.layers.operator_layers import LatentStack
from polyergalio.models.layers.spectre_layers import PersistentMemory, SpectreAttention
from polyergalio.models.model_loss import MSELoss
from polyergalio.models.neural_network import INPUT_NAME, NeuralNetwork, Node
from polyergalio.models.optimizers import SGD


def as_float64(layer):
    for name in ("weights", "bias"):
        value = getattr(layer, name, None)
        if isinstance(value, np.ndarray):
            setattr(layer, name, value.astype(np.float64))
    return layer


class FixedWidthMerge(Layer):
    """
    A two-source merge that pins both widths.

    No shipped merge layer constrains its inputs -- LatentStack concatenates
    whatever it is handed -- so a positional shape conflict has to be built
    here to be tested at all.
    """

    def __init__(self, left: int, right: int):
        super().__init__()
        self.declare_shapes(inputs=((left,), (right,)), outputs=((left + right,),))

    def forward(self, left, right):
        return np.hstack([left, right])

    def backward(self, incoming_gradient):
        left = self.shapes["input"][0][-1]
        return incoming_gradient[..., :left], incoming_gradient[..., left:]

    def update_weights(self, **kwargs) -> None:
        pass

    def purge(self) -> None:
        pass

    def get_weights(self, for_serialize: bool = False):
        return {} if for_serialize else None

    def zero_gradients(self) -> None:
        pass


@pytest.fixture()
def branching_network():
    """
    input -> a, and a feeds both b and c, which merge into the output. Node `a`
    therefore receives gradient along two paths.
    """
    net = NeuralNetwork(name="branching")
    a = net.connect(as_float64(FullyConnectedLayer(4, 6, "tanh")), net.input, name="a")
    b = net.connect(as_float64(FullyConnectedLayer(6, 3, "swish")), a, name="b")
    c = net.connect(as_float64(FullyConnectedLayer(6, 5, "relu")), a, name="c")
    merged = net.connect(LatentStack(), b, c, name="merge")
    net.output = net.connect(
        as_float64(FullyConnectedLayer(8, 2, "linear")), merged, name="out"
    )
    return net


# -------------    connecting by reference    ----------------------
def test_connect_returns_a_node():
    net = NeuralNetwork()
    node = net.connect(FullyConnectedLayer(4, 4, "relu"), net.input)
    assert isinstance(node, Node)
    assert node.sources == (net.input,)


def test_the_input_is_a_real_node():
    net = NeuralNetwork()
    assert isinstance(net.input, Node)
    assert net.input.is_source
    assert net.input.name == INPUT_NAME


def test_a_reused_node_becomes_a_fan_out():
    """passing one node to two calls is what makes a branch visible"""
    net = NeuralNetwork()
    shared = net.connect(FullyConnectedLayer(4, 6, "relu"), net.input, name="shared")
    left = net.connect(FullyConnectedLayer(6, 2, "relu"), shared, name="left")
    right = net.connect(FullyConnectedLayer(6, 2, "relu"), shared, name="right")

    assert {node.name for node in shared.consumers} == {"left", "right"}
    assert left.sources == (shared,)
    assert right.sources == (shared,)


def test_a_layer_needs_at_least_one_source():
    net = NeuralNetwork()
    with pytest.raises(ValueError, match="at least one source"):
        net.connect(FullyConnectedLayer(4, 4, "relu"))


def test_a_string_is_not_accepted_as_a_source():
    """the whole point: sources are nodes, so a name cannot be mistaken for one"""
    net = NeuralNetwork()
    with pytest.raises(TypeError, match="expected a Node"):
        net.connect(FullyConnectedLayer(4, 4, "relu"), INPUT_NAME)


def test_a_node_from_another_network_is_rejected():
    first = NeuralNetwork()
    second = NeuralNetwork()
    alien = second.connect(FullyConnectedLayer(4, 4, "relu"), second.input)
    with pytest.raises(ValueError, match="different network"):
        first.connect(FullyConnectedLayer(4, 4, "relu"), alien)


def test_arity_is_checked_when_wiring():
    """a merge given the wrong source count fails here, not inside forward"""
    net = NeuralNetwork()
    a = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    b = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    c = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    with pytest.raises(ValueError, match="declares 2 input shapes"):
        net.connect(LatentStack(), a, b, c)


def test_duplicate_names_are_rejected():
    net = NeuralNetwork()
    net.connect(FullyConnectedLayer(4, 4, "relu"), net.input, name="shared")
    with pytest.raises(ValueError, match="already taken"):
        net.connect(FullyConnectedLayer(4, 4, "relu"), net.input, name="shared")


def test_names_are_generated_from_the_layer_class():
    net = NeuralNetwork()
    first = net.connect(FullyConnectedLayer(4, 4, "relu"), net.input)
    second = net.connect(FullyConnectedLayer(4, 4, "relu"), first)
    assert first.name == "FullyConnectedLayer_0"
    assert second.name == "FullyConnectedLayer_1"


# -------------    structural guarantees    ------------------------
def test_a_cycle_cannot_be_constructed():
    """
    An edge can only point at a node that already exists, so there is no
    forward reference available to close a loop with. Every source of every
    node must appear earlier in construction order.
    """
    net = NeuralNetwork()
    first = net.connect(FullyConnectedLayer(4, 4, "relu"), net.input)
    second = net.connect(FullyConnectedLayer(4, 4, "relu"), first)
    net.connect(FullyConnectedLayer(4, 4, "relu"), second)

    positions = {node: index for index, node in enumerate(net.nodes)}
    for node in net.nodes:
        for source in node.sources:
            assert positions[source] < positions[node]


def test_insertion_order_is_a_topological_order(branching_network):
    positions = {node: index for index, node in enumerate(branching_network.nodes)}
    for node in branching_network.nodes:
        for source in node.sources:
            assert positions[source] < positions[node]


def test_edges_report_every_connection(branching_network):
    edges = set(branching_network.edges())
    assert ("input", "a") in edges
    assert ("a", "b") in edges and ("a", "c") in edges
    assert ("b", "merge") in edges and ("c", "merge") in edges
    assert ("merge", "out") in edges


def test_validate_flags_a_node_that_feeds_nothing():
    net = NeuralNetwork()
    main = net.connect(FullyConnectedLayer(4, 3, "relu"), net.input, name="main")
    net.connect(FullyConnectedLayer(4, 8, "relu"), net.input, name="orphan")
    net.output = main

    problems = net.validate()
    assert any("orphan" in problem for problem in problems)


def test_validate_flags_a_registered_but_unconnected_layer():
    class Model(NeuralNetwork):
        def __init__(self):
            super().__init__()
            self.wired = FullyConnectedLayer(4, 4, "relu")
            self.stray = FullyConnectedLayer(4, 4, "relu")
            self.output = self.connect(self.wired, self.input)

    problems = Model().validate()
    assert any("not connected" in problem for problem in problems)


def test_a_healthy_graph_validates_clean(branching_network):
    assert branching_network.validate() == []


def test_prune_drops_a_node_that_feeds_nothing():
    net = NeuralNetwork()
    main = net.connect(FullyConnectedLayer(4, 3, "relu"), net.input, name="main")
    net.connect(FullyConnectedLayer(4, 8, "relu"), net.input, name="orphan")
    net.output = main

    dropped = net.prune()

    assert dropped == ["orphan"]
    assert [node.name for node in net.nodes] == [INPUT_NAME, "main"]
    assert net.validate() == []


def test_prune_keeps_an_ancestor_shared_with_a_dead_branch():
    net = NeuralNetwork()
    shared = net.connect(FullyConnectedLayer(4, 6, "relu"), net.input, name="shared")
    live = net.connect(FullyConnectedLayer(6, 2, "linear"), shared, name="live")
    net.connect(FullyConnectedLayer(6, 8, "relu"), shared, name="dead")
    net.output = live

    dropped = net.prune()

    assert dropped == ["dead"]
    assert [node.name for node in net.nodes] == [INPUT_NAME, "shared", "live"]
    assert [consumer.name for consumer in shared.consumers] == ["live"]


def test_prune_leaves_a_healthy_graph_alone(branching_network):
    before = [node.name for node in branching_network.nodes]
    assert branching_network.prune() == []
    assert [node.name for node in branching_network.nodes] == before


def test_a_pruned_network_still_runs(branching_network):
    live_output = branching_network.output
    branching_network.connect(
        FullyConnectedLayer(6, 1, "linear"), branching_network.node("a"), name="dead"
    )
    branching_network.output = live_output
    branching_network.prune()

    x = np.random.default_rng(0).normal(size=(2, 4))
    output = branching_network.forward(x)
    assert output.shape == (2, 2)


# -------------    shapes checked while wiring    ------------------
def test_a_width_mismatch_is_caught_at_the_wiring_line():
    """
    The failure this exists for: 6 out into 9 in is a dot product that cannot
    happen, and it is knowable the moment the edge is written.
    """
    net = NeuralNetwork()
    first = net.connect(FullyConnectedLayer(4, 6, "relu"), net.input)
    with pytest.raises(ValueError, match="cannot be fed by"):
        net.connect(FullyConnectedLayer(9, 3, "relu"), first)


def test_the_error_names_the_source_and_the_axis():
    net = NeuralNetwork()
    first = net.connect(FullyConnectedLayer(4, 6, "relu"), net.input, name="wide")
    with pytest.raises(ValueError, match="'wide'"):
        net.connect(FullyConnectedLayer(9, 3, "relu"), first)


def test_a_declared_input_shape_checks_the_first_edge():
    """without one the first layer is taken on trust, since nothing precedes it"""
    net = NeuralNetwork(input_shape=(7,))
    with pytest.raises(ValueError, match="cannot be fed by"):
        net.connect(FullyConnectedLayer(6, 3, "relu"), net.input)


def test_an_undeclared_input_shape_checks_nothing():
    net = NeuralNetwork()
    assert net.connect(FullyConnectedLayer(6, 3, "relu"), net.input) is not None


def test_a_free_axis_is_not_a_conflict():
    """dropout constrains no width, so anything may feed it"""
    net = NeuralNetwork()
    first = net.connect(FullyConnectedLayer(4, 6, "relu"), net.input)
    assert net.connect(DropoutLayer(0.1), first) is not None


def test_a_node_carries_the_width_it_produces():
    net = NeuralNetwork()
    node = net.connect(FullyConnectedLayer(4, 6, "relu"), net.input)
    assert node.out_shape == (6,)


def test_the_input_node_carries_the_declared_shape():
    assert NeuralNetwork(input_shape=(4,)).input.out_shape == (4,)


def test_a_width_survives_a_merge(branching_network):
    """3 and 5 stacked, so the merge reports 8 and the head is checked on it"""
    assert branching_network.node("merge").out_shape == (8,)


def test_a_width_survives_a_shape_preserving_layer():
    net = NeuralNetwork()
    first = net.connect(FullyConnectedLayer(4, 6, "relu"), net.input)
    dropped = net.connect(DropoutLayer(0.1), first)
    assert dropped.out_shape == (6,)
    with pytest.raises(ValueError, match="cannot be fed by"):
        net.connect(FullyConnectedLayer(5, 2, "relu"), dropped)


def test_a_mismatch_after_a_merge_is_caught():
    """the case the inference exists for: 3 + 5 is 8, so a 7 wide head is wrong"""
    net = NeuralNetwork()
    left = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    right = net.connect(FullyConnectedLayer(4, 5, "linear"), net.input)
    merged = net.connect(LatentStack(), left, right)
    with pytest.raises(ValueError, match="cannot be fed by"):
        net.connect(FullyConnectedLayer(7, 2, "relu"), merged)


def test_the_offending_source_position_is_reported():
    """
    A merge takes its sources in order, so which one is wrong matters -- two
    identical shapes in the message would say nothing about where to look.
    """
    net = NeuralNetwork()
    left = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    right = net.connect(FullyConnectedLayer(4, 9, "linear"), net.input)
    with pytest.raises(ValueError, match="position 1"):
        net.connect(FixedWidthMerge(3, 5), left, right)


def test_a_rejected_edge_leaves_the_graph_alone():
    net = NeuralNetwork()
    first = net.connect(FullyConnectedLayer(4, 6, "relu"), net.input)
    with pytest.raises(ValueError):
        net.connect(FullyConnectedLayer(9, 3, "relu"), first)
    assert len(net) == 1
    assert net.output is first


def test_shapes_report_every_node(branching_network):
    reported = branching_network.shapes()
    assert set(reported) == {"a", "b", "c", "merge", "out"}
    assert reported["b"]["input"] == ((6,),)
    assert reported["merge"]["resolved"] == (8,)


def test_summary_shows_declared_shapes_without_data(branching_network):
    assert "(8,)" in branching_network.summary()


def test_a_trailing_width_match_ignores_the_leading_axes():
    """
    The rule the whole check rests on: a declaration names trailing axes, and
    the axes it does not name are free. A layer that cares only about its width
    must take (batch, sequence, hidden) and (batch, hidden) alike.
    """
    assert shape_conflict((8, 4), (4,)) is None
    assert shape_conflict((3, 8, 4), (4,)) is None
    assert shape_conflict((8, 4), (5,)) is not None


def test_a_shorter_shape_is_silent_rather_than_wrong():
    """
    Deliberately permissive. A source naming one axis says nothing about its
    leading axes, so it might well arrive with the rank a sequence-pinned
    consumer needs. Refusing here would false-alarm on most of the library.
    """
    assert shape_conflict((4,), (8, 4)) is None


def test_a_named_leading_axis_still_conflicts():
    """permissive about absent axes is not the same as permissive about wrong ones"""
    conflict = shape_conflict((8, 4), (16, 4))
    assert conflict is not None and "axis -2" in conflict


def test_a_conflict_names_the_offending_axis():
    conflict = shape_conflict((6,), (9,))
    assert conflict is not None and "axis -1" in conflict


def test_a_none_axis_is_not_a_claim():
    assert shape_conflict((None,), (9,)) is None
    assert shape_conflict((9,), (None,)) is None
    assert shape_conflict(None, (9,)) is None


def test_transposed_merge_sources_are_refused():
    """
    Both widths exist on the layer, just swapped. Checking every source against
    the first declaration would wave this through, which is the reason the
    comparison is positional.
    """
    net = NeuralNetwork()
    left = net.connect(FullyConnectedLayer(4, 5, "linear"), net.input)
    right = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    with pytest.raises(ValueError, match="position 0"):
        net.connect(FixedWidthMerge(3, 5), left, right)


def test_more_sources_than_declared_inputs_is_refused():
    """otherwise the surplus edges are wired but never compared against anything"""
    net = NeuralNetwork()
    first = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    second = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    with pytest.raises(ValueError, match="declares 1 input shapes"):
        net.connect(NormalizeLayer(3), first, second)


def test_a_multi_output_layer_is_refused():
    """
    A node stores whatever forward returned, so there is no way to say which
    output an edge carries. Refused at the wiring line rather than silently
    resolving to output zero.
    """

    class TwoOutputs(Layer):
        def __init__(self):
            super().__init__()
            self.declare_shapes(inputs=((4,),), outputs=((4,), (4,)))

        def forward(self, x):
            return x, x

        def backward(self, incoming_gradient):
            return incoming_gradient

        def update_weights(self, **kwargs) -> None:
            pass

        def purge(self) -> None:
            pass

        def zero_gradients(self) -> None:
            pass

        def get_weights(self, for_serialize: bool = False):
            return {} if for_serialize else None

    net = NeuralNetwork()
    first = net.connect(FullyConnectedLayer(4, 4, "linear"), net.input)
    with pytest.raises(ValueError, match="declares 2 outputs"):
        net.connect(TwoOutputs(), first)


def test_a_sequence_pinned_layer_checks_both_axes():
    """SpectreAttention declares (sequence, hidden), so both are compared"""
    net = NeuralNetwork(input_shape=(16, 8))
    with pytest.raises(ValueError, match="axis -2"):
        net.connect(SpectreAttention(12, 8), net.input)

    net = NeuralNetwork(input_shape=(16, 8))
    with pytest.raises(ValueError, match="axis -1"):
        net.connect(SpectreAttention(16, 4), net.input)

    net = NeuralNetwork(input_shape=(16, 8))
    assert net.connect(SpectreAttention(16, 8), net.input) is not None


# -------------    declarations agree with themselves    -----------
def concrete_layers():
    """
    Every Layer subclass in the package that can be constructed with defaults.

    Walked rather than listed, so a new layer is covered the moment it exists
    instead of when somebody remembers to add it here.
    """
    import importlib
    import pkgutil

    import polyergalio.models as models

    for info in pkgutil.walk_packages(models.__path__, f"{models.__name__}."):
        try:
            importlib.import_module(info.name)
        except Exception:
            continue

    found = []
    stack = [Layer]
    while stack:
        for subclass in stack.pop().__subclasses__():
            stack.append(subclass)
            if inspect.isabstract(subclass):
                continue
            found.append(subclass)
    return found


LAYER_ARGUMENTS = {
    "FullyConnectedLayer": ((4, 6, "relu"), {}),
    "NormalizeLayer": ((6,), {}),
    "RMSNormLayer": ((6,), {}),
    "DropoutLayer": ((), {}),
    "FourierLayer": ((), {}),
    "InverseFourierLayer": ((), {}),
    "FrequencyFFT": ((8, 4), {}),
    "LatentStack": ((), {}),
    "FourierAttention": ((6, 6), {}),
    "SpectreAttention": ((8, 6), {}),
    "RopeEmbedding": ((8, 6), {}),
    "SinusoidEmbedding": ((8, 6), {}),
    # VotingBase itself is excluded: self.stack is empty until a subclass
    # populates it, so its forward output width is input_shape, not the
    # num_experts its declared output shape claims. Its subclasses below
    # (which do populate the stack) exercise the real behavior.
    "VotingWeight": ((6, 4), {}),
    # VotingWeightBalanced and VotingGate carry a hidden width the plain
    # voters do not
    "VotingWeightBalanced": ((6, 5, 4), {"top_k": 2}),
    "VotingGate": ((6, 5, 4), {"top_k": 2}),
    # hidden_dim, num_shared_experts, num_routed_experts, top_k, gate_hidden
    "MixtureOfExperts": ((6, 2, 5, 2, 8), {}),
    "PoolingLayer": ((), {}),
    "LatentSum": ((), {}),
    "LatentProduct": ((), {}),
    "LatentDifference": ((), {}),
    "ShiftRight": ((6,), {}),
    # memory_tokens, hidden_dim
    "PersistentMemory": ((3, 6), {}),
    "SpectreDecoderAttention": ((8, 6), {}),
    # hidden_dim, sequence_length
    "WaveletRefinementModule": ((6, 8), {}),
    # hidden_dim, head_hidden
    "DecisionHead": ((6, 8), {}),
    # num_embeddings, embedding_dim
    "TextEmbedding": ((12, 6), {}),
}


@pytest.mark.parametrize("layer_class", concrete_layers(), ids=lambda c: c.__name__)
def test_every_layer_declares_shapes_consistently(layer_class):
    """
    A declaration is a claim a layer makes about itself, and the graph trusts
    it. Two ways it can be wrong regardless of what the layer computes: fewer
    declared inputs than forward requires, so an edge has nothing to check
    against, and more than one output, which no node can carry.
    """
    recipe = LAYER_ARGUMENTS.get(layer_class.__name__)
    if recipe is None:
        pytest.skip(f"{layer_class.__name__} needs constructor arguments")
    arguments, keywords = recipe
    layer = layer_class(*arguments, **keywords)

    declared_inputs = layer.shapes["input"]
    declared_outputs = layer.shapes["output"]

    assert len(declared_outputs) == 1, (
        f"{layer_class.__name__} declares {len(declared_outputs)} outputs, but "
        "a node carries one value"
    )

    parameters = list(inspect.signature(layer.forward).parameters.values())
    required = sum(
        1
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.name != "self"
        and parameter.default is parameter.empty
    )
    assert len(declared_inputs) >= required, (
        f"{layer_class.__name__}.forward requires {required} inputs but only "
        f"{len(declared_inputs)} shapes are declared, so an edge would go "
        "unchecked"
    )
    for shape in declared_inputs + declared_outputs:
        assert isinstance(shape, tuple), (
            f"{layer_class.__name__} declared {shape!r}, expected a tuple of "
            "trailing axes"
        )


@pytest.mark.parametrize("layer_class", concrete_layers(), ids=lambda c: c.__name__)
def test_every_layer_infers_exactly_one_output_shape(layer_class):
    """
    infer_output_shapes feeds Node.out_shape, so it has to return one shape per
    node however many inputs arrive. A shape-preserving layer returning its
    whole input tuple was the case that broke this.
    """
    recipe = LAYER_ARGUMENTS.get(layer_class.__name__)
    if recipe is None:
        pytest.skip(f"{layer_class.__name__} needs constructor arguments")
    arguments, keywords = recipe
    layer = layer_class(*arguments, **keywords)

    incoming = tuple(ANY_SHAPE for _ in layer.shapes["input"])
    assert len(layer.infer_output_shapes(incoming)) == 1


# -------------    forward and backward shapes agree with reality    -----
# (x_data shape, number of sources). One layer reused twice as both sources
# is how a merge layer (LatentStack, LatentSum, ...) gets exercised, since
# connect() lets the same node feed a layer more than once.
NETWORK_RECIPES = {
    "FullyConnectedLayer": ((3, 8, 4), 1),
    "DropoutLayer": ((3, 8, 6), 1),
    "NormalizeLayer": ((3, 8, 6), 1),
    "RMSNormLayer": ((3, 8, 6), 1),
    "FrequencyFFT": ((5, 4), 1),
    "FourierLayer": ((3, 8, 6), 1),
    "InverseFourierLayer": ((3, 8, 6), 1),
    "FourierAttention": ((3, 8, 6), 1),
    "VotingWeight": ((3, 8, 6), 1),
    "VotingWeightBalanced": ((3, 8, 6), 1),
    "VotingGate": ((3, 8, 6), 1),
    "MixtureOfExperts": ((3, 8, 6), 1),
    "PoolingLayer": ((3, 8, 6), 1),
    "LatentStack": ((3, 8, 6), 2),
    "LatentSum": ((3, 8, 6), 2),
    "LatentProduct": ((3, 8, 6), 2),
    "LatentDifference": ((3, 8, 6), 2),
    "ShiftRight": ((3, 8, 6), 1),
    "SpectreAttention": ((3, 8, 6), 1),
    "SpectreDecoderAttention": ((3, 8, 6), 1),
    "RopeEmbedding": ((3, 8, 6), 1),
    "SinusoidEmbedding": ((3, 8, 6), 1),
    "WaveletRefinementModule": ((3, 8, 6), 2),
    # DecisionHead needs marker_pos per batch, see test_decision_layers
    # TextEmbedding needs integer ids, see test_embedding
    # PersistentMemory takes zero inputs -- connect() requires at least one
    # source, so it cannot be wired into a graph at all. See
    # test_persistent_memory_has_no_backward_shape_to_check below instead.
}


@pytest.mark.parametrize("layer_class", concrete_layers(), ids=lambda c: c.__name__)
def test_every_layer_agrees_with_its_own_forward_and_backward_shapes(layer_class):
    """
    Wire one layer into a real NeuralNetwork, run an actual forward and
    backward pass, and check both directions against reality rather than
    declared metadata:

    - forward's actual output shape against the shape the graph resolved
      for that node at connect() time (a layer can declare its shapes
      correctly and still emit something else)
    - backward's returned input gradient shape against the actual data
      that was fed to forward()
    """
    construction = LAYER_ARGUMENTS.get(layer_class.__name__)
    recipe = NETWORK_RECIPES.get(layer_class.__name__)
    if construction is None or recipe is None:
        pytest.skip(f"{layer_class.__name__} needs a construction and network recipe")

    arguments, keywords = construction
    x_shape, num_sources = recipe
    layer = layer_class(*arguments, **keywords)

    net = NeuralNetwork(input_shape=x_shape[1:])
    node = net.connect(layer, *(net.input for _ in range(num_sources)))
    net.output = node

    x_data = np.random.RandomState(0).normal(size=x_shape)
    output = net.forward(x_data)

    conflict = shape_conflict(output.shape, node.out_shape)
    assert conflict is None, (
        f"{layer_class.__name__}: forward produced {output.shape}, but the "
        f"graph resolved this node's output as {node.out_shape} from "
        f"declared shapes -- {conflict}"
    )

    grad_output = np.random.RandomState(1).normal(size=output.shape)
    grad_input = net.backward(grad_output)

    assert grad_input.shape == x_data.shape, (
        f"{layer_class.__name__}: backward returned {grad_input.shape}, "
        f"forward was fed {x_data.shape}"
    )


def test_persistent_memory_has_no_backward_shape_to_check():
    """
    PersistentMemory declares zero inputs and its forward takes none, so it
    can never be connect()-ed into a graph -- exercised directly instead of
    through NETWORK_RECIPES.
    """
    layer = PersistentMemory(memory_tokens=3, hidden_dim=6)
    output = layer.forward()
    grad = layer.backward(np.random.RandomState(0).normal(size=output.shape))
    assert grad is None


# -------------    the output    -----------------------------------
def test_the_newest_node_is_the_output_by_default():
    net = NeuralNetwork()
    net.connect(FullyConnectedLayer(4, 6, "relu"), net.input)
    last = net.connect(FullyConnectedLayer(6, 2, "linear"), net.nodes[-1])
    assert net.output is last


def test_output_can_be_reassigned():
    net = NeuralNetwork()
    early = net.connect(FullyConnectedLayer(4, 6, "relu"), net.input)
    net.connect(FullyConnectedLayer(6, 2, "linear"), early)
    net.output = early
    assert net.output is early


def test_output_rejects_a_non_node():
    net = NeuralNetwork()
    net.connect(FullyConnectedLayer(4, 4, "relu"), net.input)
    with pytest.raises(TypeError):
        net.output = "somewhere"


def test_output_rejects_a_foreign_node():
    first = NeuralNetwork()
    first.connect(FullyConnectedLayer(4, 4, "relu"), first.input)
    second = NeuralNetwork()
    alien = second.connect(FullyConnectedLayer(4, 4, "relu"), second.input)
    with pytest.raises(ValueError, match="different network"):
        first.output = alien


def test_an_empty_network_has_no_output(small_matrix):
    with pytest.raises(ValueError):
        NeuralNetwork().forward(small_matrix)


def test_set_output_accepts_a_name():
    net = NeuralNetwork()
    early = net.connect(FullyConnectedLayer(4, 6, "relu"), net.input, name="early")
    net.connect(FullyConnectedLayer(6, 2, "linear"), early)
    net.set_output("early")
    assert net.output is early


# -------------    the passes    -----------------------------------
def test_branching_forward_shapes(branching_network):
    rng = np.random.default_rng(0)
    output = branching_network.forward(rng.normal(size=(5, 4)))
    assert output.shape == (5, 2)
    assert branching_network.activations["merge"].shape == (5, 8)


def test_input_gradient_through_a_fan_out(branching_network):
    """the accumulation path: node `a` receives gradient from both branches"""
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(5, 4))
    upstream = rng.normal(size=(5, 2))

    branching_network.forward(x_data)
    analytic = branching_network.backward(upstream.copy())
    numeric = numeric_gradient(
        lambda: float((branching_network.forward(x_data) * upstream).sum()), x_data
    )
    assert relative_error(analytic, numeric) < GRADIENT_TOLERANCE


@pytest.mark.parametrize("node_name", ("a", "b", "c", "out"))
def test_parameter_gradients_through_a_fan_out(node_name, branching_network):
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(5, 4))
    upstream = rng.normal(size=(5, 2))
    layer = branching_network.node(node_name).layer

    branching_network.forward(x_data)
    branching_network.backward(upstream.copy())
    analytic = layer.gradient_weights.copy()

    numeric = numeric_gradient(
        lambda: float((branching_network.forward(x_data) * upstream).sum()),
        layer.weights,
    )
    assert relative_error(analytic, numeric) < GRADIENT_TOLERANCE


def test_a_multi_source_layer_gets_its_gradients_in_order():
    """LatentStack returns a tuple, matched positionally to its sources"""
    net = NeuralNetwork()
    left = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    right = net.connect(FullyConnectedLayer(4, 5, "linear"), net.input)
    net.output = net.connect(LatentStack(), left, right)

    rng = np.random.default_rng(0)
    output = net.forward(rng.normal(size=(6, 4)))
    assert output.shape == (6, 8)
    assert net.backward(np.ones_like(output)).shape == (6, 4)


def test_backward_returns_the_input_gradient(branching_network):
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(5, 4))
    output = branching_network.forward(x_data)
    assert branching_network.backward(np.ones_like(output)).shape == x_data.shape


def test_activations_are_cached_per_node(branching_network):
    rng = np.random.default_rng(0)
    branching_network.forward(rng.normal(size=(5, 4)))
    cached = branching_network.activations
    assert INPUT_NAME in cached
    for node in branching_network.nodes:
        assert node.name in cached


def test_call_is_forward(branching_network):
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(5, 4))
    assert np.allclose(branching_network(x_data), branching_network.forward(x_data))


# -------------    the sequential and name-based paths    ----------
def test_sequential_construction(small_matrix):
    net = NeuralNetwork(
        [FullyConnectedLayer(4, 8, "relu"), FullyConnectedLayer(8, 2, "linear")]
    )
    assert len(net) == 2
    output = net.forward(small_matrix)
    assert output.shape == (6, 2)
    assert net.backward(np.ones_like(output)).shape == small_matrix.shape


def test_extend_chains_onto_the_current_output(small_matrix):
    net = NeuralNetwork([FullyConnectedLayer(4, 8, "relu")])
    net.extend([FullyConnectedLayer(8, 3, "linear")])
    assert net.forward(small_matrix).shape == (6, 3)


def test_name_based_add_still_wires_a_branch():
    """kept for existing graphs: names are resolved to nodes and delegated"""
    net = NeuralNetwork()
    net.add("amp", FullyConnectedLayer(4, 3, "relu"))
    net.add("freq", FullyConnectedLayer(4, 5, "tanh"))
    net.add("stack", LatentStack(), inputs=("amp", "freq"))

    rng = np.random.default_rng(0)
    output = net.forward(rng.normal(size=(5, 4)))
    assert output.shape == (5, 8)
    assert set(net.edges()) == {
        ("input", "amp"),
        ("input", "freq"),
        ("amp", "stack"),
        ("freq", "stack"),
    }


def test_add_returns_the_node_name():
    net = NeuralNetwork()
    assert net.add("first", FullyConnectedLayer(4, 4, "relu")) == "first"


def test_add_reports_an_unknown_name():
    net = NeuralNetwork()
    with pytest.raises(KeyError, match="ghost"):
        net.add("a", FullyConnectedLayer(4, 4, "relu"), inputs="ghost")


def test_node_lookup_by_name(branching_network):
    assert branching_network.node("merge").name == "merge"
    with pytest.raises(KeyError):
        branching_network.node("nowhere")


# -------------    registration    ---------------------------------
def test_assigning_a_layer_registers_it():
    class Model(NeuralNetwork):
        def __init__(self):
            super().__init__()
            self.first = FullyConnectedLayer(4, 8, "relu")
            self.second = FullyConnectedLayer(8, 2, "linear")

    assert len(Model().layers) == 2


def test_a_connected_layer_is_not_double_counted():
    class Model(NeuralNetwork):
        def __init__(self):
            super().__init__()
            self.only = FullyConnectedLayer(4, 8, "relu")
            self.output = self.connect(self.only, self.input)

    assert len(Model().layers) == 1


def test_layers_come_back_in_graph_order(branching_network):
    expected = [node.layer for node in branching_network.nodes if not node.is_source]
    assert branching_network.layers == expected


def test_num_parameters_sums_the_graph():
    net = NeuralNetwork()
    node = net.connect(FullyConnectedLayer(4, 8, "relu"), net.input)
    node = net.connect(NormalizeLayer(8, shift_scale=True), node)
    net.connect(DropoutLayer(dropout_prob=0.1), node)
    assert net.num_parameters == (4 * 8 + 8) + (8 + 8) + 0


# -------------    modes    ----------------------------------------
def test_training_flag_reaches_dropout():
    net = NeuralNetwork([DropoutLayer(dropout_prob=0.5, use_rescale=False)])
    ones = np.ones((300, 120))

    net.train()
    training_mean = net.forward(ones).mean()
    net.eval()
    evaluation_mean = net.forward(ones).mean()

    assert training_mean == pytest.approx(0.5, abs=0.03)
    assert evaluation_mean == pytest.approx(1.0)


def test_train_and_eval_return_self():
    net = NeuralNetwork()
    assert net.train() is net
    assert net.eval() is net


def test_networks_start_in_training_mode():
    assert NeuralNetwork().training is True


def test_layers_without_a_training_argument_are_unaffected(small_matrix):
    net = NeuralNetwork([FullyConnectedLayer(4, 3, "relu")])
    net.eval()
    assert net.forward(small_matrix).shape == (6, 3)


# -------------    bookkeeping    ----------------------------------
def test_purge_clears_layers_and_activations(branching_network):
    rng = np.random.default_rng(0)
    output = branching_network.forward(rng.normal(size=(5, 4)))
    branching_network.backward(np.ones_like(output))

    branching_network.purge()
    assert branching_network.activations == {}
    assert branching_network.node("a").layer.input is None


def test_zero_gradients_reaches_every_layer(branching_network):
    rng = np.random.default_rng(0)
    output = branching_network.forward(rng.normal(size=(5, 4)))
    branching_network.backward(np.ones_like(output))
    assert branching_network.node("a").layer.gradient_weights.any()

    branching_network.zero_gradients()
    assert not branching_network.node("a").layer.gradient_weights.any()


def test_summary_lists_every_node_and_marks_the_output(branching_network):
    rng = np.random.default_rng(0)
    text = branching_network.summary(rng.normal(size=(5, 4)))
    for node in branching_network.nodes:
        if not node.is_source:
            assert node.name in text
    assert "branching" in text
    assert "<- output" in text


def test_summary_works_without_sample_data(branching_network):
    assert "merge" in branching_network.summary()


def test_summary_surfaces_validation_warnings():
    net = NeuralNetwork()
    main = net.connect(FullyConnectedLayer(4, 3, "relu"), net.input, name="main")
    net.connect(FullyConnectedLayer(4, 8, "relu"), net.input, name="orphan")
    net.output = main
    assert "warning" in net.summary()


def test_repr_reports_size(branching_network):
    """a, b, c, merge, out -- the input source is not counted as a layer"""
    assert "5 nodes" in repr(branching_network)
    assert len(branching_network) == 5


def test_node_repr_names_its_sources(branching_network):
    assert "b <- a" in repr(branching_network.node("b"))
    assert "graph input" in repr(branching_network.input)


# -------------    training end to end    --------------------------
def test_branching_network_learns(regression_dataset):
    """
    A wrong fan-out would still descend, just more slowly, so require real
    progress rather than merely a decrease.
    """
    x_data, y_data, _ = regression_dataset
    y_data = y_data.reshape(-1, 1)

    net = NeuralNetwork(name="regression")
    features = net.input
    wide = net.connect(FullyConnectedLayer(3, 12, "relu"), features)
    narrow = net.connect(FullyConnectedLayer(3, 4, "tanh"), features)
    merged = net.connect(LatentStack(), wide, narrow)
    merged = net.connect(FullyConnectedLayer(16, 8, "swish"), merged)
    net.output = net.connect(
        FullyConnectedLayer(8, 1, "linear", is_output=True), merged
    )

    loss = MSELoss()
    optimizer = SGD(0.01)

    net.train()
    first = None
    for _ in range(250):
        prediction = net.forward(x_data)
        value = loss(prediction, y_data)
        if first is None:
            first = value
        net.backward(loss.backward())
        optimizer.step(net.layers)

    assert value < first * 0.9, f"expected real progress, got {first} -> {value}"


def test_optimizer_accepts_net_layers_directly(small_matrix):
    """the point of registration: no hand-maintained layer list"""
    net = NeuralNetwork(
        [FullyConnectedLayer(4, 6, "relu"), FullyConnectedLayer(6, 2, "linear")]
    )
    output = net.forward(small_matrix)
    net.backward(np.ones_like(output))

    first_layer = net.layers[0]
    before = first_layer.weights.copy()
    SGD(0.1).step(net.layers)
    assert not np.allclose(before, first_layer.weights)
