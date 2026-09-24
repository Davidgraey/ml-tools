"""
Serialize round trips: construct, train, serialize, deserialize, and check the
rebuilt object is the trained one -- for every Layer type, and for whole
NeuralNetwork graphs and the nodes that wire them.

Every prediction is an inference pass: model.eval() then forward, on both the
original and the rebuilt model, so the comparison never depends on dropout,
stochastic depth or load-balancing updates that only run while training.
"""

import inspect
import pickle

import numpy as np
import pytest
from ml_tools.models.constants import DECISION_TYPES
from ml_tools.models.layers.basal_layers import FullyConnectedLayer, Layer, NormalizeLayer
from ml_tools.models.layers.decision_layers import DecisionHead
from ml_tools.models.layers.mixture_layers import MixtureOfExperts
from ml_tools.models.layers.operator_layers import LatentStack
from ml_tools.models.model_loss import MSELoss
from ml_tools.models.neural_network import INPUT_NAME, NeuralNetwork
from ml_tools.models.optimizers import SGD
from test_network import concrete_layers

TRAIN_STEPS = 5
LEARNING_RATE = 0.01
TOLERANCE = 1e-10


def decision_inputs(rng):
    marker_pos = np.array([[2, 3, 4, 5], [2, 4, 0, 0], [3, 5, 7, 0]])
    kwargs = dict(
        marker_pos=marker_pos,
        token_mask=marker_pos > 0,
        decisiontypes=[DECISION_TYPES.CHOICE, DECISION_TYPES.BINARY, DECISION_TYPES.SCORE],
    )
    return (rng.normal(size=(3, 8, 6)),), kwargs


def token_inputs(rng):
    return (rng.integers(0, 12, size=(3, 8)),), {}


def sequence_inputs(count: int, shape: tuple = (3, 8, 6)):
    return lambda rng: (tuple(rng.normal(size=shape) for _ in range(count)), {})


# layer name -> (constructor args, constructor kwargs, input builder)
# the input builder takes an rng and returns (positional inputs, forward kwargs)
RECIPES = {
    "FullyConnectedLayer": ((4, 6, "relu"), {}, sequence_inputs(1, (3, 8, 4))),
    "DropoutLayer": ((), {"dropout_prob": 0.3, "use_rescale": True}, sequence_inputs(1)),
    "NormalizeLayer": ((6,), {}, sequence_inputs(1)),
    "RMSNormLayer": ((6,), {}, sequence_inputs(1)),
    "LatentStack": ((), {}, sequence_inputs(2)),
    "LatentSum": ((), {}, sequence_inputs(2)),
    "LatentProduct": ((), {}, sequence_inputs(2)),
    "LatentDifference": ((), {}, sequence_inputs(2)),
    "ShiftRight": ((6,), {}, sequence_inputs(1)),
    "WaveletRefinementModule": ((6, 8), {}, sequence_inputs(2)),
    "PersistentMemory": ((3, 6), {}, lambda rng: ((), {})),
    "SpectreAttention": ((8, 6), {}, sequence_inputs(1)),
    "SpectreDecoderAttention": ((8, 6), {}, sequence_inputs(1)),
    "DecisionHead": ((6, 8), {}, decision_inputs),
    "TextEmbedding": ((12, 6), {"padding_idx": 0}, token_inputs),
    "RopeEmbedding": ((8, 6), {}, sequence_inputs(1)),
    "SinusoidEmbedding": ((8, 6), {}, sequence_inputs(1)),
    "FrequencyFFT": ((8, 4), {}, sequence_inputs(1, (5, 4))),
    "FourierLayer": ((), {}, sequence_inputs(1)),
    "InverseFourierLayer": ((), {}, sequence_inputs(1)),
    "FourierAttention": ((6, 6), {}, sequence_inputs(1)),
    "MixtureOfExperts": ((6, 2, 5, 2, 8), {}, sequence_inputs(1)),
    "PoolingLayer": ((), {}, sequence_inputs(1)),
    "VotingWeight": ((6, 4), {}, sequence_inputs(1)),
    "VotingWeightBalanced": ((6, 5, 4), {"top_k": 2}, sequence_inputs(1)),
    "VotingGate": ((6, 5, 4), {"top_k": 2}, sequence_inputs(1)),
}

# VotingBase has an empty expert stack until a subclass fills it
NOT_ROUND_TRIPPED = {"VotingBase"}


def package_layers() -> list[type]:
    return [
        layer_class
        for layer_class in concrete_layers()
        if layer_class.__module__.startswith("ml_tools.")
        and layer_class.__name__ not in NOT_ROUND_TRIPPED
    ]


def predict(model, inputs: tuple, kwargs: dict) -> np.ndarray:
    """Inference output: switch the layer or network to eval(), then run forward."""
    return model.eval().forward(*inputs, **kwargs)


def reseed(layer: Layer, seed: int = 7) -> None:
    """Give a layer and all its sublayers the same fresh random stream."""
    layer.RNG = np.random.RandomState(seed)
    for sublayer in layer.sublayers():
        reseed(sublayer, seed)


def train(layer: Layer, inputs: tuple, kwargs: dict, target: np.ndarray) -> None:
    """A few SGD steps in training mode, pulling the layer's output towards target."""
    optimizer = SGD(LEARNING_RATE)
    layer.train()
    for _ in range(TRAIN_STEPS):
        layer.zero_gradients()
        output = layer.forward(*inputs, **kwargs)
        layer.backward(output - target)
        optimizer.step([layer])


def flatten(value, prefix: str = "") -> dict[str, np.ndarray]:
    """Nested weights as {path: array}, so two layers' weights can be compared leaf by leaf."""
    if isinstance(value, dict):
        leaves = {}
        for key, sub in value.items():
            leaves.update(flatten(sub, f"{prefix}/{key}"))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = {}
        for index, sub in enumerate(value):
            leaves.update(flatten(sub, f"{prefix}/{index}"))
        return leaves
    return {prefix: None if value is None else np.array(value, copy=True)}


def assert_same_leaves(expected: dict, actual: dict) -> None:
    assert expected.keys() == actual.keys()
    for path, value in expected.items():
        if value is None:
            assert actual[path] is None, path
        else:
            assert np.allclose(actual[path], value, atol=TOLERANCE), path


@pytest.fixture
def rng():
    return np.random.default_rng(2026)


def build(layer_class: type, rng) -> tuple:
    arguments, keywords, make_inputs = RECIPES[layer_class.__name__]
    layer = layer_class(*arguments, **keywords)
    inputs, kwargs = make_inputs(rng)
    target = rng.normal(size=np.shape(predict(layer, inputs, kwargs)))
    return layer, inputs, kwargs, target


def round_trip(layer: Layer) -> Layer:
    """serialize -> pickle -> deserialize, the same path NeuralNetwork.save/load take."""
    return Layer.deserialize(pickle.loads(pickle.dumps(layer.serialize())))


# -------------    coverage    -------------------------------------
def test_every_package_layer_has_a_recipe():
    missing = sorted(c.__name__ for c in package_layers() if c.__name__ not in RECIPES)
    assert not missing, f"add a serialization recipe for: {missing}"


@pytest.mark.parametrize("layer_class", package_layers(), ids=lambda c: c.__name__)
def test_config_keeps_every_constructor_argument(layer_class):
    """get_config must hand back each argument as passed, or deserialize rebuilds a different layer"""
    arguments, keywords, _ = RECIPES[layer_class.__name__]
    bound = inspect.signature(layer_class.__init__).bind(None, *arguments, **keywords)
    bound.apply_defaults()
    passed = {name: value for name, value in bound.arguments.items() if name != "self"}
    config = layer_class(*arguments, **keywords).get_config()
    assert config == passed


# -------------    inference mode    -------------------------------
def all_sublayers(layer: Layer) -> list[Layer]:
    found = []
    for sublayer in layer.sublayers():
        found.extend([sublayer, *all_sublayers(sublayer)])
    return found


@pytest.mark.parametrize("layer_class", package_layers(), ids=lambda c: c.__name__)
def test_eval_and_train_switch_the_layer_and_its_sublayers(layer_class, rng):
    layer, _, _, _ = build(layer_class, rng)
    assert layer.eval() is layer
    assert not layer.training and not any(sub.training for sub in all_sublayers(layer))
    assert layer.train() is layer
    assert layer.training and all(sub.training for sub in all_sublayers(layer))


@pytest.mark.parametrize("layer_class", package_layers(), ids=lambda c: c.__name__)
def test_inference_is_repeatable_and_leaves_the_layer_unchanged(layer_class, rng):
    layer, inputs, kwargs, target = build(layer_class, rng)
    train(layer, inputs, kwargs, target)
    weights = flatten(layer.get_weights(for_serialize=True))

    first = predict(layer, inputs, kwargs)
    assert np.allclose(predict(layer, inputs, kwargs), first, atol=TOLERANCE)
    assert_same_leaves(weights, flatten(layer.get_weights(for_serialize=True)))


@pytest.mark.parametrize(
    "layer_class",
    [c for c in package_layers() if "training_now" in inspect.signature(c.forward).parameters],
    ids=lambda c: c.__name__,
)
def test_forward_follows_the_mode_when_training_now_is_not_given(layer_class, rng):
    layer, inputs, kwargs, _ = build(layer_class, rng)
    for mode in (False, True):
        reseed(layer)
        explicit = layer.forward(*inputs, **kwargs, training_now=mode)
        reseed(layer)
        implied = layer.train(mode).forward(*inputs, **kwargs)
        assert np.allclose(implied, explicit, atol=TOLERANCE)


# -------------    round trips    ----------------------------------
@pytest.mark.parametrize("layer_class", package_layers(), ids=lambda c: c.__name__)
def test_training_moves_the_weights(layer_class, rng):
    """guards the round trip below from passing on untouched initial weights"""
    layer, inputs, kwargs, target = build(layer_class, rng)
    if not layer.num_parameters:
        pytest.skip(f"{layer_class.__name__} has no parameters")
    before = flatten(layer.get_weights(for_serialize=True))
    train(layer, inputs, kwargs, target)
    after = flatten(layer.get_weights(for_serialize=True))
    assert any(
        value is not None and not np.allclose(after[path], value) for path, value in before.items()
    ), f"{layer_class.__name__}: training did not change any weight"


@pytest.mark.parametrize("layer_class", package_layers(), ids=lambda c: c.__name__)
def test_round_trip_restores_type_config_and_weights(layer_class, rng):
    layer, inputs, kwargs, target = build(layer_class, rng)
    train(layer, inputs, kwargs, target)
    rebuilt = round_trip(layer)

    assert type(rebuilt) is layer_class
    assert rebuilt.get_config() == layer.get_config()
    assert rebuilt.num_parameters == layer.num_parameters
    assert_same_leaves(
        flatten(layer.get_weights(for_serialize=True)),
        flatten(rebuilt.get_weights(for_serialize=True)),
    )


@pytest.mark.parametrize("layer_class", package_layers(), ids=lambda c: c.__name__)
def test_round_trip_reproduces_inference(layer_class, rng):
    layer, inputs, kwargs, target = build(layer_class, rng)
    train(layer, inputs, kwargs, target)
    rebuilt = round_trip(layer)

    expected = predict(layer, inputs, kwargs)
    assert np.allclose(predict(rebuilt, inputs, kwargs), expected, atol=TOLERANCE)


@pytest.mark.parametrize("layer_class", package_layers(), ids=lambda c: c.__name__)
def test_round_trip_keeps_training_identically(layer_class, rng):
    """
    a rebuilt layer resumes training on the same trajectory as the original;
    both get the same random stream, since RNG state is not serialized
    """
    layer, inputs, kwargs, target = build(layer_class, rng)
    train(layer, inputs, kwargs, target)
    rebuilt = round_trip(layer)

    reseed(layer)
    reseed(rebuilt)
    train(layer, inputs, kwargs, target)
    train(rebuilt, inputs, kwargs, target)
    assert np.allclose(predict(rebuilt, inputs, kwargs), predict(layer, inputs, kwargs), atol=TOLERANCE)


@pytest.mark.parametrize("layer_class", package_layers(), ids=lambda c: c.__name__)
def test_rebuilt_layer_shares_no_memory_with_the_original(layer_class, rng):
    layer, inputs, kwargs, target = build(layer_class, rng)
    train(layer, inputs, kwargs, target)
    rebuilt = Layer.deserialize(layer.serialize())
    saved = flatten(rebuilt.get_weights(for_serialize=True))

    train(layer, inputs, kwargs, target)
    assert_same_leaves(saved, flatten(rebuilt.get_weights(for_serialize=True)))


# -------------    networks and nodes    ---------------------------
def branching_network() -> NeuralNetwork:
    """input -> a, a fans out to b and c, which merge into the output"""
    net = NeuralNetwork(name="branching", input_shape=(4,))
    a = net.connect(FullyConnectedLayer(4, 6, "tanh"), net.input, name="a")
    b = net.connect(FullyConnectedLayer(6, 3, "swish"), a, name="b")
    c = net.connect(FullyConnectedLayer(6, 5, "relu"), a, name="c")
    merged = net.connect(LatentStack(), b, c, name="merge")
    normed = net.connect(NormalizeLayer(8), merged, name="norm")
    net.output = net.connect(FullyConnectedLayer(8, 2, "linear"), normed, name="out")
    return net


def mixture_network() -> NeuralNetwork:
    net = NeuralNetwork(name="mixture", input_shape=(None, 6))
    experts = net.connect(MixtureOfExperts(6, 2, 5, 2, 8), net.input, name="experts")
    net.output = net.connect(FullyConnectedLayer(6, 3, "linear"), experts, name="out")
    return net


def decision_network() -> NeuralNetwork:
    """a head fed by forward kwargs, which the rebuilt graph has to route again"""
    net = NeuralNetwork(name="decision", input_shape=(None, 6))
    hidden = net.connect(FullyConnectedLayer(6, 6, "tanh"), net.input, name="encoder")
    net.output = net.connect(DecisionHead(6, 8), hidden, name="head")
    return net


def network_inputs(builder, rng) -> tuple:
    if builder is branching_network:
        return rng.normal(size=(12, 4)), {}
    (x_data,), kwargs = decision_inputs(rng)
    return x_data, kwargs if builder is decision_network else {}


NETWORK_BUILDERS = (branching_network, mixture_network, decision_network)


def train_network(net: NeuralNetwork, x_data, kwargs: dict, target, steps: int = TRAIN_STEPS) -> None:
    loss = MSELoss()
    optimizer = SGD(LEARNING_RATE)
    net.train()
    for _ in range(steps):
        net.zero_gradients()
        loss(net.forward(x_data, **kwargs), target)
        net.backward(loss.backward())
        optimizer.step(net.layers)


def build_network(builder, rng) -> tuple:
    net = builder()
    x_data, kwargs = network_inputs(builder, rng)
    target = rng.normal(size=predict(net, (x_data,), kwargs).shape)
    return net, x_data, kwargs, target


def network_round_trip(net: NeuralNetwork) -> NeuralNetwork:
    return NeuralNetwork.deserialize(pickle.loads(pickle.dumps(net.serialize())))


def network_weights(net: NeuralNetwork) -> dict:
    return {node.name: flatten(node.layer.get_weights(for_serialize=True)) for node in net.nodes if not node.is_source}


@pytest.mark.parametrize("builder", NETWORK_BUILDERS, ids=lambda b: b.__name__)
def test_network_round_trip_restores_the_graph(builder, rng):
    net, x_data, kwargs, target = build_network(builder, rng)
    train_network(net, x_data, kwargs, target)
    rebuilt = network_round_trip(net)

    assert rebuilt.name == net.name
    assert rebuilt.input.shapes == net.input.shapes
    assert [node.name for node in rebuilt.nodes] == [node.name for node in net.nodes]
    assert rebuilt.edges() == net.edges()
    assert rebuilt.output.name == net.output.name
    assert rebuilt.num_parameters == net.num_parameters
    assert rebuilt.validate() == net.validate()


@pytest.mark.parametrize("builder", NETWORK_BUILDERS, ids=lambda b: b.__name__)
def test_network_round_trip_restores_every_node(builder, rng):
    net, x_data, kwargs, target = build_network(builder, rng)
    train_network(net, x_data, kwargs, target)
    rebuilt = network_round_trip(net)

    for original, restored in zip(net.nodes, rebuilt.nodes):
        assert restored.name == original.name
        assert restored.is_source == original.is_source
        assert [source.name for source in restored.sources] == [source.name for source in original.sources]
        assert [consumer.name for consumer in restored.consumers] == [consumer.name for consumer in original.consumers]
        assert restored.shapes == original.shapes
        if not original.is_source:
            assert type(restored.layer) is type(original.layer)
            assert restored.layer.get_config() == original.layer.get_config()
            assert restored.layer is not original.layer
    for name, leaves in network_weights(net).items():
        assert_same_leaves(leaves, network_weights(rebuilt)[name])


@pytest.mark.parametrize("builder", NETWORK_BUILDERS, ids=lambda b: b.__name__)
def test_network_eval_and_train_reach_every_layer(builder):
    net = builder()
    assert net.eval() is net
    assert not net.training and not any(layer.training for layer in net.layers)
    assert net.train() is net
    assert net.training and all(layer.training for layer in net.layers)


def test_a_fan_out_stays_one_shared_node(rng):
    rebuilt = network_round_trip(branching_network())
    shared = rebuilt.node("a")
    consumers = [node for node in rebuilt.nodes if any(source is shared for source in node.sources)]
    assert [node.name for node in consumers] == ["b", "c"]


@pytest.mark.parametrize("builder", NETWORK_BUILDERS, ids=lambda b: b.__name__)
def test_network_round_trip_reproduces_both_passes(builder, rng):
    net, x_data, kwargs, target = build_network(builder, rng)
    train_network(net, x_data, kwargs, target)
    rebuilt = network_round_trip(net)

    expected = predict(net, (x_data,), kwargs)
    assert np.allclose(predict(rebuilt, (x_data,), kwargs), expected, atol=TOLERANCE)
    upstream = rng.normal(size=expected.shape)
    assert np.allclose(rebuilt.backward(upstream), net.backward(upstream), atol=TOLERANCE)


@pytest.mark.parametrize("builder", NETWORK_BUILDERS, ids=lambda b: b.__name__)
def test_network_round_trip_keeps_training_identically(builder, rng):
    net, x_data, kwargs, target = build_network(builder, rng)
    train_network(net, x_data, kwargs, target)
    rebuilt = network_round_trip(net)

    for layer in net.layers + rebuilt.layers:
        reseed(layer)
    train_network(net, x_data, kwargs, target)
    train_network(rebuilt, x_data, kwargs, target)
    assert np.allclose(predict(rebuilt, (x_data,), kwargs), predict(net, (x_data,), kwargs), atol=TOLERANCE)


@pytest.mark.parametrize("builder", NETWORK_BUILDERS, ids=lambda b: b.__name__)
def test_rebuilt_network_shares_no_memory_with_the_original(builder, rng):
    net, x_data, kwargs, target = build_network(builder, rng)
    train_network(net, x_data, kwargs, target)
    rebuilt = NeuralNetwork.deserialize(net.serialize())
    saved = network_weights(rebuilt)

    train_network(net, x_data, kwargs, target)
    for name, leaves in saved.items():
        assert_same_leaves(leaves, network_weights(rebuilt)[name])


def test_serialized_network_is_a_snapshot(rng):
    net, x_data, kwargs, target = build_network(branching_network, rng)
    snapshot = net.serialize()
    before = {entry["name"]: flatten(entry["layer"]["weights"]) for entry in snapshot["nodes"]}
    train_network(net, x_data, kwargs, target)
    for entry in snapshot["nodes"]:
        assert_same_leaves(before[entry["name"]], flatten(entry["layer"]["weights"]))


@pytest.mark.parametrize("builder", NETWORK_BUILDERS, ids=lambda b: b.__name__)
def test_save_and_load_through_a_file(builder, rng, tmp_path):
    net, x_data, kwargs, target = build_network(builder, rng)
    train_network(net, x_data, kwargs, target)
    path = tmp_path / f"{builder.__name__}.pkl"
    net.save(str(path))
    loaded = NeuralNetwork.load(str(path))
    assert np.allclose(predict(loaded, (x_data,), kwargs), predict(net, (x_data,), kwargs), atol=TOLERANCE)


def test_serialized_nodes_are_listed_by_name_without_the_input():
    serialized = branching_network().serialize()
    assert [entry["name"] for entry in serialized["nodes"]] == ["a", "b", "c", "merge", "norm", "out"]
    assert INPUT_NAME not in [entry["name"] for entry in serialized["nodes"]]
    assert serialized["nodes"][3]["sources"] == ["b", "c"]
    assert serialized["output"] == "out"


def test_out_of_order_nodes_are_refused():
    serialized = branching_network().serialize()
    serialized["nodes"] = serialized["nodes"][::-1]
    with pytest.raises(KeyError, match="out of order"):
        NeuralNetwork.deserialize(serialized)


def test_an_unknown_layer_type_is_refused():
    serialized = branching_network().serialize()
    serialized["nodes"][0]["layer"]["type"] = "NoSuchLayer"
    with pytest.raises(KeyError, match="NoSuchLayer"):
        NeuralNetwork.deserialize(serialized)


def test_a_reassigned_output_survives_the_round_trip():
    net = branching_network()
    net.set_output("merge")
    assert network_round_trip(net).output.name == "merge"

