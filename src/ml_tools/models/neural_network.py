"""
A container for wiring layers into a network.

Layers know how to transform an array and how to push a gradient back through
themselves. What they do not know is what feeds them. This module owns that.

Connections are object references, not names. `connect` returns the node it
created, and you pass that node in as the source of the next one, so an edge is
a pointer from consumer to producer rather than a string resolved later. Three
things follow from that:

  * a mistyped source is a NameError where you wrote it, not a wrong edge
  * a cycle cannot be built, since a node can only reference nodes that already
    exist, so there is no forward reference to close a loop with
  * insertion order is therefore already a topological order, and the forward
    pass is a walk down the list

Names still exist, but only as labels for reading a summary or fetching a node
after the fact. Nothing about the graph's structure depends on them.

Shapes are checked as the graph is built. Every layer declares the trailing
axes it accepts and emits (`layer.shapes`), so `connect` can compare what a
source produces against what its consumer wants and refuse the edge on the spot
-- a width mismatch is a ValueError at the wiring line, not a dot-product error
several layers into the first forward pass.
"""

import inspect
from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray

from ml_tools.models.layers.layers import ANY_SHAPE, Layer, shape_conflict


# the label carried by the graph's source node. Structure does not depend on it
# -- it exists so summaries and add() have something to say.
INPUT_NAME = "input"


class Node:
    """
    One step in the graph: a layer, and references to the nodes feeding it.

    A node with no layer is a source -- the graph input. Nodes are compared and
    hashed by identity, so the same node passed to two consumers is one shared
    producer, which is what makes a fan-out visible in the code that builds it.

    Each node also carries the trailing shape it produces, resolved from its
    sources at construction. That is what the next node's check is made
    against, so a width flows down the graph as it is wired.
    """

    def __init__(
        self,
        name: str,
        layer: Optional[Layer] = None,
        sources: tuple = (),
        shape: tuple = ANY_SHAPE,
    ):
        self.name = name
        self.layer = layer
        self.sources = sources
        self.consumers: list = []

        if layer is None:
            self.out_shape = shape
        else:
            incoming = tuple(source.out_shape for source in sources)
            self.out_shape = layer.infer_output_shapes(incoming)[0]

        # Check once here instead of inspecting the signature on every forward pass.
        self.accepts_training = bool(layer) and (
            "training_now" in inspect.signature(layer.forward).parameters
        )

        for source in sources:
            source.consumers.append(self)

    @property
    def is_source(self) -> bool:
        return self.layer is None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        if self.is_source:
            return f"Node({self.name}, graph input)"
        feeding = ", ".join(source.name for source in self.sources)
        return f"Node({self.name} <- {feeding})"

    def __str__(self):
        return f"{self.__repr__()} producing {self.out_shape}"


class NeuralNetwork:
    """
    A directed acyclic graph of layers.

    connect the DAG by passing nodes:

        net = NeuralNetwork()
        audio = net.input
        # fanning out to multiple outputs
        amp = net.connect(amplitude_fc, audio)
        freq = net.connect(frequency_fft, audio)
        merged = net.connect(LatentStack(), amp, freq)
        net.output = net.connect(head, merged)

    or sequentially, when there is nothing to branch::

        net = NeuralNetwork([layer_a, layer_b, layer_c])
    """

    def __init__(
        self,
        layers: Optional[Iterable[Layer]] = None,
        name: Optional[str] = None,
        input_shape: tuple = ANY_SHAPE,
    ):
        """
        Parameters
        ----------
        layers : optional layers to chain end to end, for a graph with nothing
            to branch
        name : label for summaries
        input_shape : trailing axes of the data the graph will be fed, with
            None for any axis that varies. Given, the first edge is checked
            like every other one; left out, the first layer is taken on trust
            until data arrives.
        """
        # through object.__setattr__, so the attribute interception below has
        # its registry available before any assignment happens
        object.__setattr__(self, "_nodes", [])
        object.__setattr__(self, "_registered", [])
        object.__setattr__(self, "_output", None)
        object.__setattr__(self, "training", True)
        object.__setattr__(self, "activations", {})
        object.__setattr__(self, "name", name or self.__class__.__name__)

        source = Node(INPUT_NAME, shape=tuple(input_shape))
        object.__setattr__(self, "_input", source)
        self._nodes.append(source)

        if layers is not None:
            self.extend(layers)

    # ------------- connecting
    @property
    def input(self) -> Node:
        """ the graph's source node; pass it as an input source to the first layer """
        return self._input

    def connect(self,
                layer: Layer,
                *sources: Node,
                name: Optional[str] = None
                ) -> Node:
        """
        Place a layer in the graph, fed by the given nodes, and return its node.

        Parameters
        ----------
        layer : the layer to run at this node
        sources : the nodes whose outputs feed it, in the order the layer's
            forward takes them. Passing one node to two different calls is how
            a fan-out is expressed.
        name : optional label. Defaults to the layer's class name with a
            counter, and is only used for display and lookup.

        Returns
        -------
        the new node, to pass as a source to whatever comes next
        """
        if not sources:
            raise ValueError(
                f"{layer.__class__.__name__} needs at least one source. Pass "
                "net.input for the first layer in a graph."
            )

        for position, source in enumerate(sources):
            if not isinstance(source, Node):
                raise TypeError(
                    f"source {position} is {type(source).__name__}, expected a "
                    "Node. Use the value returned by connect(), or net.input."
                )
            if not any(known is source for known in self._nodes):
                raise ValueError(
                    f"source {source.name!r} belongs to a different network"
                )

        self._check_output_shapes(layer, len(sources))
        self._check_shapes(layer, sources)

        label = name or self._auto_name(layer)
        if any(node.name == label for node in self._nodes):
            raise ValueError(f"node name {label!r} is already taken")

        node = Node(label, layer, tuple(sources))
        self._nodes.append(node)
        self._remember(layer)
        # a freshly connected node is the natural output until told otherwise
        object.__setattr__(self, "_output", node)
        return node

    def _check_output_shapes(self, layer: Layer, given: int) -> None:
        """
        Compare the source count against the layer's forward signature, so a
        merge given the wrong number of inputs fails here rather than as a
        positional-argument TypeError mid-forward.
        """
        parameters = list(inspect.signature(layer.forward).parameters.values())
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            and parameter.name != "self"
        ]
        if any(parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters):
            return

        required = sum(
            1 for parameter in positional if parameter.default is parameter.empty
        )
        if not (required <= given <= len(positional)):
            raise ValueError(
                f"{layer.__class__.__name__}.forward takes {required} to "
                f"{len(positional)} inputs, got {given}"
            )

    def _check_shapes(self, layer: Layer, sources: tuple[Node, ...]) -> None:
        """
        Compare what each source produces against what the layer says it takes.
        """
        expected = layer.shapes["input"]

        if isinstance(expected, tuple):
            if isinstance(expected[0], tuple):
                expected = expected[0]
            expected = expected[0]

        for position, source in enumerate(sources):
            # forward may accept more inputs than the layer
            if source.layer is None:
                upstream = source.out_shape
            else:
                upstream = source.layer.shapes["output"]

            if isinstance(upstream, tuple):
                if isinstance(upstream[-1], tuple):
                    upstream = upstream[-1]
                upstream = upstream[-1]
            print(upstream, expected)
            conflict = shape_conflict(upstream, expected)

            if conflict:
                raise ValueError(
                    f"{layer.__class__.__name__} cannot be fed by "
                    f"{source.name!r} at position {position}: {conflict}"
                )

    def _auto_name(self, layer: Layer) -> str:
        stem = layer.__class__.__name__
        taken = {node.name for node in self._nodes}
        index = 0
        while f"{stem}_{index}" in taken:
            index += 1
        return f"{stem}_{index}"

    def extend(self, layers: Iterable[Layer]) -> Node:
        """ chain layers end to end """
        node = self._output or self._input
        for layer in layers:
            node = self.connect(layer, node)
        return node

    def add(
        self,
        name: str,
        layer: Layer,
        inputs: str | Node | Iterable = "input",
    ) -> str:
        """
        name-based wiring, kept so existing graphs keep working.

        Resolves each name to a node and delegates to connect. Prefer connect:
        a name is matched at wiring time, so a typo that happens to hit another
        real node produces a valid graph with the wrong edge, which nothing can
        detect.
        """
        requested = (
            (inputs,) if isinstance(inputs, (str, Node)) else tuple(inputs)
        )
        sources = tuple(
            source if isinstance(source, Node) else self.get_node(source)
            for source in requested
        )
        return self.connect(layer, *sources, name=name).name

    def get_node(self, name: str) -> Node:
        """fetch a node by label"""
        for node in self._nodes:
            if node.name == name:
                return node
        known = [node.name for node in self._nodes]
        raise KeyError(f"no node named {name!r}. Known nodes: {known}")

    # ------------- the output
    @property
    def output(self) -> Node:
        if self._output is None:
            raise ValueError("the network has no layers")
        return self._output

    @output.setter
    def output(self, node: Node) -> None:
        if not isinstance(node, Node):
            raise TypeError("the output must be a Node() returned by connect()")
        if not any(known is node for known in self._nodes):
            raise ValueError(f"node {node.name!r} belongs to a different network")
        object.__setattr__(self, "_output", node)

    def set_output(self, node: Node | str) -> None:
        """as the output property, accepting a name for convenience"""
        self.output = self.get_node(node) if isinstance(node, str) else node

    # ------------- registration
    def __setattr__(self, attribute: str, value):
        """ assigning a Layer registers it directly """
        if isinstance(value, Layer):
            self._remember(value)
        elif isinstance(value, NeuralNetwork) and value is not self:
            for layer in value.layers:
                self._remember(layer)
        object.__setattr__(self, attribute, value)

    def _remember(self, layer: Layer) -> None:
        if not any(known is layer for known in self._registered):
            self._registered.append(layer)

    # ------------- the passes
    def forward(self, x_data: NDArray) -> NDArray:
        """
        forward pass -- taking the insertion order or navigating the node-to-node process
        """
        output = self.output
        values = {self._input: x_data}

        for node in self._nodes:
            if node.is_source:
                continue
            arguments = [values[source] for source in node.sources]
            if node.accepts_training:
                values[node] = node.layer.forward(
                    *arguments, training_now=self.training
                )
            else:
                values[node] = node.layer.forward(*arguments)

        object.__setattr__(
            self, "activations", {node.name: value for node, value in values.items()}
        )
        return values[output]

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        """
        Navigate the gradient back through the graph
        """
        gradients = {self.output: incoming_gradient}

        for node in reversed(self._nodes):
            if node.is_source or node not in gradients:
                # nothing downstream in the DAG
                continue

            returned = node.layer.backward(gradients.pop(node))
            parts = returned if len(node.sources) > 1 else (returned,)

            if len(parts) != len(node.sources):
                raise ValueError(
                    f"node {node.name!r} has {len(node.sources)} sources but "
                    f"its backward returned {len(parts)} gradients"
                )

            for source, part in zip(node.sources, parts):
                if source in gradients:
                    gradients[source] = gradients[source] + part
                else:
                    gradients[source] = part

        return gradients.get(self._input)

    def __call__(self, x_data: NDArray) -> NDArray:
        return self.forward(x_data)

    # ------------- inspection
    def edges(self) -> list[tuple[str, str]]:
        """every (producer, consumer) pair, for tracing or rendering"""
        return [
            (source.name, node.name)
            for node in self._nodes
            for source in node.sources
        ]

    def validate(self) -> list[str]:
        """
        Check for structural problem
        """
        problems = []
        output = self._output

        for node in self._nodes:
            if node.is_source or node is output:
                continue
            if not node.consumers:
                problems.append(
                    f"{node.name} feeds nothing and is not the output, so it "
                    "runs forward but never trains"
                )

        orphans = [
            layer
            for layer in self._registered
            if not any(node.layer is layer for node in self._nodes)
        ]
        for layer in orphans:
            problems.append(
                f"{layer.__class__.__name__} is registered but not connected"
            )
        return problems

    def train(self) -> NeuralNetwork:
        object.__setattr__(self, "training", True)
        return self

    def eval(self) -> NeuralNetwork:
        """
        switch to inference -- changes training behavior and training-specific behaviors
        """
        object.__setattr__(self, "training", False)
        return self

    @property
    def nodes(self) -> list[Node]:
        """every node including the input source, in construction order"""
        return list(self._nodes)

    @property
    def layers(self) -> list[Layer]:
        """
        return every registered layer in graph order
        """
        graph_nodes = [node.layer for node in self._nodes if not node.is_source]
        extra = [
            layer
            for layer in self._registered
            if not any(known is layer for known in graph_nodes)
        ]
        return graph_nodes + extra

    @property
    def num_parameters(self) -> int:
        total = 0
        for layer in self.layers:
            count = layer.num_parameters
            if count:
                total += count
        return total

    def purge(self) -> None:
        for layer in self.layers:
            layer.purge()
        object.__setattr__(self, "activations", {})

    def zero_gradients(self) -> None:
        for layer in self.layers:
            layer.zero_gradients()

    def shapes(self) -> dict[str, dict[str, tuple]]:
        """
        every node's declared input and output shapes, keyed by node name
        """
        return {
            node.name: {**node.layer.shapes, "resolved": node.out_shape}
            for node in self._nodes
            if not node.is_source
        }

    def summary(self, x_data: Optional[NDArray] = None) -> str:
        """
        Summary of the network -- shapshot view of the setup
        """
        if x_data is not None:
            self.forward(x_data)
        shapes = {
            name: np.shape(value) for name, value in self.activations.items()
        }

        listed = [node for node in self._nodes if not node.is_source]
        width = max((len(node.name) for node in listed), default=4)

        lines = [
            f"{self.name}: {len(listed)} nodes, {self.num_parameters} parameters"
        ]
        lines.append(f"  {'node'.ljust(width)}  {'sources':<26} shape")
        for node in listed:
            marker = " <- output" if node is self._output else ""
            lines.append(
                f"  {node.name.ljust(width)}  "
                f"{','.join(source.name for source in node.sources):<26} "
                f"{shapes.get(node.name, node.out_shape)}{marker}"
            )

        for problem in self.validate():
            lines.append(f"  warning: {problem}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return sum(1 for node in self._nodes if not node.is_source)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self)} nodes, "
            f"{self.num_parameters} parameters)"
        )


if __name__ == "__main__":
    from ml_tools.models.layers.layers import (
        DropoutLayer,
        FullyConnectedLayer,
        NormalizeLayer,
    )
    from ml_tools.models.layers.operators import LatentStack
    from ml_tools.models.model_loss import MSELoss
    from ml_tools.models.optimizers import SGD
    from ml_tools.generators import RandomDatasetGenerator

    generator = RandomDatasetGenerator(random_seed=42)
    x_data, y_data, _ = generator.generate(
        "regression", num_samples=600, num_features=6, noise_scale=0.5
    )
    y_data = y_data.reshape(-1, 1)

    # the input shape is declared, so the first edge is checked like the rest
    net = NeuralNetwork(name="two_branch", input_shape=(6,))

    features = net.input
    wide = net.connect(FullyConnectedLayer(6, 12, "relu"), features, name="wide")
    wide = net.connect(NormalizeLayer(12, shift_scale=True), wide, name="wide_norm")
    # `features` used a second time, so the fan-out is visible right here
    narrow = net.connect(FullyConnectedLayer(6, 4, "tanh"), features, name="narrow")

    merged = net.connect(LatentStack(), wide, narrow, name="merge")
    merged = net.connect(DropoutLayer(dropout_prob=0.1), merged, name="drop")
    merged = net.connect(FullyConnectedLayer(16, 8, "swish"), merged, name="head")
    net.output = net.connect(
        FullyConnectedLayer(8, 1, "linear", is_output=True), merged, name="out"
    )

    print(net.summary(x_data))
    print(f"\nedges: {net.edges()}")
    print(f"merge shapes: {net.shapes()['merge']}")

    try:
        net.connect(FullyConnectedLayer(9, 3, "relu"), net.node("merge"))
    except ValueError as refused:
        print(f"refused: {refused}")

    loss = MSELoss()
    optimizer = SGD(0.01)

    net.train()
    first = None
    for _ in range(300):
        prediction = net.forward(x_data)
        value = loss(prediction, y_data)
        if first is None:
            first = value
        net.backward(loss.backward())
        optimizer.step(net.layers)

    print(f"\ntraining loss {first:.4f} -> {value:.4f}")
    net.eval()
    print(f"eval loss (dropout off) {loss(net.forward(x_data), y_data):.4f}")
