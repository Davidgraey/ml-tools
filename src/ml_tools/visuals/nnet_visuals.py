import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from ml_tools.models.neural_network import NeuralNetwork, Node

def plot(network: NeuralNetwork, ax: Optional[Axes] = None, figsize: tuple = (10, 6)) -> Axes:
    """
    Draw the graph's nodes and edges.

    Nodes are placed in columns by their longest-path depth from the
    input, so a fan-in node sits to the right of every one of its
    sources; within a column, nodes are spread vertically and centered.
    Depth can be computed in a single pass over self._nodes because
    connect() only ever appends a node after its sources are already
    present, so every source's depth is known before it's needed.

    Parameters
    ----------
    ax : existing axes to draw on; a new figure is created if omitted
    figsize : figure size when ax is not given

    Returns
    -------
    the axes drawn on
    """
    nodes = self._nodes
    depth = {}
    for node in nodes:
        depth[node] = (
            0 if node.is_source
            else max(depth[source] for source in node.sources) + 1
        )

    columns: dict[int, list] = {}
    for node in nodes:
        columns.setdefault(depth[node], []).append(node)

    positions = {}
    for x, column_nodes in columns.items():
        count = len(column_nodes)
        for i, node in enumerate(column_nodes):
            positions[node] = (x, i - (count - 1) / 2)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for node, (x, y) in positions.items():
        is_output = node is self._output
        color = "#ffd166" if node.is_source else ("#06d6a0" if is_output else "#118ab2")
        label = node.name if node.is_source else f"{node.name}\n{type(node.layer).__name__}"
        ax.add_patch(FancyBboxPatch(
            (x - 0.4, y - 0.2), 0.8, 0.4,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="black", zorder=2,
        ))
        ax.text(x, y, label, ha="center", va="center", fontsize=7, zorder=3)

    for node in nodes:
        if node.is_source:
            continue
        x_to, y_to = positions[node]
        for source in node.sources:
            x_from, y_from = positions[source]
            ax.add_patch(FancyArrowPatch(
                (x_from + 0.4, y_from), (x_to - 0.4, y_to),
                arrowstyle="-|>", mutation_scale=12, color="gray",
                zorder=1, connectionstyle="arc3,rad=0.1",
            ))

    ax.set_xlim(-1, max(columns) + 1)
    ys = [y for _, y in positions.values()]
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_title(self.name)
    ax.axis("off")
    return ax