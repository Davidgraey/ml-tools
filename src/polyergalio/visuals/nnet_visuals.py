import textwrap
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from polyergalio.models.neural_network import NeuralNetwork, Node


def plot_network(
    network: NeuralNetwork,
    ax: Optional[Axes] = None,
    figsize: tuple = (9, 12),
    x_spacing: float = 2.0,
    y_spacing: float = 2.5,
    box_width: float = 1.5,
    box_height: float = 0.8,
    fontsize: float = 8,
) -> Axes:
    """
    Draw the graph's nodes and edges, top to bottom.

    Nodes are placed in rows by their longest-path depth from the input, so
    a fan-in node sits below every one of its sources; within a row, nodes
    are spread horizontally and centered. Depth can be computed in a single
    pass over self._nodes because connect() only ever appends a node after
    its sources are already present, so every source's depth is known
    before it's needed.

    Edges that span more than one row (residual/skip connections) are drawn
    with wider curvature and a dashed line, since a straight or gently
    curved line for a multi-row edge is visually indistinguishable from an
    adjacent single-row edge and tends to pass directly through the boxes
    in between.

    Parameters
    ----------
    ax : existing axes to draw on; a new figure is created if omitted
    figsize : figure size when ax is not given
    x_spacing : horizontal distance between adjacent nodes in a row
    y_spacing : vertical distance between rows
    box_width, box_height : node box size, in the same units as the spacing
    fontsize : label font size

    Returns
    -------
    the axes drawn on
    """
    nodes = network._nodes
    depth = {}
    for node in nodes:
        depth[node] = (
            0 if node.is_source else max(depth[source] for source in node.sources) + 1
        )

    rows: dict[int, list] = {}
    for node in nodes:
        rows.setdefault(depth[node], []).append(node)

    positions = {}
    for y, row_nodes in rows.items():
        count = len(row_nodes)
        for i, node in enumerate(row_nodes):
            positions[node] = ((i - (count - 1) / 2) * x_spacing, -y * y_spacing)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for node, (x, y) in positions.items():
        is_output = node is network._output
        color = "#ffd166" if node.is_source else ("#06d6a0" if is_output else "#118ab2")
        label = (
            node.name if node.is_source else f"{node.name}\n{type(node.layer).__name__}"
        )
        label = "\n".join(textwrap.fill(line, width=14) for line in label.split("\n"))
        ax.add_patch(
            FancyBboxPatch(
                (x - box_width / 2, y - box_height / 2),
                box_width,
                box_height,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="black",
                zorder=2,
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, zorder=3)

    # alternate curve direction per span so multiple skip edges of the same
    # length fan out instead of overlapping each other
    span_seen: dict[int, int] = {}

    for node in nodes:
        if node.is_source:
            continue
        x_to, y_to = positions[node]
        for source in node.sources:
            x_from, y_from = positions[source]
            span = depth[node] - depth[source]

            if span > 1:
                order = span_seen.get(span, 0)
                span_seen[span] = order + 1
                sign = 1 if order % 2 == 0 else -1
                rad = sign * (0.25 + 0.1 * (order // 2))
                style = dict(linestyle="--", color="#ef476f", linewidth=1.3)
            else:
                rad = 0.1
                style = dict(linestyle="-", color="gray", linewidth=1.0)

            ax.add_patch(
                FancyArrowPatch(
                    (x_from, y_from - box_height / 2),
                    (x_to, y_to + box_height / 2),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    zorder=1,
                    connectionstyle=f"arc3,rad={rad}",
                    **style,
                )
            )

    xs = [x for x, _ in positions.values()]
    ax.set_xlim(min(xs) - box_width, max(xs) + box_width)
    ax.set_ylim(-max(rows) * y_spacing - box_height, box_height)
    ax.set_title(network.name)
    ax.axis("off")
    return ax
