"""Box-and-arrow diagrams for models that are not NeuralNetwork graphs."""

import textwrap

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SOURCE_COLOR = "#ffd166"
STEP_COLOR = "#118ab2"
SINK_COLOR = "#06d6a0"


def plot_graph(
    edges: list,
    title: str = "",
    x_spacing: float = 3.2,
    y_spacing: float = 1.8,
    box_width: float = 2.8,
    box_height: float = 1.0,
) -> Figure:
    """
    Layered top-to-bottom diagram of labelled boxes joined by arrows.

    Parameters
    ----------
    edges : (source, target) label pairs forming an acyclic graph
    title : figure title
    x_spacing, y_spacing : distance between boxes in a row and between rows
    box_width, box_height : box size

    Returns
    -------
    Figure
    """
    nodes = list(dict.fromkeys(name for edge in edges for name in edge))
    parents = {node: [s for s, t in edges if t == node] for node in nodes}
    children = {node: [t for s, t in edges if s == node] for node in nodes}
    depth = {}

    def level(node):
        if node not in depth:
            depth[node] = 1 + max((level(p) for p in parents[node]), default=-1)
        return depth[node]

    rows = {}
    for node in nodes:
        rows.setdefault(level(node), []).append(node)

    positions = {}
    for row, members in rows.items():
        for index, node in enumerate(members):
            positions[node] = ((index - (len(members) - 1) / 2) * x_spacing, -row * y_spacing)

    columns = max(len(members) for members in rows.values())
    width = (columns - 1) * x_spacing + box_width + 1
    height = (len(rows) - 1) * y_spacing + box_height + 1
    fig, ax = plt.subplots(figsize=(width * 0.55, height * 0.55))

    for node, (x, y) in positions.items():
        color = SOURCE_COLOR if not parents[node] else SINK_COLOR if not children[node] else STEP_COLOR
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
        ax.text(x, y, textwrap.fill(node, width=22), ha="center", va="center", fontsize=8, zorder=3)

    for source, target in edges:
        x_from, y_from = positions[source]
        x_to, y_to = positions[target]
        skips = depth[target] - depth[source] > 1
        ax.add_patch(
            FancyArrowPatch(
                (x_from, y_from - box_height / 2),
                (x_to, y_to + box_height / 2),
                arrowstyle="-|>",
                mutation_scale=12,
                connectionstyle="arc3,rad=0.35" if skips else "arc3,rad=0",
                linestyle="--" if skips else "-",
                color="#ef476f" if skips else "gray",
                zorder=1,
            )
        )

    xs = [x for x, _ in positions.values()]
    ax.set_xlim(min(xs) - box_width / 2 - 0.5, max(xs) + box_width / 2 + 0.5)
    ax.set_ylim(-(len(rows) - 1) * y_spacing - box_height / 2 - 0.5, box_height / 2 + 0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")
    return fig
