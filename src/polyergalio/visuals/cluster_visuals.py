from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from polyergalio.visuals.animation_utils import save_and_show, thin_history
from numpy.typing import NDArray

COLORS = plt.get_cmap("tab20", 20)


def plot_clusters(
    x_data: NDArray, labels: NDArray, centroids: Optional[NDArray] = None
) -> None:
    """
    plot the clusters
    Parameters
    ----------
    x_data : - the data points to plot
    centroids : our extracted centroids from the centnn process
    labels : labels for each point in x_data- giving it's cluster membership as an integer

    Returns
    -------
    None
    """
    # ax = plt.figure(figsize=(8, 6)).add_subplot(projection='3d')
    ax = plt.figure(figsize=(8, 6)).add_subplot()
    plt.title("Centroid Neural Network Clustering")
    # unique labels rather than range(ptp+1): a plain 0..k-1 prediction and a
    # generator label set that also carries -1 (outliers, see
    # RandomDatasetGenerator's outlier_fraction) both work, and centroids
    # stay indexed by real cluster id since -1 has none
    unique_labels = np.unique(labels)
    color_for = {label: idx for idx, label in enumerate(unique_labels)}

    for label in unique_labels:
        cluster_points = x_data[labels == label]
        name = "Outliers" if label < 0 else f"Cluster {label + 1}"
        ax.scatter(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            color=COLORS(color_for[label]),
            label=name,
            alpha=0.8,
        )
    if centroids is None:
        pass
    else:
        for label in unique_labels:
            if label < 0 or label >= len(centroids):
                continue
            ax.scatter(
                x=centroids[label, 0],
                y=centroids[label, 1],
                color=COLORS(color_for[label]),
                marker="X",
                edgecolor="black",
                linewidth=2,
                s=200,
                alpha=0.66,
            )

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()


def plot_final_predictions(
    scenario_name: str, x_data: NDArray, predictions: dict
) -> None:
    """
    One grid figure, one subplot per mechanism, colored by its own final prediction.

    Parameters
    ----------
    scenario_name : title label
    x_data : the data points that were clustered
    predictions : mechanism name -> its predicted cluster labels for x_data
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f"{scenario_name}: final predicted clusters")
    for ax, (name, prediction) in zip(axes.flat, predictions.items()):
        for color_idx, label in enumerate(np.unique(prediction)):
            points = x_data[prediction == label]
            name_label = "Outliers" if label < 0 else f"Cluster {label + 1}"
            ax.scatter(
                points[:, 0],
                points[:, 1],
                color=COLORS(color_idx),
                s=10,
                alpha=0.7,
                label=name_label,
            )
        ax.set_title(f"{name} ({len(np.unique(prediction))} clusters)")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def _draw_grid_mesh(ax, weights: NDArray, shape: tuple) -> None:
    """The row/column mesh a PLSOM-family grid draws, on shared axes."""
    rows, cols = shape
    ws = weights.reshape(rows, cols, -1)
    for row in range(rows):
        ax.plot(
            ws[row, :, 0],
            ws[row, :, 1],
            "o-",
            color="crimson",
            markersize=4,
            linewidth=1,
        )
    for col in range(cols):
        ax.plot(
            ws[:, col, 0],
            ws[:, col, 1],
            "o-",
            color="crimson",
            markersize=4,
            linewidth=1,
        )


def animate_growth(
    scenario_name: str,
    histories: dict,
    interval_ms: int = 120,
    save_path: Optional[str] = None,
    show: bool = True,
    stride: int = 1,
) -> FuncAnimation:
    """
    One grid animated figure showing how each mechanism's prototypes or
    centroids evolve -- one subplot per mechanism, stepping through that
    mechanism's own training epochs / growth steps. A mechanism with fewer
    steps than the others just holds its last frame once it runs out.

    Every panel is drawn the same way -- a static, unchanging data scatter
    with that mechanism's evolving prototypes/centroids on top -- so the
    mechanisms are a fair side by side comparison of *process*, not result
    (plot_final_predictions already covers the final result).

    A snapshot is read by which keys it carries: 'centroids'+'num_centroids'
    (e.g. CentroidNeuralNetwork.growth_history()), 'weights'+'shape' (a fixed
    grid, e.g. PLSOM/GPLSOM), or plain 'weights' (a free neuron set, e.g.
    FreePLSOM). Every snapshot also carries 'background', the data scatter.

    Parameters
    ----------
    scenario_name : title label
    histories : mechanism name -> list of per-step snapshots
    save_path : if given, the animation is written there as a GIF (via
        matplotlib's built-in Pillow writer -- no external encoder needed)
        before it is shown
    show : whether to also open the interactive window. False is useful
        for a headless/batch run that only wants the file on disk
    stride : keep only every `stride`-th snapshot per mechanism (the final
        snapshot is always kept). Fewer frames means less drawing and a
        smaller, faster GIF; 1 keeps every step

    Returns
    -------
    the FuncAnimation
    """
    histories = {
        name: thin_history(history, stride) for name, history in histories.items()
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = dict(zip(histories.keys(), axes.flat))
    fig.suptitle(f"{scenario_name}: growth over time")

    num_frames = max((len(h) for h in histories.values()), default=1)

    def update(frame):
        for name, ax in axes.items():
            history = histories[name]
            if not history:
                continue
            snap = history[min(frame, len(history) - 1)]
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.scatter(
                snap["background"][:, 0],
                snap["background"][:, 1],
                color="lightgray",
                s=8,
                alpha=0.5,
            )

            if "centroids" in snap:
                ax.scatter(
                    snap["centroids"][:, 0],
                    snap["centroids"][:, 1],
                    color="crimson",
                    marker="X",
                    s=80,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.set_title(f"{name} ({snap['num_centroids']} centroids)")
            elif "shape" in snap:
                _draw_grid_mesh(ax, snap["weights"], snap["shape"])
                ax.set_title(f"{name} {snap['shape'][0]}x{snap['shape'][1]}")
            else:
                ax.scatter(
                    snap["weights"][:, 0],
                    snap["weights"][:, 1],
                    color="crimson",
                    s=40,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.set_title(f"{name} ({len(snap['weights'])} neurons)")
        return []

    anim = FuncAnimation(
        fig, update, frames=num_frames, interval=interval_ms, repeat=True
    )
    plt.tight_layout()
    save_and_show(anim, save_path, interval_ms, show)
    return anim


def animate_membership(
    scenario_name: str,
    x_data: NDArray,
    histories: dict,
    interval_ms: int = 120,
    save_path: Optional[str] = None,
    show: bool = True,
    stride: int = 1,
) -> FuncAnimation:
    """
    The membership counterpart to animate_growth: one grid animated figure
    showing how each mechanism's *predicted cluster assignment* for every
    point evolves over training/growth -- the per-epoch version of
    plot_final_predictions, instead of the per-epoch prototypes/centroids
    animate_growth shows.

    Each history snapshot needs a per-point cluster label already attached
    -- 'prediction' (grid-based mechanisms, computed only for the steps
    `stride` keeps) or 'labels' (e.g. CentroidNeuralNetwork.growth_history()).
    A snapshot with neither is skipped.

    Parameters
    ----------
    scenario_name : title label
    x_data : the data points that were clustered
    histories : mechanism name -> list of per-step snapshots
    save_path : if given, the animation is written there as a GIF (via
        matplotlib's built-in Pillow writer -- no external encoder needed)
        before it is shown
    show : whether to also open the interactive window
    stride : keep only every `stride`-th snapshot per mechanism (the final
        snapshot is always kept). Should match the stride used to produce
        the 'prediction' entries, otherwise some kept frames have none

    Returns
    -------
    the FuncAnimation
    """
    histories = {
        name: thin_history(history, stride) for name, history in histories.items()
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = dict(zip(histories.keys(), axes.flat))
    fig.suptitle(f"{scenario_name}: predicted membership over time")

    num_frames = max((len(h) for h in histories.values()), default=1)

    def update(frame):
        for name, ax in axes.items():
            history = histories[name]
            if not history:
                continue
            snap = history[min(frame, len(history) - 1)]
            labels = snap.get("prediction", snap.get("labels"))
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            if labels is None:
                ax.set_title(f"{name} (no membership yet)")
                continue
            for color_idx, label in enumerate(np.unique(labels)):
                points = x_data[labels == label]
                ax.scatter(
                    points[:, 0], points[:, 1], color=COLORS(color_idx), s=10, alpha=0.7
                )
            title = (
                f"{name} ({snap['num_centroids']} centroids)"
                if "num_centroids" in snap
                else f"{name} ({len(np.unique(labels))} clusters)"
            )
            ax.set_title(title)
        return []

    anim = FuncAnimation(
        fig, update, frames=num_frames, interval=interval_ms, repeat=True
    )
    plt.tight_layout()
    save_and_show(anim, save_path, interval_ms, show)
    return anim
