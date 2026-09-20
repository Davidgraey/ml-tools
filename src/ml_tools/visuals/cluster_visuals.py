import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from numpy.typing import NDArray

COLORS = plt.get_cmap('tab20', 20)


def plot_clusters(x_data: NDArray,
                  labels: NDArray,
                  centroids: Optional[NDArray]=None) -> None:
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
    plt.title('Centroid Neural Network Clustering')
    # unique labels rather than range(ptp+1): a plain 0..k-1 prediction and a
    # generator label set that also carries -1 (outliers, see
    # RandomDatasetGenerator's outlier_fraction) both work, and centroids
    # stay indexed by real cluster id since -1 has none
    unique_labels = np.unique(labels)
    color_for = {label: idx for idx, label in enumerate(unique_labels)}

    for label in unique_labels:
        cluster_points = x_data[labels == label]
        name = 'Outliers' if label < 0 else f'Cluster {label + 1}'
        ax.scatter(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            color=COLORS(color_for[label]),
            label=name,
            alpha=0.8
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
                marker='X',
                edgecolor='black',
                linewidth=2,
                s=200,
                alpha=0.66)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()
