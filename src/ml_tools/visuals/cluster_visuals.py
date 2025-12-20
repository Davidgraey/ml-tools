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
    current_k = np.ptp(labels) + 1

    for clust_idx in range(current_k):
        cluster_points = x_data[labels == clust_idx]
        ax.scatter(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            color=COLORS(clust_idx),
            label=f'Cluster {clust_idx + 1}',
            alpha=0.8
        )
    if centroids is None:
        pass
    else:
        for clust_idx in range(current_k):
            ax.scatter(
                x=centroids[clust_idx, 0],
                y=centroids[clust_idx, 1],
                color=COLORS(clust_idx),
                marker='X',
                edgecolor='black',
                linewidth=2,
                s=200,
                alpha=0.66)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()
