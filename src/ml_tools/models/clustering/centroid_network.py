import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from ml_tools.models.clustering.cluster_metrics import silhouette_score
EPSILON = 1e-12


# TODO: fucked up the plotting?  Or the "closest labels" --> centroids appear good, but labels are wack
class CentroidNeuralNetwork:
    def __init__(self,
                 max_clusters: int,
                 seed: int = 42,
                 initial_clusters: NDArray | None = None,
                 epsilon: float = 2e-3):
        self.max_clusters = max_clusters
        self.epsilon = epsilon
        self.RNG = np.random.RandomState(seed)

        if initial_clusters is not None:
            self.centroids = initial_clusters
        else:
            self.centroids = None
        self.previous_labels = None
        self.x_means = None
        self.x_stds = None
        self._label_tracker = {}
        self._centroid_tracker = {}

        self._is_fitted = False

        self.metrics = {k: 0.0 for k in range(2, max_clusters)}

    @staticmethod
    def distance_metric(data_x: NDArray, data_y: NDArray, axis: int = -1) -> NDArray:
        """
        Calculate the distance metric between data points
        """
        distances = cdist(data_x, data_y, 'sqeuclidean')
        return distances

    def standardize(self, data_array: NDArray) -> NDArray:
        assert data_array.shape[-1] == self.x_means.shape[0]

        return (data_array - self.x_means) / (self.x_stds + EPSILON)

    def unstandardize(self, data_array: NDArray) -> NDArray:
        assert data_array.shape[-1] == self.x_means.shape[0]

        return (self.x_stds - EPSILON) * data_array + self.x_means

    def init_standardize(self, x_data: NDArray) -> None:
        self.x_means = np.mean(x_data, axis=0)
        self.x_stds = np.std(x_data, axis=0)
        pass

    def batch_update(self, x_data, new_labels, batch_mask, batch=True) -> int:
        k = self.centroids.shape[0]
        changed_mask = new_labels != self.previous_labels[batch_mask]
        changed_count = np.sum(changed_mask)
        # Update the position / weight for each centroid based on their new membership "winners"
        if batch is True:
            for cluster_idx in range(k):
                indices = np.where(new_labels == cluster_idx)[0]
                num_members = len(indices)
                if num_members > 0:
                    self.centroids[cluster_idx] = np.mean(x_data[indices], axis=0)

        return changed_count

    def fit_predict(self,
            x_data: NDArray,
            verbose: bool = False,
            num_iterations: int = 25,
            fast_forward: bool = True,
            mini_batch: bool = False) -> tuple[NDArray, NDArray]:
        """
        Fit the clustering model to the data, return the centroids and labels for each data point.
        Parameters
        ----------
        x_data : numpy.ndarray
        verbose : bool Verbose will trigger plotting and additional outputs during the fit process
        fast_forward : bool - if True, will skip metric calculations and plotting

        Returns
        -------
        the unstandardized centroid locations and the class labels for each data point
        """
        # check and see if we've done the standardization process:
        if self.x_means is None:
            self.init_standardize(x_data)
        else:
            pass

        x_data = self.standardize(x_data)
        num_samples, num_features = x_data.shape

        # overwrite any prior labels -- set all to -1 (no cluster)
        self.previous_labels = np.full(num_samples, -1, dtype=int)
        starting_cent = np.mean(x_data, axis=0)

        # set up inertia for early termination
        if mini_batch and (num_samples > 500):
            batch_size = max((num_samples // 5), 500)
            print("processing in minibatches of size", batch_size)
            breakpoint = batch_size * 0.1
        else:
            batch_size = num_samples
            breakpoint = batch_size * 0.1

        # initalize the first two centroids--
        if self.centroids is None:
            self.centroids = np.vstack([
                starting_cent + self.RNG.uniform(-self.epsilon, self.epsilon, size=num_features),
                starting_cent - self.RNG.uniform(-self.epsilon, self.epsilon, size=num_features)
            ])
        else:
            self.centroids = self.standardize(self.centroids)

        num_centroids = self.centroids.shape[0]
        print(num_centroids)
        while num_centroids <= self.max_clusters:
            # minibatch
            mini_mask = self.RNG.randint(0, num_samples, size=batch_size)
            xs = x_data[mini_mask, ...]

            for i in range(0, num_iterations):
                # print(i)
                distances = self.distance_metric(
                    data_x=xs,
                    data_y=self.centroids[:num_centroids],
                )
                closest_centroids = np.argmin(distances, axis=-1)

                # early convergence:
                if breakpoint >= np.sum(closest_centroids != self.previous_labels[mini_mask]):
                    break

                # "loser count" -- the points which have changed their cluster membership
                _loser_count = self.batch_update(xs, closest_centroids, batch_mask=mini_mask)
                # plot_clusters(xs, self.centroids[:num_centroids], closest_centroids)

            if num_centroids == self.max_clusters:
                break

            error = np.zeros(num_centroids)
            for cluster_idx in range(num_centroids):
                mask = (closest_centroids == cluster_idx)
                if np.any(mask):
                    error[cluster_idx] = np.sum(((xs[mask] - self.centroids[cluster_idx]) ** 2))
                else:
                    error[cluster_idx] = 0

            if verbose and (fast_forward is False):
                plot_clusters(x_data, self.centroids[:num_centroids], closest_centroids)

            # Splitting the highest error cluster (the "worst fit" cluster)
            split_target = np.argmax(error)
            self.centroids = np.vstack([
                self.centroids[:num_centroids],
                self.centroids[split_target] + self.RNG.uniform(-self.epsilon, self.epsilon, size=num_features)
            ])

            # cleanup ending loop ---
            self._label_tracker.update({num_centroids: closest_centroids.copy()})
            self._centroid_tracker.update({num_centroids: self.centroids[:num_centroids].copy()})

            # these metric calculations take time - if we know a target, we can 'fast forward' and skip metrics
            if fast_forward is False:
                self.metrics.update(
                    {
                        num_centroids: silhouette_score(
                            xs,
                            closest_centroids
                        )
                    }
                )
                print(self.metrics[num_centroids])
            num_centroids += 1
        self._is_fitted = True

        # Calculate for all datapoints ----
        full_distances = self.distance_metric(data_x=x_data, data_y=self.centroids[:num_centroids])
        closest_centroids = np.argmin(full_distances, axis=-1)
        print(num_centroids)
        return self.unstandardize(self.centroids[:num_centroids-1, ...]), self._label_tracker.get(num_centroids-1, None)

    def get_optimal(self) -> tuple[int, NDArray, NDArray]:
        """
        Get the optimal number of clusters based on the metrics collected during fitting.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        best_scoring = max(self.metrics, key=self.metrics.get)
        _cents = self.unstandardize(self._centroid_tracker.get(best_scoring))
        return best_scoring, _cents[:best_scoring, ...], self._label_tracker[best_scoring]


def plot_clusters(x_data: NDArray, centroids: NDArray, labels: NDArray) -> None:
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
    k = 15
    current_k = len(centroids)
    colors = plt.get_cmap('tab20', current_k)
    for clust_idx in range(current_k):
        cluster_points = x_data[labels == clust_idx]
        ax.scatter(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            # zs=cluster_points[:, 0],
            color=colors(clust_idx),
            label=f'Cluster {clust_idx + 1}',
            alpha=0.25)
        plt.show()
    for clust_idx in range(current_k):
        ax.scatter(
            x=centroids[clust_idx, 0],
            y=centroids[clust_idx, 1],
            # zs=centroids[clust_idx, 0],
            color=colors(clust_idx),
            marker='X',
            edgecolor='black',
            linewidth=2,
            s=200,
            alpha=0.75)

    plt.title('Centroid Neural Network Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()



# Example usage:
if __name__ == "__main__":
    import time
    from ml_tools.generators import RandomDatasetGenerator

    gen = RandomDatasetGenerator(random_seed=42)
    X, y, meta = gen.generate(task="clustering",
                              num_samples=1000,
                              num_features=2,
                              noise_scale=0.33,
                              num_clusters=3,
                              verbose=True
                              )

    plot_clusters(X, meta["centroids"], y)
    plt.show()
    st = time.time()

    for eps in np.arange(3, 5):
        cnn = CentroidNeuralNetwork(max_clusters=eps, epsilon=0.1)
        centroids, labels = cnn.fit_predict(X, verbose=False, fast_forward=False)
        print(cnn.metrics)
        print("verbose took", (time.time() - st ) * 1000)
        plot_clusters(X, centroids, labels)
        # clust_count, opt_centroids, opt_labels = cnn.get_optimal()
        # plot_clusters(X, opt_centroids, opt_labels)


    # for eps in np.arange(2, 5):
    #     st = time.time()
    #     cnn = CentroidNeuralNetwork(max_clusters=10, epsilon=1/eps)
    #     centroids, labels = cnn.fit_predict(X, verbose=False, fast_forward=False)
    #     print("EPS: ", 1/eps)
    #     print(cnn.metrics)
    #     print("verbose took", (time.time() - st) * 1000)
    #     plt.scatter(X[:, 0], X[:, 1], color=labels)
    #     plot_clusters(X, centroids, labels)
    #     plt.show()
