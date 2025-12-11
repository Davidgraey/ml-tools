import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, pdist
from ml_tools.models.clustering.cluster_metrics import silhouette_score
from ml_tools.types import BasalModel
EPSILON = 1e-12


class CentroidNeuralNetwork(BasalModel):
    def __init__(self,
                 max_clusters: int,
                 seed: int = 42,
                 initial_clusters: NDArray | None = None,
                 epsilon: float = 2e-3):
        super().__init__(seed=seed)
        self.max_clusters = max_clusters
        self.epsilon = epsilon

        if initial_clusters is not None:
            self.centroids = initial_clusters
        else:
            self.centroids = None
        self.previous_labels = None
        self.num_seen_samples: int = 0

        self._label_tracker = {}
        self._centroid_tracker = {}

        self._is_fitted = False

        self.metrics = {k: 0.0 for k in range(2, max_clusters)}

    @staticmethod
    def distance_metric(data_x: NDArray, data_y: NDArray) -> NDArray:
        """
        Calculate the distance metric between data points
        Parameters
        ----------
        data_x :
        data_y :

        Returns
        -------

        """
        distances = cdist(data_x, data_y, metric='sqeuclidean')
        return distances

    def batch_update(
            self,
            num_centroids: int,
            x_data: NDArray,
            new_labels: NDArray,
            batch_mask:NDArray
    ) -> int:
        previous_labels = self._label_tracker.get(num_centroids)
        changed_mask = new_labels != previous_labels[batch_mask]
        changed_count = np.sum(changed_mask)

        # Update the position / weight for each centroid based on their new membership "winners"
        for cluster_idx in range(num_centroids):
            indices = np.where(new_labels == cluster_idx)[0]
            num_members = len(indices)
            if num_members > 0:
                self.centroids[cluster_idx] = np.mean(x_data[indices], axis=0)

        self._update_labels(num_centroids, closest_centroids=new_labels, shuffle_mask=batch_mask)
        # self._label_tracker[k][batch_mask] = new_labels

        return changed_count

    def _update_labels(
            self,
            num_clusters: int,
            closest_centroids: NDArray,
            shuffle_mask: NDArray
    ) -> None:
        if num_clusters not in self._label_tracker.keys():
            self._label_tracker[num_clusters] = np.zeros_like(closest_centroids)

        self._label_tracker[num_clusters][shuffle_mask] = closest_centroids
        pass

    def fit(
        self,
        org_x_data: NDArray,
        verbose: bool = False,
        num_iterations: int = 25,
        fast_forward: bool = True,
        mini_batch: bool = False
    ) -> tuple[NDArray, NDArray]:
        """
        Fit the clustering model to the data, return the centroids and labels for each data point.
        Parameters
        ----------
        org_x_data : numpy.ndarray
        verbose : bool Verbose will trigger plotting and additional outputs during the fit process
        fast_forward : bool - if True, will skip metric calculations and plotting

        Returns
        -------
        the unstandardized centroid locations and the class labels for each data point
        """
        # check and see if we've done the standardization process:
        self.init_standardize(org_x_data)

        x_data = self.standardize(org_x_data)
        num_samples, num_features = x_data.shape

        # overwrite any prior labels -- set all to -1 (no cluster)
        previous_labels = np.full(num_samples, -1, dtype=int)
        starting_cent = np.mean(x_data, axis=0)

        # set up inertia for early termination
        if mini_batch and (num_samples > 500):
            batch_size = max((num_samples // 5), 500)
            print("processing in minibatches of size", batch_size)
            breakpoint = batch_size * 0.1
        else:
            batch_size = num_samples
            breakpoint = batch_size * 0.02

        # initalize the first two centroids ------------------
        if self.centroids is None:
            self.centroids = np.vstack([
                starting_cent + self.RNG.uniform(-self.epsilon, self.epsilon, size=num_features),
                starting_cent - self.RNG.uniform(-self.epsilon, self.epsilon, size=num_features)
            ])
        else:
            self.centroids = self.standardize(self.centroids)

        num_centroids = self.centroids.shape[0]
        self._label_tracker[num_centroids] = previous_labels
        self._batch = np.arange(0, num_samples)

        # =================== MAIN LOOP ====================
        while num_centroids <= self.max_clusters:
            # minibatch--------
            shuffle_mask = self.RNG.permutation(self._batch)[:batch_size].copy()
            xs = x_data[shuffle_mask, ...]
            for i in range(0, num_iterations):
                # get the "previous labels" --------------------------
                closest_centroids = self.forward(xs, num_centroids, axis=-1)

                changed_count = self.batch_update(
                    num_centroids=num_centroids,
                    x_data=xs,
                    new_labels=closest_centroids,
                    batch_mask=shuffle_mask
                )

                # early convergence:
                if breakpoint >= changed_count:
                    print(f"breaking from iterations at step {i} ")
                    break

            if num_centroids == self.max_clusters:
                print(f"breaking from iterations at {num_centroids} centroids created ")
                break

            # calculate loss
            error = self.calculate_loss(
                x_data=xs,
                closest_centroids=closest_centroids,
                num_centroids=num_centroids
            )

            if verbose and (fast_forward is False):
                plot_clusters(xs, self.centroids[:num_centroids], closest_centroids)

            # Splitting the highest error cluster (the "worst fit" cluster)
            split_target = np.argmax(error)
            self.centroids = np.vstack([
                self.centroids[:num_centroids],
                self.centroids[split_target] + self.RNG.uniform(-self.epsilon, self.epsilon, size=num_features)
            ])

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

            # cleanup, update & ending of the loop ---------------
            self._centroid_tracker.update({num_centroids: self.centroids[:num_centroids].copy()})
            num_centroids += 1
            # carry this latest "closest" forward into next loop
            self._update_labels(num_centroids, closest_centroids, shuffle_mask)

        # ending conditions ========================================
        self._is_fitted = True

        # Calculate for all datapoints ---- do we need this?
        # full_distances = self.distance_metric(data_x=x_data, data_y=self.centroids[:num_centroids])
        # closest_centroids = np.argmin(full_distances, axis=-1)
        # self._label_tracker[num_centroids] = closest_centroids.copy()

        return (
            self.unstandardize(self.centroids[:num_centroids, ...]),
            self._label_tracker[num_centroids]
            )

    def fit_predict(
            self,
            org_x_data: NDArray,
            verbose: bool = False,
            num_iterations: int = 25,
            fast_forward: bool = True,
            mini_batch: bool = False
        ) -> tuple[NDArray, NDArray]:
        """
        Fit the clustering model to the data, return the centroids and labels for each data point.
        Parameters
        ----------
        org_x_data : numpy.ndarray
        verbose : bool Verbose will trigger plotting and additional outputs during the fit process
        fast_forward : bool - if True, will skip metric calculations and plotting

        Returns
        -------
        the unstandardized centroid locations and the class labels for each data point
        """

        return self.fit(org_x_data, verbose, num_iterations, fast_forward, mini_batch)

    def forward(self, x_data: NDArray, num_centroids: int, axis: int=-1) -> NDArray:
        """
        Calculate the closest centroid to each sampl in x_data

        Parameters
        ----------
        x_data : the data array
        num_centroids : integer, the number of centroids
        axis : which axis to collect across (usually -1)

        Returns
        -------
        array of integers (cluster membership for the closest centroid to each datapoint
        """
        full_distances = self.distance_metric(
            data_x=x_data,
            data_y=self.centroids[:num_centroids],
        )

        closest_centroids = np.argmin(full_distances, axis=axis)

        return closest_centroids

    def predict(self, x_data: NDArray) -> tuple[int, NDArray, NDArray]:
        """
        Standardize and predict the x_data using the already fitted parameters of the centroid nnet

        Parameters
        ----------
        x_data : input sample data -> must match the dimension count of the data used during fitting

        Returns
        -------
        the results of get_optimal -> optimal number of centroids, the centroid positions, and the centroid
        membership (integer per x data sample)
        """

        xs = self.standardize(x_data)
        for num_centroids in range(0, self.max_clusters):
            closest_centroids = self.forward(
                x_data=xs,
                num_centroids=num_centroids,
                axis=-1
            )
            self._update_labels(
                num_clusters=num_centroids,
                closest_centroids=closest_centroids,
                shuffle_mask=np.arange(0, xs.shape[0])
            )

        return self.get_optimal()

    # predictions, targets
    def calculate_loss(
            self,
            x_data: NDArray,
            closest_centroids: NDArray,
            num_centroids: int
    ) -> NDArray:
        """

        Parameters
        ----------
        x_data :
        closest_centroids :
        num_centroids :

        Returns
        -------

        """
        error = np.zeros(num_centroids)
        for cluster_idx in range(0, num_centroids):
            mask = (closest_centroids == cluster_idx)
            if np.any(mask):
                # distance metric --
                error[cluster_idx] = np.sum(
                    self.distance_metric(
                        x_data[mask],
                        self.centroids[cluster_idx].reshape(1, -1)
                    )
                )
            else:
                error[cluster_idx] = 0
        return error

    def get_optimal(self) -> tuple[int, NDArray, NDArray]:
        """
        Get the optimal number of clusters based on the metrics collected during fitting.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        best_scoring = max(self.metrics, key=self.metrics.get)
        _cents = self.unstandardize(self._centroid_tracker.get(best_scoring))
        return (
            best_scoring,
            _cents[:best_scoring, ...],
            self._label_tracker[best_scoring]
        )


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
    plt.title('Centroid Neural Network Clustering')
    current_k = len(centroids)
    colors = plt.get_cmap('tab20', current_k)
    for clust_idx in range(current_k):
        cluster_points = x_data[labels == clust_idx]
        ax.scatter(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            color=colors(clust_idx),
            label=f'Cluster {clust_idx + 1}',
            alpha=0.8
        )

    for clust_idx in range(current_k):
        ax.scatter(
            x=centroids[clust_idx, 0],
            y=centroids[clust_idx, 1],
            color=colors(clust_idx),
            marker='X',
            edgecolor='black',
            linewidth=2,
            s=200,
            alpha=0.66)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()


# Example usage:
if __name__ == "__main__":
    import time
    from ml_tools.generators import RandomDatasetGenerator
    from ml_tools.models.clustering.cluster_metrics import homogeneity

    gen = RandomDatasetGenerator(random_seed=42)
    X, y, meta = gen.generate(task="clustering",
                              num_samples=2000,
                              num_features=2,
                              noise_scale=0.22,
                              num_clusters=8,
                              verbose=True
                              )

    plot_clusters(X, meta["centroids"], y)
    plt.show()
    st = time.time()

    # for eps in np.arange(3, 5):
    cnn = CentroidNeuralNetwork(max_clusters=20, epsilon=0.2)
    centroids, labels = cnn.fit_predict(X, num_iterations=512, mini_batch=False, verbose=True, fast_forward=False)
    print(cnn.metrics)
    print("verbose took", (time.time() - st ) * 1000)
    plot_clusters(X, centroids, labels)
    clust_count, opt_centroids, opt_labels = cnn.get_optimal()
    plot_clusters(X, opt_centroids, opt_labels)

    print(f"homogenity score vs labels: {homogeneity(y, opt_labels)}")

    # for eps in np.arange(2, 5):
    #     st = time.time()
    #     cnn = CentroidNeuralNetwork(max_clusters=10, epsilon=1/eps)
    #     centroids, labels = cnn.fit_predict(X, verbose=False, fast_forward=True)
    #     print("EPS: ", 1/eps)
    #     print(cnn.metrics)
    #     print("verbose took", (time.time() - st) * 1000)
    #     plt.scatter(X[:, 0], X[:, 1])
    #     plot_clusters(X, centroids, labels)
    #     plt.show()
