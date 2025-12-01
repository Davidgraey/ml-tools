import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Local Files
import ml_tools.models.clustering.plsom_utils as plsom_utils
from ml_tools.models.clustering.centroid_network import CentroidNeuralNetwork, plot_clusters
from ml_tools.models.constants import EPSILON
from ml_tools.types import BasalModel


class PLSOM:
    """
    Parameterless Self Organizing Map
    https://arxiv.org/pdf/0705.0199
    Unlike a traditional SOM, the PLSOM has minimal configurable hyperparameters.
    ε(t)   = ||x(t) - w_c(t)||_2 / r(t)
    r(0) = ||x(0)-w_c(0)||_2
    r(t)   = max( ||x(t)-w_c(t)||_2, r(t-1) )
    Θ(ε)   = neighborhood scale from ε via Eq. (9), (10), or (11)
    h_ci   = exp( - d(i,c)^2 / Θ(ε)^2 )
    Δw_i   = ε * h_ci * (x - w_i)


    The PLSOM algorithm works by adjusting the weights of the neurons to approximate the input data.  In this
    process, we reduce the sample dimensionality by calculating which of the neurons ("primitives") is a best fit for a datapoint.
    This accomplishes two goals: reduction in dimensionality (fewer samples) and pseudo-clustering (each neuron is a prototype for a set of samples).
    The PLSOM's neurons can be easily clustered using a more traditional algorithm; leaving us with a
    sample:pseudo-clustering:cluster membership chain.
    """

    def __init__(self,
                 width: int,
                 height: int,
                 input_dim: int,
                 theta_min=None,
                 theta_max=None,
                 lock_seed: int = 42,
                 distance='euclidean'):
        """
        Create the Self Organizing Map  - rectangular / square grids

        Parameters
        ----------
        width : for simplicity's sake, we make map of width x height neurons.
        height : should be the same as width, number of neurons in h
        input_dim : dimensionality of our input data (should be shape[-1]) -- this is the original number of features
            that we'll be reprojecting through the SOM's dimensions
        theta_min : smallest neighborhood value - set at 0
        theta_max : Maximum neighborhood value - set at or close to width
        lock_seed :int, give an int to lock numpy seed
        distance :distance measure to use, can be 'euclidean', 'manhattan', 'cosine', 'hamming'
        """

        distance_dict = {'euclidean': plsom_utils.euclidian_distance,
                         'manhattan': plsom_utils.manhattan_distance,
                         'hamming': plsom_utils.hamming_distance,
                         'cosine': plsom_utils.cosine_distance}

        np.random.seed(lock_seed)
        self.N_DIMS = input_dim

        self.VERBOSE = True

        # variables that will need to be updated as we grow / shrink
        self.network_shape = [height, width]
        self.n_neurons = height * width
        self.grid_distances = self.build_grid()

        self.hit_map = np.zeros(shape=self.n_neurons)

        # Constants
        # minimum theta (within neighborhood influence)- 1 for alternate equations - might be worth trying both!
        self.THETAMIN = theta_min if theta_min else 1
        # Maximum value for neighborhood influence - also called Beta in some papers
        self.THETAMAX = theta_max if theta_max else width

        # TODO: add feedback messaging if invalid selection.
        self.distance_function = distance_dict[distance]

        self.n_iter = 0
        self.means = 0
        self.stds = 1
        self.q_error_trace = []
        self.epsilon_trace = []
        self.previous_step_r = 0

    def initialize_weights(self, x: NDArray, method='kaiming') -> NDArray:
        """
        We will use a nnet-style of initialization for weights
        Parameters
        ----------
        x : array of x dataponts
        method :  string - use 'kaiming', "he" or "sample_space"

        Returns
        -------
        returns the init weights -> to assing to self.weights
        """
        if method in ['kaiming', 'he']:
            # may need to scale down bound further!
            bound = np.sqrt(2 / self.N_DIMS)
            # all our values will be between 0 and 1 for categorical data
            return np.random.uniform(low=0, high=bound, size=(self.n_neurons, self.N_DIMS))

        elif method == 'sample_space':
            weights = np.zeros(shape=(self.n_neurons, self.N_DIMS))
            for dim in range(self.N_DIMS):
                low = np.min(x, axis=0)[dim]
                high = np.max(x, axis=0)[dim]
                weights[:, dim] = np.random.uniform(low=low, high=high, size=(self.n_neurons))
            return weights

        else:
            return np.random.uniform(low=-0.1, high=0.1, size=(self.n_neurons, self.N_DIMS))

    def pretrain_step(self, x: NDArray) -> NDArray:
        _x = copy.deepcopy(x)

        if self.n_iter == 0:
            self.weights = self.initialize_weights(_x, method='sample_space')
        return _x

    def fit(self, x: NDArray, epochs: int) -> None:

        _x = self.pretrain_step(x)

        for e_iter in range(epochs):
            np.random.shuffle(_x)

            # Decay our maximum value of THETA slightly - To  keep the full grid from being pulled back and forth by outliers
            self.THETAMAX = self.THETAMAX * 0.98 if self.THETAMAX > self.THETAMIN else self.THETAMIN + 1

            bmu_i, bmu_dist = self.calc_bmu(_x[0])
            self.previous_step_r = bmu_dist[bmu_i]

            error_trace = []
            epsilon_trace = []

            for sample_i in range(1, _x.shape[0]):
                _xs = _x[sample_i:sample_i + 1, :]

                bmu_i, sample_distances = self.calc_bmu(_xs)
                epsilon = self.calc_epsilon(sample_distances[bmu_i])
                theta = self.calc_theta(epsilon)
                neighborhood = self.calc_neighborhood(theta, bmu_i)

                # neighborhood is shape (n_samples, n_neurons, dimensions)
                weight_update = epsilon * (neighborhood.reshape(self.n_neurons, -1) * (_xs - self.weights))

                self.weights += weight_update
                self.hit_map[bmu_i] += 1

                # update error trace
                error_trace.append(np.mean(sample_distances))
                epsilon_trace.append(epsilon)

                # two debugging calls-
                # print('BMU:', self._idx_to_grid(bmu_i),'epsilon: ', epsilon)
                # self.plot_neighborhood(epsilon * (neighborhood.reshape(self.n_neurons, -1)))

            self.q_error_trace.append(np.mean(error_trace))
            self.epsilon_trace.append(np.mean(epsilon_trace))
            # debug step:
            # self.plot_grid(samples=_x, highlight_idx=np.argmin(self.hit_map))

        # self.plot_grid(samples=_x, highlight_idx=np.argmin(self.hit_map))
        # plt.plot(self.q_error_trace)
        # plt.plot(self.epsilon_trace)
        # plt.legend(['q_error', 'epsilon'])
        # plt.show()

        self.n_iter += x.shape[0]

    def calc_bmu(self, x: NDArray) -> tuple[NDArray | int, NDArray]:
        """
        find the best matching unit to the input x

        Parameters
        ----------
        x : input data

        Returns
        -------
        BMU for each sample (shape batch_size, 1), distances to each neuron (shape batch_size, n_neurons)
        """
        dist = self.distance_function(x, self.weights)
        bmu_i = np.argmin(dist, axis=-1)
        return bmu_i, dist

    def calc_epsilon(self, bmu_distance: NDArray) -> NDArray:
        """
        calculate epsilon -- value for the magnitude of the update - it's a value driven by the goodness-of-fit (how
        close / far is this sample from its best matching untit?)

        Parameters
        ----------
        bmu_distance : the distance array between the sample and its best matching unit

        Returns
        -------
        the single-point value epsilon
        """
        self.previous_step_r = np.max((bmu_distance, self.previous_step_r), axis=0)
        return bmu_distance / self.previous_step_r

    def calc_theta(self, epsilon: float | NDArray) -> float | NDArray:
        """
        sets bounds on the reach of the neighborhood function
        uses constants self.THETAMAX and self.THETAMIN - these are usually  set at 0, 1 or 2
        Parameters
        ----------
        epsilon : the calculated epsilon value

        Returns
        -------
        single-value theta for use in neighborhood
        """
        # Using the PLSOM epsilon scale
        theta = max(self.THETAMAX * epsilon, self.THETAMIN)

        return theta

    def get_lateral_distance(self, bmu_i: int, method: str = 'dist') -> NDArray:
        """
        applies the distance function to find distance between the bmu and all other neurons
        Parameters
        ----------
        bmu_i : index of the best matching unit under consideration
        method : 'grid' or 'dist' - use the precomputed grid distances (manhattan),
            or calculate distance between the neurons' embedding dimensions

        Returns
        -------
        distnace array between bmu and all other neurons
        """
        # return distance between this unit and all others - return should be n_neurons, n_dims
        if method == 'dist':
            bmw = self.weights[bmu_i, :]
            return self.distance_function(bmw, self.weights)

        elif method == 'grid':
            return self.grid_distances[bmu_i, :] ** 2

    def calc_neighborhood(self, theta: float | NDArray, bmu_i: int) -> NDArray:
        """
        This is the magic -- PLSOM defines the neighborhood function as an area of influence around a neuron that's
        being updated. The neighborhood is defined as a Gaussian function (exp(-x**2) of the distance between the
        active and the other neurons.

        Parameters
        ----------
        theta : the derived bounds (see calc_theta)
        bmu_i : the index of the best matching unit

        Returns
        -------
        adjusted gaussian kernel applied to distances
        """

        return np.exp((-1 * self.get_lateral_distance(bmu_i, method='grid') / theta ** 2))

    def _idx_to_grid(self, idx: int) -> tuple[int, int]:
        """
        numpy function takes 1d vector index to 2d [r,c] grid.
        """
        r = idx // self.network_shape[0]
        c = idx % self.network_shape[1]
        return (int(r), int(c))

    def _grid_to_idx(self, grid_i: tuple[int, int]) -> int:
        """
        numpy - takes grid index [r,c] converts to 1d index
        """
        return int(grid_i[1] + (grid_i[0] * self.network_shape[0]))

    @staticmethod
    def grid_manhattan_distance(row_a: int, col_a: int, row_b: int, col_b: int) -> float:
        """ returns the cityblock / manhattan dist between two grid points """
        return abs(row_a - row_b) + abs(col_a - col_b)

    def build_grid(self, method: str = 'grid') -> NDArray:
        """
        Build the manhattan distance grid for the SOM, it will contain the distances from each unit, to each other unit
        :return: np.array, (shape n_neurons, n_neurons)
        """
        rows = self.network_shape[0]
        cols = self.network_shape[1]
        distance_matrix = np.zeros((self.n_neurons, self.n_neurons))

        for idx in range(self.n_neurons):
            home_row, home_col = self._idx_to_grid(idx)
            for row in range(rows):
                for col in range(cols):
                    if method == 'grid':
                        distance_matrix[idx, self._grid_to_idx((row, col))] = self.grid_manhattan_distance(home_row,
                                                                                                           home_col,
                                                                                                           row, col)
                    elif method == 'distance':
                        distance_matrix[idx, self._grid_to_idx((row, col))] = self.distance_function(
                            self.weights[self._idx_to_grid((row, col))], )

        return distance_matrix

    def standardize_inputs(self, x: NDArray) -> NDArray:
        """
        apply standardization to our inputs - reducing to zero mean and 1 stdev
        Parameters
        ----------
        x : array of datapoints

        Returns
        -------
        the standardized data
        """
        this_means = np.mean(x, axis=0)
        this_stds = np.std(x, axis=0)

        num_new = x.shape[0]
        if self.n_iter == 0:
            self.means = this_means
            self.stds = this_stds
        else:
            self.means, self.stds = plsom_utils.update_mean_std(self.means,
                                                                self.stds,
                                                                self.n_iter,
                                                                this_means,
                                                                this_stds,
                                                                num_new)

        return (x - self.means) / (self.stds + EPSILON)

    def plot_grid(self, samples=0, highlight_idx=None):
        """ utility function to plot the first two dimensions of the grid of weights"""
        ws = self.weights.reshape(self.network_shape[0], self.network_shape[1], self.N_DIMS)
        # Draw lines between each 2d weight vector in grid
        plt.figure(figsize=(15, 10))
        if samples != 0:
            plt.scatter(samples[:, 0], samples[:, 1])
        for row in range(self.network_shape[0]):
            plt.plot(ws[row, :, 0], ws[row, :, 1], 'bo-')
        for col in range(self.network_shape[1]):
            plt.plot(ws[:, col, 0], ws[:, col, 1], 'bo-')
        if highlight_idx:
            focus = self._idx_to_grid(highlight_idx)
            plt.scatter(ws[focus[0], focus[1], 0], ws[focus[0], focus[1], 1], s=200.0, c='r')
        plt.show()

    def plot_neighborhood(self, neighborhood):
        """ utility function to plot the neighborhood, "gravity" or sphere of influence around the bmu"""
        # shape the weights into the 2D grid representation
        ns = neighborhood.reshape(self.network_shape[0], self.network_shape[1])
        # Draw lines between each 2d weight vector in grid
        plt.figure(figsize=(5, 5))
        for row in range(self.network_shape[0]):
            for col in range(self.network_shape[1]):
                plt.scatter(row, col, s=ns[row, col] * 1000)
        plt.show()

    def plot_heatmap(self):
        """ utilty function to draw the hit map as aa heat map"""
        plt.figure(figsize=(10, 10))
        self.hit_map.reshape(self.network_shape)
        plt.imshow(self.hit_map.reshape(self.network_shape), cmap="hot", interpolation="nearest")
        plt.show()

    def u_matrix(self):
        """
        Build a u_matrix
        :return:
        """
        # for each neuron, we need to get the self.distance() to each adjacent unit
        # 3x3 will become
        # o - o - o - o
        # |   |   |   |
        # o - o - o - o
        # |   |   |   |
        # o - o - o - o

        # n*2-1 x n*2-1
        # I'm sure there's a more elegant way to do this...
        rows = self.network_shape[0] * 2 - 1
        cols = self.network_shape[1] * 2 - 1
        u_matrix = np.zeros((rows, cols))
        # now iter
        for row_i in range(rows):
            for col_i in range(cols):
                if row_i % 2 == 0:
                    if col_i % 2 == 0:
                        idx = self._grid_to_idx((row_i / 2, col_i / 2))
                        # u_matrix[row_i, col_i] = copy.copy(self.weights[idx])
                    else:
                        # 'horizontal linkage'
                        left = self._grid_to_idx((row_i // 2, col_i // 2))
                        right = self._grid_to_idx((row_i // 2, (col_i // 2) + 1))
                        u_matrix[row_i, col_i] = self.distance_function(self.weights[left], self.weights[right])
                else:
                    if col_i % 2 == 0:
                        # vertical linkage
                        upper = self._grid_to_idx((row_i // 2, col_i // 2))
                        lower = self._grid_to_idx(((row_i // 2) + 1, col_i // 2))
                        u_matrix[row_i, col_i] = self.distance_function(self.weights[upper], self.weights[lower])
                    else:
                        pass

        u_matrix = plsom_utils.umat_fill_outer(u_matrix, rows, cols)
        u_matrix = plsom_utils.umat_infill(u_matrix, rows, cols)

        blurred = plsom_utils.blur(u_matrix)
        # adding original image to the blur - essentially 50/50 blend so we don't 'overweight' the blurred version.
        u_matrix += blurred

        # inverting -
        # u_matrix = u_matrix.max() - u_matrix

        u_matrix *= 1.0 / u_matrix.max()
        return u_matrix

    def neural_distances(self, w_prime: NDArray = None):
        """
        calc the distances between neurons in the map with the determined distance func on the embedding dimensions

        Parameters
        ----------
        w_prime :

        Returns
        -------

        """
        if type(w_prime) == np.ndarray:
            neural_dist = self.distance_function(w_prime[:, np.newaxis, :], w_prime)
            for n_i in range(neural_dist.shape[0]):
                neural_dist[n_i, n_i] = np.inf
            return neural_dist
        else:
            neural_dist = self.distance_function(self.weights[:, np.newaxis, :], self.weights)
            for n_i in range(neural_dist.shape[0]):
                neural_dist[n_i, n_i] = np.inf

            return neural_dist

    def predict_clusters(self, x: NDArray, n_clusters: int, verbose: bool = False):
        """
        Using a dedicated clustering method to predict classes of samples based on the SOM's re-projected
        representation
        We use the clustering method on the SOM's weights, rather than on the dataset directly
        """
        # determine which protoype (SOM neuron) is closest to each sample
        _x = self.standardize_inputs(x)
        idxs, dist = self.calc_bmu(_x[:, np.newaxis, :])
        # grid_idxs = np.array(
        #     [self._idx_to_grid(_i) for _i in idxs]
        # )

        if verbose:
            print(f"Weights: {self.weights.shape}")
            self.plot_heatmap()

        self.clust_model = CentroidNeuralNetwork(
            max_clusters=n_clusters,
            seed=42,
            initial_clusters=None,
            epsilon=1e-4
        )

        _, _ = self.clust_model.fit_predict(
            org_x_data=self.weights,
            num_iterations=25,
            fast_forward=False,
            verbose=False
        )

        best_scoring, centroids, labels = self.clust_model.get_optimal()
        print(f"Optimal Clusters at {best_scoring}")

        if verbose:
            plot_clusters(self.weights, centroids, labels)

        # assign each sample to its closest neuron prototype (idxs), then assign that neuron to its cluster (labels)
        return np.take_along_axis(labels, idxs, axis=0)


# ------------------------predict function------------------------
def plsom_fit_predict(processed_features: NDArray, grid_dim: int, max_clusters: int):
    """

    Parameters
    ----------
    processed_features :
    max_clusters :  maximum count of clusters to estimate

    Returns
    -------

    """
    start = time.time()
    n_dims = processed_features.shape[-1]

    size = min(n_dims, grid_dim)
    n_epochs = 25  # min(size // 4, 2)

    som_nnet = PLSOM(width=size,
                     height=size,
                     input_dim=n_dims,
                     theta_min=0 + 0.01,
                     theta_max=size - 0.01,
                     lock_seed=1,
                     distance='euclidean')

    som_nnet.fit(processed_features, epochs=n_epochs)
    som_nnet.plot_grid(samples=0, highlight_idx=None)

    prediction = som_nnet.predict_clusters(x=processed_features,
                                           n_clusters=max_clusters,
                                           verbose=True)

    print('plsomv_fit_predict took', time.time() - start)

    return prediction


if __name__ == "__main__":
    # Example usage
    from ml_tools.generators import RandomDatasetGenerator
    from ml_tools.models.clustering.cluster_metrics import homogeneity

    gen = RandomDatasetGenerator(random_seed=123)
    x_clust, y_clust, meta_clust = gen.generate(
        task='clustering',
        num_samples=1500,
        num_features=128,
        num_clusters=9,
        noise_scale=0.33
    )

    max_clusters = 15
    for dim in [5, 10, 15, 20, 35]:
        st = time.time()
        predictions = plsom_fit_predict(x_clust,
                                        grid_dim=dim,
                                        max_clusters=max_clusters)
        print("one_cluster predict took: ", time.time() - st)


        print("Predictions:", predictions[:20])
        print("Truth:", y_clust[:20])
        print(homogeneity(predictions, y_clust))
