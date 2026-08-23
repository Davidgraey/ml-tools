"""
Clustering: the parameterless SOM, its growing variant, the centroid network,
and the cluster quality metrics.

The GPLSOM tests care most about invariants that must survive a structural
change -- after adding or dropping a row, the weights, the per-node records and
the lattice distance matrix all have to stay consistent with each other.
"""

import numpy as np
import pytest

from ml_tools.models.clustering.centroid_network import CentroidNeuralNetwork
from ml_tools.models.clustering.cluster_metrics import (
    calinski_harabasz_index,
    contingency_matrix,
    davies_bouldin_index,
    entropy,
    homogeneity,
    mutual_information_score,
    silhouette_score,
)
from ml_tools.models.clustering.gplsom_clustering import GPLSOM
from ml_tools.models.clustering.plsom_clustering import PLSOM


LATTICES = ((4, 4), (3, 5), (5, 3), (2, 9))


def lattice_is_consistent(som) -> bool:
    """every per-node structure agrees with the current lattice shape"""
    rows, cols = som.network_shape
    grid = som.grid_distances
    return (
        som.n_neurons == rows * cols
        and som.weights.shape == (som.n_neurons, som.N_DIMS)
        and som.hit_map.size == som.n_neurons
        and som.node_error.size == som.n_neurons
        and grid.shape == (som.n_neurons, som.n_neurons)
        and np.allclose(grid, grid.T)
        and np.allclose(np.diag(grid), 0.0)
        and grid.max() == (rows - 1) + (cols - 1)
        and np.isfinite(som.weights).all()
    )


# -------------    lattice geometry    -----------------------------
@pytest.mark.parametrize("height,width", LATTICES)
def test_index_helpers_round_trip(height, width):
    """
    Row major, so the divisor is the column count. Using the row count agrees
    only on square lattices, which is why non-square maps were broken.
    """
    som = PLSOM(width=width, height=height, input_dim=2)
    for index in range(som.n_neurons):
        assert som._grid_to_idx(som._idx_to_grid(index)) == index


@pytest.mark.parametrize("height,width", LATTICES)
def test_grid_distances_are_a_metric(height, width):
    som = PLSOM(width=width, height=height, input_dim=2)
    grid = som.grid_distances
    assert np.allclose(grid, grid.T)
    assert np.allclose(np.diag(grid), 0.0)
    assert grid.max() == (height - 1) + (width - 1)


def test_grid_distance_is_manhattan():
    som = PLSOM(width=4, height=3, input_dim=2)
    first = som._grid_to_idx((0, 0))
    second = som._grid_to_idx((2, 3))
    assert som.grid_distances[first, second] == pytest.approx(5.0)


# -------------    PLSOM training    -------------------------------
@pytest.mark.parametrize("height,width", LATTICES)
def test_plsom_fits_any_lattice(height, width, clustering_dataset):
    x_data, _, _ = clustering_dataset
    som = PLSOM(
        width=width, height=height, input_dim=2, theta_min=0.01,
        theta_max=max(width - 0.01, 1),
    )
    som.fit(x_data, num_iterations=3)
    assert lattice_is_consistent(som)
    assert len(som.q_error_trace) == 3


def test_plsom_hit_map_decays(clustering_dataset):
    """
    Without decay a node that was dead early still reads as dead once busy.
    The recurrence is hits <- decay * hits + new, so the total must sit below
    the undecayed count.
    """
    x_data, _, _ = clustering_dataset
    som = PLSOM(width=4, height=4, input_dim=2, hit_decay=0.5)
    som.fit(x_data, num_iterations=4)

    per_epoch = len(x_data) - 1
    undecayed = 4 * per_epoch
    assert 0 < som.hit_map.sum() < undecayed


def test_plsom_decay_matches_the_recurrence(clustering_dataset):
    x_data, _, _ = clustering_dataset
    decay = 0.5
    epochs = 4
    som = PLSOM(width=4, height=4, input_dim=2, hit_decay=decay)
    som.fit(x_data, num_iterations=epochs)

    per_epoch = len(x_data) - 1
    expected = 0.0
    for _ in range(epochs):
        expected = decay * expected + per_epoch
    assert som.hit_map.sum() == pytest.approx(expected)


def test_plsom_node_error_also_decays(clustering_dataset):
    x_data, _, _ = clustering_dataset
    som = PLSOM(width=4, height=4, input_dim=2, hit_decay=0.5)
    som.fit(x_data, num_iterations=3)
    assert som.node_error.sum() > 0
    assert np.isfinite(som.node_error).all()


def test_plsom_quantisation_error_uses_the_winner(clustering_dataset):
    """
    Distance to the best matching unit, not the mean over all units -- the
    latter grows as the map spreads and cannot be compared across map sizes.
    """
    x_data, _, _ = clustering_dataset
    small = PLSOM(width=3, height=3, input_dim=2, theta_min=0.01, theta_max=2.99)
    large = PLSOM(width=8, height=8, input_dim=2, theta_min=0.01, theta_max=7.99)
    small.fit(x_data, num_iterations=25)
    large.fit(x_data, num_iterations=25)
    assert large.q_error_trace[-1] < small.q_error_trace[-1]


def test_plsom_after_epoch_is_a_no_op_hook(clustering_dataset):
    x_data, _, _ = clustering_dataset
    som = PLSOM(width=3, height=3, input_dim=2)
    shape_before = list(som.network_shape)
    som.fit(x_data, num_iterations=2)
    assert som.network_shape == shape_before


# -------------    GPLSOM growth and pruning    --------------------
def test_gplsom_constructs():
    """the super() call and the truncated init both used to break this"""
    som = GPLSOM(width=4, height=4, input_dim=2)
    assert som.weights is None, "weights are allocated on the first fit"
    assert som.growth_threshold > 0
    for attribute in (
        "hit_map", "node_error", "THETAMIN", "THETAMAX", "distance_function",
        "q_error_trace", "epsilon_trace", "previous_step_r",
    ):
        assert hasattr(som, attribute), f"missing {attribute}"


def test_gplsom_rejects_a_degenerate_spread_factor():
    for spread in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(AssertionError):
            GPLSOM(width=4, height=4, input_dim=2, spread_factor=spread)


@pytest.mark.parametrize("axis,position,edge", (
    (0, 2, False), (1, 1, False), (0, 0, True), (1, 0, True),
))
def test_growth_keeps_the_lattice_consistent(axis, position, edge, clustering_dataset):
    x_data, _, _ = clustering_dataset
    som = GPLSOM(width=5, height=4, input_dim=2)
    som.fit(x_data, num_iterations=1)

    before = list(som.network_shape)
    som.grow_line(axis, position, edge)

    assert som.network_shape[axis] == before[axis] + 1
    assert som.network_shape[1 - axis] == before[1 - axis]
    assert lattice_is_consistent(som)


@pytest.mark.parametrize("axis", (0, 1))
def test_pruning_keeps_the_lattice_consistent(axis, clustering_dataset):
    x_data, _, _ = clustering_dataset
    som = GPLSOM(width=5, height=4, input_dim=2)
    som.fit(x_data, num_iterations=1)

    before = list(som.network_shape)
    som.prune_line(axis, 1)

    assert som.network_shape[axis] == before[axis] - 1
    assert lattice_is_consistent(som)


def test_interior_growth_interpolates():
    som = GPLSOM(width=3, height=3, input_dim=1)
    som.weights = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0],
                            [20.0], [21.0], [22.0]])
    som.hit_map = np.zeros(9)
    som.node_error = np.zeros(9)

    som.grow_line(axis=0, position=2, edge=False)
    rows = som.weight_grid()[:, :, 0]
    assert np.allclose(rows[2], [15.0, 16.0, 17.0])


def test_edge_growth_extrapolates():
    som = GPLSOM(width=3, height=3, input_dim=1)
    som.weights = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0],
                            [20.0], [21.0], [22.0]])
    som.hit_map = np.zeros(9)
    som.node_error = np.zeros(9)

    som.grow_line(axis=0, position=3, edge=True)
    rows = som.weight_grid()[:, :, 0]
    assert np.allclose(rows[-1], [30.0, 31.0, 32.0])


def test_a_new_line_inherits_neighbouring_activity():
    """
    Seeding hits at zero makes the new line instantly the coldest, so pruning
    removes it again and the map oscillates.
    """
    som = GPLSOM(width=3, height=3, input_dim=1)
    som.weights = np.arange(9, dtype=float).reshape(9, 1)
    som.hit_map = np.full(9, 10.0)
    som.node_error = np.zeros(9)

    som.grow_line(axis=0, position=2, edge=False)
    inserted = som.hit_map.reshape(som.network_shape)[2]
    assert np.allclose(inserted, 10.0)


def test_growth_respects_the_neuron_budget(clustering_dataset):
    """a line adds a whole row, so the check must use the post-growth count"""
    x_data, _, _ = clustering_dataset
    for cap in (16, 24, 40):
        som = GPLSOM(
            width=3, height=3, input_dim=2, theta_min=0.01, theta_max=2.99,
            spread_factor=0.95, max_neurons=cap,
        )
        som.fit(x_data, num_iterations=30)
        assert som.n_neurons <= cap, f"cap {cap} exceeded at {som.n_neurons}"


def test_pruning_respects_the_floor(clustering_dataset):
    x_data, _, _ = clustering_dataset
    som = GPLSOM(
        width=6, height=6, input_dim=2, theta_min=0.01, theta_max=5.99,
        spread_factor=0.02, prune_ratio=5.0, min_shape=(2, 3),
    )
    som.fit(x_data, num_iterations=40)
    assert som.network_shape[0] >= 2
    assert som.network_shape[1] >= 3


def test_growth_does_not_oscillate(clustering_dataset):
    """
    A grow immediately undone by a prune, repeatedly, means the two signals are
    fighting. Look for the lattice revisiting a shape two changes later.
    """
    x_data, _, _ = clustering_dataset
    som = GPLSOM(
        width=3, height=3, input_dim=2, theta_min=0.01, theta_max=2.99,
        spread_factor=0.4,
    )
    som.fit(x_data, num_iterations=40)

    shapes = [(rows, cols) for _, _, rows, cols in som.structure_trace]
    repeats = sum(1 for i in range(2, len(shapes)) if shapes[i] == shapes[i - 2])
    assert repeats <= max(1, len(shapes) // 4), f"oscillating: {shapes}"


def test_settle_epochs_spaces_structural_changes(clustering_dataset):
    x_data, _, _ = clustering_dataset
    settle = 5
    som = GPLSOM(
        width=3, height=3, input_dim=2, theta_min=0.01, theta_max=2.99,
        spread_factor=0.95, settle_epochs=settle,
    )
    som.fit(x_data, num_iterations=30)

    epochs = [step for step, *_ in som.structure_trace]
    gaps = np.diff(epochs)
    assert (gaps >= settle).all(), f"changes too close together: {epochs}"


def test_gplsom_survives_a_full_run(clustering_dataset):
    x_data, _, _ = clustering_dataset
    som = GPLSOM(
        width=3, height=4, input_dim=2, theta_min=0.01, theta_max=2.99,
        spread_factor=0.9,
    )
    som.fit(x_data, num_iterations=30)
    assert lattice_is_consistent(som)


def test_mean_node_error_is_scale_free(clustering_dataset):
    """
    Dividing by the hit count puts the error in the units the growth threshold
    is expressed in, so it does not ride on dataset size.
    """
    x_data, _, _ = clustering_dataset
    small = GPLSOM(width=4, height=4, input_dim=2)
    large = GPLSOM(width=4, height=4, input_dim=2)
    small.fit(x_data[:200], num_iterations=3)
    large.fit(x_data, num_iterations=3)

    ratio = large.mean_node_error().max() / small.mean_node_error().max()
    assert 0.2 < ratio < 5.0, f"threshold signal scales with dataset size: {ratio}"


# -------------    cluster metrics    ------------------------------
def test_metrics_prefer_the_true_labelling(clustering_dataset):
    x_data, y_data, _ = clustering_dataset
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(y_data)

    assert silhouette_score(x_data, y_data) > silhouette_score(x_data, shuffled)
    assert calinski_harabasz_index(x_data, y_data) > calinski_harabasz_index(
        x_data, shuffled
    )
    assert davies_bouldin_index(x_data, y_data) < davies_bouldin_index(
        x_data, shuffled
    )


def test_homogeneity_is_perfect_for_an_exact_match(clustering_dataset):
    _, y_data, _ = clustering_dataset
    assert homogeneity(y_data, y_data) == pytest.approx(1.0)


def test_homogeneity_is_invariant_to_relabelling(clustering_dataset):
    _, y_data, _ = clustering_dataset
    relabelled = (y_data + 1) % (y_data.max() + 1)
    assert homogeneity(y_data, relabelled) == pytest.approx(
        homogeneity(y_data, y_data)
    )


def test_mutual_information_is_zero_for_a_constant_prediction(clustering_dataset):
    _, y_data, _ = clustering_dataset
    constant = np.zeros_like(y_data)
    assert mutual_information_score(y_data, constant) == pytest.approx(0.0, abs=1e-9)


def test_entropy_is_zero_for_one_class():
    assert entropy(np.zeros(50, dtype=int)) == pytest.approx(0.0)


def test_entropy_is_maximal_for_a_uniform_split():
    balanced = np.repeat(np.arange(4), 25)
    assert entropy(balanced) == pytest.approx(np.log(4), rel=1e-6)


def test_contingency_matrix_counts_every_sample(clustering_dataset):
    _, y_data, _ = clustering_dataset
    rng = np.random.default_rng(0)
    predicted = rng.integers(0, 3, size=y_data.shape)
    table = contingency_matrix(y_data, predicted)
    assert table.sum() == len(y_data)


# -------------    CentroidNeuralNetwork    ------------------------
def test_centroid_network_finds_the_planted_clusters(clustering_dataset):
    pytest.importorskip("scipy.spatial")
    x_data, y_data, _ = clustering_dataset

    model = CentroidNeuralNetwork(max_clusters=8, seed=42, epsilon=1e-4)
    model.fit_predict(
        org_x_data=x_data, num_iterations=40, fast_forward=False, verbose=False
    )
    best, centroids, labels = model.get_optimal()

    assert labels.shape[0] == x_data.shape[0]
    assert 2 <= best <= 8
    assert homogeneity(y_data, labels) > 0.5
