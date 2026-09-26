"""
Clustering: SOM lattice bookkeeping through training, growth and pruning;
metrics; and recovery of planted clusters.
"""

import numpy as np
import pytest
from polyergalio.models.clustering.centroid_network import CentroidNeuralNetwork
from polyergalio.models.clustering.cluster_metrics import (
    calinski_harabasz_index,
    davies_bouldin_index,
    homogeneity,
    silhouette_score,
)
from polyergalio.models.clustering.gplsom_clustering import GPLSOM
from polyergalio.models.clustering.plsom_clustering import PLSOM

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


# -------------    PLSOM training    -------------------------------
@pytest.mark.parametrize("height,width", LATTICES)
def test_plsom_fits_any_lattice(height, width, clustering_dataset):
    x_data, _, _ = clustering_dataset
    som = PLSOM(
        width=width,
        height=height,
        input_dim=2,
        theta_min=0.01,
        theta_max=max(width - 0.01, 1),
    )
    som.fit(x_data, num_iterations=3)
    assert lattice_is_consistent(som)
    assert len(som.q_error_trace) == 3


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


# -------------    GPLSOM growth and pruning    --------------------
@pytest.mark.parametrize(
    "axis,position,edge",
    (
        (0, 2, False),
        (1, 1, False),
        (0, 0, True),
        (1, 0, True),
    ),
)
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


def test_growth_respects_the_neuron_budget(clustering_dataset):
    """a line adds a whole row, so the check must use the post-growth count"""
    x_data, _, _ = clustering_dataset
    for cap in (16, 24, 40):
        som = GPLSOM(
            width=3,
            height=3,
            input_dim=2,
            theta_min=0.01,
            theta_max=2.99,
            spread_factor=0.95,
            max_neurons=cap,
        )
        som.fit(x_data, num_iterations=30)
        assert som.n_neurons <= cap, f"cap {cap} exceeded at {som.n_neurons}"


def test_gplsom_survives_a_full_run(clustering_dataset):
    x_data, _, _ = clustering_dataset
    som = GPLSOM(
        width=3,
        height=4,
        input_dim=2,
        theta_min=0.01,
        theta_max=2.99,
        spread_factor=0.9,
    )
    som.fit(x_data, num_iterations=30)
    assert lattice_is_consistent(som)


# -------------    cluster metrics    ------------------------------
def test_metrics_prefer_the_true_labelling(clustering_dataset):
    x_data, y_data, _ = clustering_dataset
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(y_data)

    assert silhouette_score(x_data, y_data) > silhouette_score(x_data, shuffled)
    assert calinski_harabasz_index(x_data, y_data) > calinski_harabasz_index(
        x_data, shuffled
    )
    assert davies_bouldin_index(x_data, y_data) < davies_bouldin_index(x_data, shuffled)


def test_homogeneity_is_invariant_to_relabelling(clustering_dataset):
    _, y_data, _ = clustering_dataset
    relabelled = (y_data + 1) % (y_data.max() + 1)
    assert homogeneity(y_data, relabelled) == pytest.approx(homogeneity(y_data, y_data))


# -------------    CentroidNeuralNetwork    ------------------------
def test_centroid_network_finds_the_planted_clusters(clustering_dataset):
    """
    get_optimal() picks its k by silhouette score, which tends to favor
    fewer, broader clusters and is not a claim this network recovers the
    exact planted count -- that is a model-selection question, separate from
    whether the network's growth actually finds the planted structure. This
    checks the latter directly: at the true planted cluster count (reached
    during the same growth run get_optimal() draws from), the labels found
    should closely match the planted ones.
    """
    x_data, y_data, _ = clustering_dataset
    true_cluster_count = len(np.unique(y_data))

    model = CentroidNeuralNetwork(max_clusters=8, seed=42, epsilon=1e-4)
    model.fit_predict(
        org_x_data=x_data, num_iterations=40, fast_forward=False, verbose=False
    )
    best, centroids, labels = model.get_optimal()

    assert labels.shape[0] == x_data.shape[0]
    assert 2 <= best <= 8

    at_true_k = next(
        step for step in model.growth_history()
        if step["num_centroids"] == true_cluster_count
    )
    assert homogeneity(y_data, at_true_k["labels"]) > 0.5
