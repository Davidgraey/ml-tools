import numpy as np
from numpy.typing import NDArray


def silhouette_score(x_data: NDArray, prediction: NDArray) -> float:
    """
    Compute the mean silhouette score for a clustering prediction
    Vectorized over samples using a full distance matrix

    Parameters
    ----------
    x_data : our original data samples
    prediction : predictied labels as integers

    Returns
    -------
    mean silhouette score across the dataset -1 to 1
    """
    # Input validation
    if int(x_data.shape[0]) != int(prediction.shape[0]):
        raise ValueError("we need a label for every sample in x_data")

    # calc all the distances between points-- this may be slow for huge datasets.
    distance_matrix = np.linalg.norm(x_data[:, None, :] - x_data[None, :, :], axis=2)

    # get counts & sizes collected
    unique_labels, inv = np.unique(prediction, return_inverse=True)
    num_samples = int(x_data.shape[0])
    num_clusters = int(unique_labels.size)
    if num_clusters < 2 or num_samples == 0:
        return 0.0

    # get the size of each cluster based on the mask
    cluster_sizes = np.bincount(inv, minlength=num_clusters)
    summed_distances = np.empty((num_samples, num_clusters), dtype=distance_matrix.dtype)
    for k in range(num_clusters):
        clust_mask = inv == k
        summed_distances[:, k] = distance_matrix[:, clust_mask].sum(axis=1)


    # intra-cluster mean distance for each sample (exclude self)
    own_sizes = cluster_sizes[inv]
    own_sums = summed_distances[np.arange(num_samples), inv]
    is_singleton = own_sizes <= 1
    with np.errstate(divide="ignore", invalid="ignore"):
        # I hate that npwhere doesn't like kwargs.
        intra_dist = np.where(
            own_sizes > 1,
            own_sums / (own_sizes - 1),
            0.0
        )

    # average distance from each sample to each cluster
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_distances = summed_distances / np.maximum(cluster_sizes, 1)

    # excluding self-references to the same cluster -this works better than setting infinity.
    eye = np.eye(num_clusters, dtype=bool)
    row_mask = eye[inv]
    masked_means = np.where(row_mask, np.inf, mean_distances)
    nearest_clust_dist = masked_means.min(axis=1)

    # score per sample
    largest_delta = np.maximum(intra_dist, nearest_clust_dist)
    with np.errstate(divide="ignore", invalid="ignore"):
        # I hate that where doesn't like kwargs.
        scores = np.where(
            largest_delta > 0,
            (nearest_clust_dist - intra_dist) / largest_delta,
            0.0
        )

    # a sample alone in its own cluster has no intra-cluster distance to
    # measure cohesion against -- convention (matching sklearn) is silhouette
    # 0 for it, not the 1.0 that "perfect cohesion" (intra_dist=0) would imply
    scores = np.where(is_singleton, 0.0, scores)

    return np.mean(scores)


def davies_bouldin_index(x_data: NDArray, prediction: NDArray) -> float:
    """
    DB index quantifies the within-cluster spread to inter-cluster distances
    Lower DB scores indicate a more compact, more seperated cluster
    Parameters
    ----------
    x_data : our original data samples
    prediction : predictied labels as integers

    Returns
    -------
    DB score - lower is "better"
    """
    unique_labels = np.unique(prediction)
    num_clusters = len(unique_labels)

    # calculate centroids
    centroids = np.array(
        [
            x_data[prediction == k].mean(axis=0)
            for k in unique_labels
        ]
    )

    # calculate s_i for each cluster
    s = np.zeros(num_clusters)
    for i, k in enumerate(unique_labels):
        cluster_points = x_data[prediction == k]
        s[i] = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))

    # calculate Davies-Bouldin index - ratio of norm'd dist between centroids
    db_index = 0
    for i in range(num_clusters):
        max_ratio = 0
        for j in range(num_clusters):
            if i != j:
                dist = np.linalg.norm(centroids[i] - centroids[j])
                ratio = (s[i] + s[j]) / dist

                if ratio > max_ratio:
                    max_ratio = ratio
        db_index += max_ratio

    return db_index / num_clusters


def calinski_harabasz_index(x_data: NDArray, prediction: NDArray) -> float:
    """
    Ratio of inter-cluster dispersion over the intra-cluster dispersion
    Higher CH index scores show us more dense, separated clusters
    Parameters
    ----------
    x_data : our original data samples
    prediction : predictied labels as integers

    Returns
    -------
    CH index - variance ratio criterion
    """
    unique_labels = np.unique(prediction)
    n_samples, n_features = x_data.shape
    n_clusters = len(unique_labels)
    # undefined (both terms divide by n_clusters - 1) when everything landed
    # in one cluster -- match silhouette_score's convention for a degenerate
    # partition rather than raising a ZeroDivisionError
    if n_clusters < 2 or n_samples == 0:
        return 0.0
    overall_mean = np.mean(x_data, axis=0)


    cluster_means = np.array(
        [np.mean(x_data[prediction == k], axis=0) for k in unique_labels]
    )

    cluster_sizes = np.array(
        [np.sum(prediction == k) for k in unique_labels]
    )

    # inter-cluster dispersion
    inter_dispersion = np.sum(cluster_sizes[:, None] * (cluster_means - overall_mean) ** 2)

    # intra-cluster dispersion
    intra_dispersion = 0
    for i, label in enumerate(unique_labels):
        cluster_data = x_data[prediction == label]
        intra_dispersion += np.sum((cluster_data - cluster_means[i]) ** 2)

    return (inter_dispersion / intra_dispersion) * ((n_samples - n_clusters) / (n_clusters - 1))


def cubic_clustering_criterion(x_data: NDArray, prediction: NDArray) -> float:
    """
    Cubic Clustering Criterion (CCC) for estimating the number of clusters
    Compares the observed R^2 of a clustering against the R^2 expected from
    points spread uniformly over the data's own principal-axis extent, with
    the same number of clusters. Values above ~2-3 suggest real structure;
    values near or below 0 suggest the clustering has no more structure than
    a random split of uniform data.

    Parameters
    ----------
    x_data : our original data samples
    prediction : predictied labels as integers

    Returns
    -------
    CCC statistic - higher is "better", roughly 0 for no structure

    References
    ----------
    Sarle, W.S. (1983), "Cubic Clustering Criterion", SAS Technical
    Report A-108, SAS Institute Inc.
    """
    unique_labels = np.unique(prediction)
    n_samples, n_features = x_data.shape
    n_clusters = len(unique_labels)
    # same degenerate cases as calinski_harabasz_index, plus we need at
    # least 2 samples to form a covariance matrix
    if n_clusters < 2 or n_samples < 2:
        return 0.0

    x_centered = x_data - x_data.mean(axis=0)
    total_scatter = x_centered.T @ x_centered
    trace_total = np.trace(total_scatter)
    if trace_total <= 0:
        return 0.0

    within_scatter = 0.0
    for label in unique_labels:
        cluster_data = x_data[prediction == label]
        cluster_centered = cluster_data - cluster_data.mean(axis=0)
        within_scatter += np.sum(cluster_centered ** 2)

    r_squared = 1.0 - within_scatter / trace_total

    # eigenvalues of the covariance matrix describe the data's own
    # principal-axis spread -- this is the "hyperbox" the null hypothesis
    # (uniformly distributed points) is compared against
    eigenvalues = np.linalg.eigvalsh(total_scatter / (n_samples - 1))
    eigenvalues = np.sort(eigenvalues)[::-1]
    sqrt_eigs = np.sqrt(np.maximum(eigenvalues, 0.0))
    s = np.where(sqrt_eigs > 0, sqrt_eigs, 1.0)
    vv = np.prod(s)

    # scale s down to a per-cluster cell size, then decide how many of the
    # largest (>= cell size) axes to treat exactly (hypercube) vs.
    # approximate (hypersphere tail) -- Sarle's mixed approximation
    cell_scale = (vv / n_clusters) ** (1.0 / n_features)
    u = s / cell_scale
    k1 = int(np.sum(u >= 1.0))
    p1 = min(k1, n_clusters - 1)

    if 0 < p1 < n_features:
        v1 = np.prod(s[:p1])
        cell_scale = (v1 / n_clusters) ** (1.0 / p1)
        u = s / cell_scale
        b1 = np.sum(1.0 / (n_samples + u[:p1]))
        tail = u[p1:]
        b2 = np.sum(tail ** 2 / (n_samples + tail))
        b_total = b1 + b2
        effective_dims = p1
    else:
        b_total = np.sum(1.0 / (n_samples + u))
        effective_dims = n_features

    sum_u2 = np.sum(u ** 2)
    if sum_u2 <= 0:
        return 0.0
    expected_r_squared = 1.0 - (b_total / sum_u2) * (
        (n_samples - n_clusters) ** 2 / n_samples
    ) * (1.0 + 4.0 / n_samples)

    ratio_num = 1.0 - expected_r_squared
    ratio_den = 1.0 - r_squared
    if ratio_num <= 0 or ratio_den <= 0:
        return 0.0

    ccc = np.log(ratio_num / ratio_den) * (
        np.sqrt(n_samples * effective_dims / 2.0)
        / (0.001 + expected_r_squared) ** 1.2
    )
    return float(ccc)


# ---- performance metrics with labels available ----
def contingency_matrix(labels_true, labels_pred):
    """Create a contingency matrix for two labelings."""
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    cont_matrix = np.zeros((n_classes, n_clusters), dtype=np.int64)
    np.add.at(cont_matrix, (class_idx, cluster_idx), 1)
    return cont_matrix


def mutual_information_score(labels_true, labels_pred):
    """Compute the mutual information score between two clusterings."""
    contingency = contingency_matrix(labels_true, labels_pred)
    n_samples = np.sum(contingency)
    pi = contingency / n_samples
    pi_i = np.sum(pi, axis=1)
    pi_j = np.sum(pi, axis=0)

    non_zero = pi > 0
    mi = np.sum(pi[non_zero] * np.log(pi[non_zero] / np.outer(pi_i, pi_j)[non_zero]))
    return mi


def entropy_from_counts(counts: NDArray) -> float:
    """
    Shannon entropy (nats) of a frequency/count vector. Zero-count entries
    are dropped first (0 * log(0) is conventionally 0, not nan), which a
    contingency-matrix row or column routinely has.
    """
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log(probabilities))


def entropy(labels):
    """Compute entropy of a label distribution."""
    _, counts = np.unique(labels, return_counts=True)
    return entropy_from_counts(counts)


def homogeneity(labels_true, labels_pred):
    """Compute homogeneity score of predicted labels given true labels."""

    # get unique class and cluster indices
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)

    # contingency matrix creation
    num_classes = classes.shape[0]
    num_clusters = clusters.shape[0]
    cont_matrix = np.zeros((num_classes, num_clusters), dtype=np.int64)
    np.add.at(cont_matrix, (class_idx, cluster_idx), 1)

    num_samples = np.sum(cont_matrix)
    class_freqs = np.sum(cont_matrix, axis=1)
    class_entropy = entropy_from_counts(class_freqs)

    # conditional entropy H(C|K) = sum_k (n_k / n) * H(C | K=k) -- each
    # cluster's own class-mixture entropy, weighted by its share of the data
    # and accumulated across every cluster, not just the last one
    cond_ent = 0.0

    for i in range(num_clusters):
        cluster = cont_matrix[:, i]
        cluster_size = np.sum(cluster)
        if cluster_size > 0:
            cond_ent += (cluster_size / num_samples) * entropy_from_counts(cluster)

    # homogeneity score
    if class_entropy == 0:
        return 1.0
    return 1 - cond_ent / class_entropy
