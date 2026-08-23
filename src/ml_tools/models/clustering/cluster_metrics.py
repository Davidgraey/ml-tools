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


def entropy(labels):
    """Compute entropy of a label distribution."""
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log(probabilities))


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
    class_entropy = entropy(class_freqs)

    cond_ent = 0.0

    for i in range(num_clusters):
        cluster = cont_matrix[:, i]
        cluster_size = np.sum(cluster)
        if cluster_size > 0:
            cond_ent = entropy(cluster)

    cond_ent /= num_samples

    # homogeneity score
    if class_entropy == 0:
        return 1.0
    return 1 - cond_ent / class_entropy
