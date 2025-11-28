import numpy as np
from numpy.typing import NDArray
# from sklearn.metrics import silhouette_score as sk_silhouette_score


def silhouette_score(x_data: NDArray, labels: NDArray) -> float:
    """
    Compute the mean silhouette score for a clustering.
    Vectorized over samples using a full distance matrix. Variable names kept consistent.
    """
    # Input validation
    if int(x_data.shape[0]) != int(labels.shape[0]):
        raise ValueError("we need a label for every sample in x_data")

    # calc all the distances between points-- this may be slow for huge datasets.
    distance_matrix = np.linalg.norm(x_data[:, None, :] - x_data[None, :, :], axis=2)

    # get counts & sizes collected
    unique_labels, inv = np.unique(labels, return_inverse=True)
    num_samples = int(x_data.shape[0])
    num_clusters = int(unique_labels.size)
    if num_clusters < 2 or num_samples == 0:
        return 0.0

    # get the size of each cluster based on the mask
    cluster_sizes = np.bincount(inv, minlength=num_clusters)

    # Sum distances from every sample to each cluster members
    summed_distances = np.empty((num_samples, num_clusters), dtype=distance_matrix.dtype)
    for k in range(num_clusters):
        clust_mask = inv == k
        summed_distances[:, k] = distance_matrix[:, clust_mask].sum(axis=1)


    # intra-cluster mean distance for each sample (exclude self)
    own_sizes = cluster_sizes[inv]
    own_sums = summed_distances[np.arange(num_samples), inv]
    with np.errstate(divide="ignore", invalid="ignore"):
        # I hate that where doesn't like kwargs.
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

    # Compute silhouette values per sample
    largest_delta = np.maximum(intra_dist, nearest_clust_dist)
    with np.errstate(divide="ignore", invalid="ignore"):
        # I hate that where doesn't like kwargs.
        scores = np.where(
            largest_delta > 0,
            (nearest_clust_dist - intra_dist) / largest_delta,
            0.0
        )

    return np.mean(scores)


def davies_bouldin_index(x_data: NDArray, labels: NDArray):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    # Calculate cluster centroids
    centroids = np.array(
        [
            x_data[labels == k].mean(axis=0)
            for k in unique_labels
        ]
    )

    # Calculate s_i for each cluster
    s = np.zeros(num_clusters)
    for i, k in enumerate(unique_labels):
        cluster_points = x_data[labels == k]
        s[i] = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))

    # Calculate Davies-Bouldin index
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


def calinski_harabasz_index(x_data, labels):
    unique_labels = np.unique(labels)
    n_samples, n_features = x_data.shape
    n_clusters = len(unique_labels)

    # Overall mean
    overall_mean = np.mean(x_data, axis=0)

    # Cluster means and sizes
    cluster_means = np.array(
        [np.mean(x_data[labels == k], axis=0) for k in unique_labels]
    )
    cluster_sizes = np.array(
        [np.sum(labels == k) for k in unique_labels]
    )

    # Between-cluster dispersion
    B_k = np.sum(cluster_sizes[:, None] * (cluster_means - overall_mean) ** 2)

    # Within-cluster dispersion
    W_k = 0
    for i, label in enumerate(unique_labels):
        cluster_data = x_data[labels == label]
        W_k += np.sum((cluster_data - cluster_means[i]) ** 2)

    return (B_k / W_k) * ((n_samples - n_clusters) / (n_clusters - 1))

# ---- performance metrics with labels ----

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

    # Get unique class and cluster indices
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)

    # Contingency matrix creation
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    cont_matrix = np.zeros((n_classes, n_clusters), dtype=np.int64)
    np.add.at(cont_matrix, (class_idx, cluster_idx), 1)

    # Total number of samples
    n_samples = np.sum(cont_matrix)

    # Marginal frequencies for the true classes
    class_freqs = np.sum(cont_matrix, axis=1)

    # Entropy of the true labels
    class_entropy = entropy(class_freqs)

    # Conditional entropy of class labels given cluster assignments
    cond_ent = 0.
    for i in range(n_clusters):
        cluster = cont_matrix[:, i]
        cluster_size = np.sum(cluster)
        if cluster_size > 0:
            cond_ent = entropy(cluster)

    cond_ent /= n_samples

    # Homogeneity score
    if class_entropy == 0:
        return 1.0
    return 1 - cond_ent / class_entropy
