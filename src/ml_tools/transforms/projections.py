from numpy.typing import NDArray
import numpy as np


def mca(
        data: NDArray,
        top_k_components: int = 3
) -> NDArray:
    """ expects data to be one-hot encodings - normalized by row / column masses and inertia correction """
    # we're expecting one-hot encodings; so Num_samples x num_variables 2D array
    org_shape = data.shape
    row_sum = np.sum(data, axis=1)
    column_sum = np.sum(data, axis=0)

    expected = np.outer(row_sum, column_sum)

    probability = data / np.sum(data)
    residuals = (probability - expected) / np.sqrt(expected)
    U, D, V = np.linalg.svd(residuals, full_matrices=False)

    # row_coords = np.diag(1.0 / np.sqrt(row_sum)) @ U @ D
    col_coords = np.diag(1.0 / np.sqrt(column_sum)) @ V.T @ D
    # deltas = np.diag(residuals)

    _x = data / data.sum(axis=1)

    return _x @ col_coords


def pca(
        data: NDArray,
        top_k_components: int = 3

) -> NDArray:
    """ faster? implementation - just do the reduction! """
    org_shape = data.shape
    # standardize--
    standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # TODO: this forces 2D -- this will get covaraince across all values - for future, implement a mechanim across
    #  the axes, ... cov_list = [np.cov(slice_2d) for slice_2d in data]
    standardized_data = standardized_data.reshape(org_shape[0], -1)
    covariance_matrix = np.cov(standardized_data, ddof=1, rowvar=False, bias=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # take the top_k components -argsort returns low-> high so we reverse index
    top_components = np.argsort(eigenvalues)[:-(top_k_components+1):-1]
    # take copy for memory optimization
    principal_components = eigenvectors[:, top_components].copy()

    # sorted_eigenvalues = eigenvalues[sorted_index]
    # sorted_eigenvectors = eigenvectors[:, sorted_index]
    #
    # # explained variance
    # r_squared = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    #
    # principal_components = sorted_eigenvectors[:, :k]
    # explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    #
    # total_explained_variance = sum(explained_variance[:k])
    # top_k_components

    return standardized_data @ principal_components
