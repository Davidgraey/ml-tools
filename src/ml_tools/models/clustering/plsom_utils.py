import copy
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from ml_tools.models.constants import EPSILON

# from smlib.mltools import log


def preformat_expected_shapes(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """ conform the shapes of input and targets, and recast into lower-precision dtype """
    x_prime = np.asarray(x).astype(np.float16)
    y_prime = np.asarray(y).astype(np.float16)

    if x.ndim == 1:  # single sample
        x_prime = x_prime.reshape(1, -1)
    if y.ndim == 1:
        y_prime = y_prime.reshape(-1, 1)

    assert x_prime.shape[-1] == y_prime.shape[-1]

    return x_prime, y_prime

# --------------- Standardization / Normalize ---------------
def update_mean_std(mean1, std1, count1, mean2, std2, count2):
    '''**********ARGUMENTS**********
    :param mean1: self.data_type_means
    :param std1: self.data_type_stds
    :param count1: num_before
    :param mean2: this_data_mean
    :param std2: this_data_std
    :param count2: num_new
    **********RETURNS**********
    :return: new mean value, new std value
    '''
    full_count = count1 + count2
    full_mean = (count1 * mean1 + count2 * mean2) / full_count
    var1 = std1 ** 2
    var2 = std2 ** 2
    # error sum of squares
    sum_square_errors = var1 * (count1 - 1) + var2 * (count2 - 1)
    # total group sum of squares
    sum_squares = (mean1 - full_mean) ** 2 * count1 + (mean2 - full_mean) ** 2 * count2
    full_var = (sum_square_errors + sum_squares) / (full_count - 1)
    full_std = np.sqrt(full_var)

    return full_mean, full_std


def scale_colormap(col: NDArray) -> NDArray:
    absmax = np.abs(col).max()
    norm = plt.Normalize(-absmax, absmax)
    cmap = plt.get_cmap("RdYlGn")
    colors = cmap(norm(col))
    return colors


def plot_variable_importance(variable_coef, feature_names):
    length = len(variable_coef)
    n_feats = len(feature_names)
    fig = plt.figure(figsize=(12, n_feats * length // 3))
    for i in range(length):
        plt.subplot(length + 1, 1, i + 1)
        # datset = datset.sort_values(by='p_score1', ascending=False)
        plt.bar(x=variable_coef[i], y=feature_names, palette=scale_colormap(variable_coef[i]))
        plt.ylabel('Feature', fontsize=10)
        plt.xlabel('Score', fontsize=18)

    return fig


def umat_fill_outer(u_matrix, cols, rows):
    """
    takes a partially constructed Rectangular U_matrix or distance matrix and fills in missing values

    """
    umatrix = copy.copy(u_matrix)
    for row_i in range(rows):
        for col_i in range(cols):
            # let's resolve corner cases, then do edges
            if row_i == 0 and col_i == 0:
                # topleft - only right and lower
                lower_v = umatrix[row_i + 1, col_i]
                right_v = umatrix[row_i, col_i + 1]
                umatrix[row_i, col_i] = np.mean((right_v, lower_v))
            elif row_i == 0 and col_i == cols - 1:
                # topright - only right and lower
                lower_v = umatrix[row_i + 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                umatrix[row_i, col_i] = np.mean((left_v, lower_v))
            elif row_i == rows - 1 and col_i == 0:
                # bottomleft - only right and upper
                right_v = umatrix[row_i, col_i + 1]
                upper_v = umatrix[row_i - 1, col_i]
                umatrix[row_i, col_i] = np.mean((right_v, upper_v))
            elif row_i == rows - 1 and col_i == cols - 1:
                # bottomright - only left and upper
                upper_v = umatrix[row_i - 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                umatrix[row_i, col_i] = np.mean((left_v, upper_v))

            # Left Edge
            elif 0 < row_i < rows - 1 and row_i % 2 == 0 and col_i == 0:
                upper_v = umatrix[row_i - 1, col_i]
                lower_v = umatrix[row_i + 1, col_i]
                right_v = umatrix[row_i, col_i + 1]
                umatrix[row_i, col_i] = np.mean((right_v, lower_v, upper_v))

            # Right Edge
            elif 0 < row_i < rows - 1 and row_i % 2 == 0 and col_i == cols - 1:
                upper_v = umatrix[row_i - 1, col_i]
                lower_v = umatrix[row_i + 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                umatrix[row_i, col_i] = np.mean((left_v, lower_v, upper_v))

            # Top Edge - corners calculated
            elif row_i == 0 and 0 < col_i < cols - 1 and col_i % 2 == 0:
                # print('top_edge')
                lower_v = umatrix[row_i + 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                right_v = umatrix[row_i, col_i + 1]
                umatrix[row_i, col_i] = np.mean((left_v, lower_v, right_v))

            # Bottom Edge - corners calculated
            elif row_i == rows - 1 and 0 < col_i < cols - 1 and col_i % 2 == 0:
                # print('bottom_edge')
                upper_v = umatrix[row_i - 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                right_v = umatrix[row_i, col_i + 1]
                umatrix[row_i, col_i] = np.mean((left_v, upper_v, right_v))

    return umatrix


def umat_infill(u_matrix, cols, rows):
    """
    Fills in the 'inbetween' values that are not directly calculable.
    This is highly inelegant
    """
    umatrix = copy.copy(u_matrix)
    for row_i in range(rows):
        for col_i in range(cols):
            # evens
            if 0 < row_i < rows - 1 and row_i % 2 == 0 and 0 < col_i < cols - 1 and col_i % 2 == 0:
                upper_v = umatrix[row_i - 1, col_i]
                lower_v = umatrix[row_i + 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                right_v = umatrix[row_i, col_i + 1]
                umatrix[row_i, col_i] = np.mean((upper_v, lower_v, left_v, right_v))

    # fill odds
    for row_i in range(rows):
        for col_i in range(cols):
            if 0 < row_i < rows - 1 and row_i % 2 != 0 and 0 < col_i < cols - 1 and col_i % 2 != 0:
                upper_v = umatrix[row_i - 1, col_i]
                lower_v = umatrix[row_i + 1, col_i]
                left_v = umatrix[row_i, col_i - 1]
                right_v = umatrix[row_i, col_i + 1]
                umatrix[row_i, col_i] = np.mean((upper_v, lower_v, left_v, right_v))

    return umatrix


class Watershed(object):
    MASK = -2
    WSHD = 0
    INIT = -1
    INQE = -3

    def __init__(self, levels=25):
        self.levels = levels

    # Neighbour (coordinates of) pixels, including the given pixel.
    def _get_neighbors(self, height, width, pixel):
        return np.mgrid[
            max(0, pixel[0] - 1):min(height, pixel[0] + 2),
            max(0, pixel[1] - 1):min(width, pixel[1] + 2)
        ].reshape(2, -1).T

    def apply(self, image):
        current_label = 0
        flag = False
        fifo = deque()

        height, width = image.shape
        total = height * width
        labels = np.full((height, width), self.INIT, np.int32)

        reshaped_image = image.reshape(total)
        # [y, x] pairs of pixel coordinates of the flattened image.
        pixels = np.mgrid[0:height, 0:width].reshape(2, -1).T
        # Coordinates of neighbour pixels for each pixel.
        neighbours = np.array([self._get_neighbors(height, width, p) for p in pixels])
        if len(neighbours.shape) == 3:
            # Case where all pixels have the same number of neighbours.
            neighbours = neighbours.reshape(height, width, -1, 2)

        else:
            # Case where pixels may have a different number of pixels.
            neighbours = neighbours.reshape(height, width)

        indices = np.argsort(reshaped_image)
        sorted_image = reshaped_image[indices]
        sorted_pixels = pixels[indices]

        # self.levels evenly spaced steps from minimum to maximum.
        levels = np.linspace(sorted_image[0], sorted_image[-1], self.levels)
        level_indices = []
        current_level = 0

        # Get the indices that deleimit pixels with different values.
        for i in range(total):
            if sorted_image[i] > levels[current_level]:
                # Skip levels until the next highest one is reached.
                while sorted_image[i] > levels[current_level]: current_level += 1
                level_indices.append(i)
        level_indices.append(total)

        start_index = 0
        for stop_index in level_indices:
            # Mask all pixels at the current level.
            for p in sorted_pixels[start_index:stop_index]:
                labels[p[0], p[1]] = self.MASK
                # Initialize queue with neighbours of existing basins at the current level.
                for q in neighbours[p[0], p[1]]:
                    # p == q is ignored here because labels[p] < WSHD
                    if labels[q[0], q[1]] >= self.WSHD:
                        labels[p[0], p[1]] = self.INQE
                        fifo.append(p)
                        break

            # Extend basins.
            while fifo:
                p = fifo.popleft()
                # Label p by inspecting neighbours.
                for q in neighbours[p[0], p[1]]:
                    # Don't set lab_p in the outer loop because it may change.
                    lab_p = labels[p[0], p[1]]
                    lab_q = labels[q[0], q[1]]
                    if lab_q > 0:
                        if lab_p == self.INQE or (lab_p == self.WSHD and flag):
                            labels[p[0], p[1]] = lab_q
                        elif lab_p > 0 and lab_p != lab_q:
                            labels[p[0], p[1]] = self.WSHD
                            flag = False
                    elif lab_q == self.WSHD:
                        if lab_p == self.INQE:
                            labels[p[0], p[1]] = self.WSHD
                            flag = True
                    elif lab_q == self.MASK:
                        labels[q[0], q[1]] = self.INQE
                        fifo.append(q)

            # Detect and process new minima at the current level.
            for p in sorted_pixels[start_index:stop_index]:
                # p is inside a new minimum. Create a new label.
                if labels[p[0], p[1]] == self.MASK:
                    current_label += 1
                    fifo.append(p)
                    labels[p[0], p[1]] = current_label
                    while fifo:
                        q = fifo.popleft()
                        for r in neighbours[q[0], q[1]]:
                            if labels[r[0], r[1]] == self.MASK:
                                fifo.append(r)
                                labels[r[0], r[1]] = current_label

            start_index = stop_index

        return labels


def blur(image: NDArray) -> NDArray:
    """
    blur the image array by the given kernel
    :param image: np array - imagelike
    :return: convolved kernel blur image
    """
    image_array = copy.deepcopy(image)
    kernel = np.array([1.0, 2.0, 1.0])
    image_array = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, image_array)
    image_array = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, image_array)
    return image_array


########### DISTANCES ###########


def manhattan_distance(x: NDArray, y: NDArray, summed: bool = True) -> NDArray:
    """
    aka Chebyshev
    :param x:
    :param y:
    :return:
    """
    x, y = preformat_expected_shapes(x, y)
    # sum(np.abs(x-y), axis=?)
    if summed:
        return np.sum(np.abs(x - y), axis=-1)
    else:
        return np.abs(x - y)


def grid_manhattan_distance(row_a, col_a, row_b, col_b):
    return abs(row_a - row_b) + abs(col_a - col_b)


def euclidian_distance(x: NDArray, y: NDArray) -> NDArray:
    """
    for size (5000, 100), per sample, np.sqrtsum method took 1.07ms
    linalg.norm took 1.21ms
    numpy 1.20.3
    :param x:
    :param y:
    :return:
    """
    x, y = preformat_expected_shapes(x, y)
    # np.linalg.norm(x - y)
    # featurewise = np.sqrt((x - y) ** 2)
    # summed = np.sum(featurewise, axis=-1)
    return np.sqrt(np.sum((x - y) ** 2, axis=-1))  # + EPSILON


def norm_euclidian_distance(x: NDArray, y: NDArray) -> NDArray:
    # numerator = (np.linalg.norm((x - np.mean(x)) - (y - np.mean(y))) ** 2)
    # denom = (np.linalg.norm(x - np.mean(x)) ** 2 + np.linalg.norm(y - np.mean(y)) ** 2)
    # return 0.5 * (numerator / denom)

    vx = np.var(x, axis=-1)
    vy = np.var(y, axis=-1)
    return 0.5 * (vx / (vy + vx))


def mahalonobis_distance(x: NDArray, y: NDArray) -> NDArray:
    """

    multivariate equivilant of euclidian distance, comparing point to distribution
    :param x:
    :param y:
    :return:
    """
    stacked_vecs = np.vstack([x, y])
    covariance = np.cov(stacked_vecs.T)
    inv_covariance = np.linalg.inv(covariance)
    delta = x - y
    # may have to fix reshape for multi-dim...
    return np.sqrt(np.einsum('nj,jk,nk->n', delta, inv_covariance, delta)).reshape(x.shape[0], -1)


def hamming_distance(x: NDArray, y: NDArray) -> NDArray:
    """
    :param x:
    :param y:
    :return:
    """
    # axis?
    return np.count_nonzero(x != y, axis=-1)


def cosine_distance(x: NDArray, y: NDArray) -> NDArray:
    """
    inverse cosine similarity
    :param x:
    :param y:
    :return:
    """
    x, y = preformat_expected_shapes(x, y)
    return 1 - cosine_similarity(x, y)


# +---------Similarity---------
def cosine_similarity(x: NDArray, y: NDArray) -> NDArray:
    """
    requires numpy v1.7+ for 'where' in np.divide()
    numpy array is [samples, dimension_1, dimension_2, ...]
    we will compute cosine distance per samples dimension
        input array is [500, 5] - output will be [500, 1]
    :param x:
    :param y:
    :return:
    """
    dot = np.sum(x * y, axis=-1)
    x_norm = np.sqrt(np.sum(x * x, axis=-1))
    y_norm = np.sqrt(np.sum(y * y, axis=-1))

    denom = x_norm * y_norm
    return dot / np.maximum(denom, EPSILON)


def jaccard_similarity(x: NDArray, y: NDArray) -> NDArray:
    """aka Jaccard Index
    for use in boolean / on-hot / binary array
    """
    return np.bitwise_and(x, y).sum(axis=-1) / np.bitwise_or(x, y).sum(axis=-1)
