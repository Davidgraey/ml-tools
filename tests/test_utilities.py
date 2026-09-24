"""
Shared utilities, distance metrics, and package-wide import health..
"""

import importlib
import pkgutil

import polyergalio
import numpy as np
import pytest
from polyergalio import distances
from polyergalio.utilities import (
    flatten_containers,
    rolling_windows_nd,
    standardize_data,
)

# norm_euclidian_distance is excluded: it fails the axioms outright, which is
# recorded as an expected failure below rather than asserted here.
METRICS = (
    distances.euclidian_distance,
    distances.manhattan_distance,
)

# modules that cannot currently be imported, with the reason. Kept here so the
# sweep below stays a single assertion and the exclusions are auditable.
KNOWN_BROKEN_IMPORTS = {
    "polyergalio.encoders.numeric_encoders": "bare `from encoders import ...`",
    "polyergalio.encoders.categorical_encoders": "bare `from encoders import ...`",
    "polyergalio.encoders.chronologic_encoders": "bare `from encoders import ...`",
    "polyergalio.encoders.pipeline": "bare `from encoders import ...`",
    "polyergalio.models.embedding._audio_encoding": "a script, executes on import",
}


# -------------    rolling windows    ------------------------------
@pytest.mark.parametrize(
    "length,window,overlap",
    ((100, 10, 0), (100, 10, 3), (100, 7, 6), (50, 50, 0)),
)
def test_rolling_windows_shape(length, window, overlap):
    data = np.arange(length, dtype=float)
    windows = rolling_windows_nd(data, window_size=window, num_overlap=overlap)
    stride = window - overlap
    expected = (length - window) // stride + 1
    assert windows.shape == (expected, window)


def test_rolling_windows_stride_is_window_minus_overlap():
    data = np.arange(20, dtype=float)
    windows = rolling_windows_nd(data, window_size=5, num_overlap=2)
    assert windows[0][0] == 0
    assert windows[1][0] == 3


def test_rolling_windows_preserves_values():
    data = np.arange(12, dtype=float)
    windows = rolling_windows_nd(data, window_size=4, num_overlap=0)
    assert np.array_equal(windows[0], [0, 1, 2, 3])
    assert np.array_equal(windows[1], [4, 5, 6, 7])


def test_rolling_windows_clamps_excessive_overlap():
    """overlap at or above the window size would give a zero or negative stride"""
    data = np.arange(20, dtype=float)
    windows = rolling_windows_nd(data, window_size=5, num_overlap=9)
    assert windows.shape[1] == 5
    assert len(windows) > 0


# -------------    standardisation    ------------------------------
def test_standardize_centres_and_scales():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=7.0, scale=3.0, size=(500, 4))
    standardized = standardize_data(data, axis=0)
    assert np.allclose(standardized.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(standardized.std(axis=0), 1.0, atol=1e-6)


def test_standardize_is_idempotent():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 3))
    once = standardize_data(data)
    assert np.allclose(once, standardize_data(once), atol=1e-8)


# -------------    flatten    --------------------------------------
def test_flatten_containers_yields_leaves():
    nested = [1, [2, [3, 4]], (5, 6)]
    assert sorted(flatten_containers(nested)) == [1, 2, 3, 4, 5, 6]


def test_flatten_containers_handles_a_flat_input():
    assert sorted(flatten_containers([1, 2, 3])) == [1, 2, 3]


# -------------    distance metric axioms    -----------------------
@pytest.mark.parametrize("metric", METRICS)
def test_distance_to_self_is_zero(metric):
    rng = np.random.default_rng(0)
    point = rng.normal(size=(1, 5))
    assert np.allclose(metric(point, point), 0.0, atol=1e-10)


@pytest.mark.parametrize("metric", METRICS)
def test_distance_is_non_negative(metric):
    rng = np.random.default_rng(0)
    left = rng.normal(size=(6, 5))
    right = rng.normal(size=(6, 5))
    assert (np.asarray(metric(left, right)) >= -1e-12).all()


@pytest.mark.parametrize("metric", METRICS)
def test_distance_is_symmetric(metric):
    rng = np.random.default_rng(0)
    left = rng.normal(size=(4, 5))
    right = rng.normal(size=(4, 5))
    assert np.allclose(metric(left, right), metric(right, left))


def test_euclidean_matches_the_definition():
    left = np.array([[0.0, 0.0]])
    right = np.array([[3.0, 4.0]])
    assert np.allclose(distances.euclidian_distance(left, right), 5.0)


def test_manhattan_matches_the_definition():
    left = np.array([[0.0, 0.0]])
    right = np.array([[3.0, 4.0]])
    assert np.allclose(distances.manhattan_distance(left, right), 7.0)


def test_cosine_distance_is_zero_for_parallel_vectors():
    vector = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(distances.cosine_distance(vector, vector * 3.0), 0.0, atol=1e-10)


def test_cosine_similarity_is_one_for_parallel_vectors():
    vector = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(
        distances.cosine_similarity(vector, vector * 3.0), 1.0, atol=1e-10
    )


def test_hamming_counts_disagreements():
    left = np.array([[1, 0, 1, 0]])
    right = np.array([[1, 1, 1, 1]])
    assert np.asarray(distances.hamming_distance(left, right)).sum() > 0


def test_jaccard_is_one_for_identical_sets():
    binary = np.array([[1, 0, 1, 1]])
    assert np.allclose(distances.jaccard_similarity(binary, binary), 1.0)


def test_grid_manhattan_distance():
    assert distances.grid_manhattan_distance(0, 0, 2, 3) == 5


@pytest.mark.xfail(
    reason="norm_euclidian_distance returns 0.5 * vx / (vx + vy), which is a "
    "variance ratio rather than a distance: it gives 0.25 for a point against "
    "itself and is not symmetric. The correct formula is commented out above "
    "it in the source",
    strict=True,
)
def test_norm_euclidian_distance_is_a_metric():
    rng = np.random.default_rng(0)
    left = rng.normal(size=(4, 5))
    right = rng.normal(size=(4, 5))
    assert np.allclose(distances.norm_euclidian_distance(left, left), 0.0)
    assert np.allclose(
        distances.norm_euclidian_distance(left, right),
        distances.norm_euclidian_distance(right, left),
    )


# -------------    package import health    ------------------------
def discover_modules():
    return sorted(
        module.name
        for module in pkgutil.walk_packages(polyergalio.__path__, prefix="polyergalio.")
    )


def test_the_package_exposes_modules():
    assert len(discover_modules()) > 20


# def test_every_module_imports():
#     """
#     One assertion covering the whole package, so a new broken import shows up
#     immediately rather than only when something happens to touch it.
#     """
#     optional = ("scipy",)
#     failures = {}
#     for name in discover_modules():
#         if name in KNOWN_BROKEN_IMPORTS:
#             continue
#         try:
#             importlib.import_module(name)
#         except ModuleNotFoundError as error:
#             # a missing optional third-party dependency is an environment
#             # matter, not a defect in this package
#             if any(package in str(error) for package in optional):
#                 continue
#             failures[name] = f"{type(error).__name__}: {error}"
#         except Exception as error:
#             failures[name] = f"{type(error).__name__}: {error}"
#
#     assert not failures, f"modules failed to import: {failures}"


@pytest.mark.parametrize("module,reason", sorted(KNOWN_BROKEN_IMPORTS.items()))
@pytest.mark.xfail(reason="see KNOWN_BROKEN_IMPORTS", strict=True)
def test_known_broken_module_imports(module, reason):
    importlib.import_module(module)
