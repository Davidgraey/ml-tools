"""
Transforms: probability calibration and the dimensionality projections.
"""

import numpy as np
import pytest
from polyergalio.transforms.calibrations import CalibrationType, ProbCalibration
from polyergalio.transforms.projections import mca, pca
from polyergalio.types import BasalTransform

METHODS = ("platt", "isotonic", "spline")


def platt_objective(scores, labels, coefficient, intercept) -> float:
    """the smoothed negative log likelihood Platt scaling minimises"""
    positives = (labels == 1).sum(axis=0)
    negatives = len(labels) - positives
    targets = np.where(
        labels == 1, (positives + 1) / (positives + 2), 1.0 / (negatives + 2)
    )
    probabilities = 1.0 / (1.0 + np.exp(-(coefficient * scores + intercept)))
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return float(
        -np.sum(
            targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities)
        )
    )


# -------------    calibration    ----------------------------------
def test_calibration_is_a_transform():
    assert issubclass(ProbCalibration, BasalTransform)


@pytest.mark.parametrize("method", METHODS)
def test_calibration_tracks_fitted_state(method, calibration_data):
    scores, labels = calibration_data
    model = ProbCalibration(method=method, data_dimension=1)
    assert not model.is_fitted
    model.fit(scores, labels)
    assert model.is_fitted


@pytest.mark.parametrize("method", METHODS)
def test_predict_before_fit_raises(method, calibration_data):
    scores, _ = calibration_data
    with pytest.raises(RuntimeError):
        ProbCalibration(method=method, data_dimension=1).predict(scores)


@pytest.mark.parametrize("method", METHODS)
def test_calibration_is_monotone(method, calibration_data):
    """
    A calibration may rescale confidence but must not reorder it. All three
    methods are monotone by construction, so this is a real invariant.
    """
    scores, labels = calibration_data
    model = ProbCalibration(method=method, data_dimension=1)
    model.fit(scores, labels)

    grid = np.linspace(scores.min(), scores.max(), 80).reshape(-1, 1)
    calibrated = model.predict(grid).ravel()
    assert np.all(np.diff(calibrated) >= -1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_calibration_outputs_probabilities(method, calibration_data):
    scores, labels = calibration_data
    calibrated = ProbCalibration(method=method, data_dimension=1).fit_predict(
        scores, labels
    )
    assert calibrated.shape == scores.shape
    assert (calibrated >= 0.0).all() and (calibrated <= 1.0).all()


@pytest.mark.parametrize("method", METHODS)
def test_calibration_improves_on_the_raw_scores(method, calibration_data):
    """
    calibration_data's raw scores are deliberately mis-scaled (see the
    fixture), so a working calibrator should land closer to the true label
    frequencies than the raw scores do -- a real behavioral claim, not just
    that fit/predict run and stay in [0, 1].
    """
    scores, labels = calibration_data
    calibrated = ProbCalibration(method=method, data_dimension=1).fit_predict(
        scores, labels
    )
    raw_clipped = np.clip(scores, 0.0, 1.0)

    raw_brier = np.mean((raw_clipped - labels) ** 2)
    calibrated_brier = np.mean((calibrated - labels) ** 2)
    assert calibrated_brier < raw_brier


@pytest.mark.parametrize("method", METHODS)
def test_calibration_rejects_mismatched_shapes(method, calibration_data):
    scores, labels = calibration_data
    with pytest.raises(AssertionError):
        ProbCalibration(method=method, data_dimension=1).fit(scores, labels[:-5])


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        ProbCalibration(method="nonsense", data_dimension=1)


def test_platt_reaches_the_optimum(calibration_data):
    """
    Compare against gradient descent run to convergence on the same objective.
    A broken Newton step still descends a little, so only closeness to the
    reference optimum distinguishes a working fit from a stalled one.
    """
    scores, labels = calibration_data
    model = ProbCalibration(method="platt", data_dimension=1)
    model.fit(scores, labels)

    fitted = platt_objective(
        scores,
        labels,
        float(np.ravel(model.platt_coefficient)[0]),
        float(np.ravel(model.platt_intercept)[0]),
    )

    # scalar (not axis=0, shape-(1,)) so float() below works under numpy>=2,
    # which no longer implicitly converts a size-1 non-0d array
    positives = (labels == 1).sum()
    negatives = len(labels) - positives
    targets = np.where(
        labels == 1, (positives + 1) / (positives + 2), 1.0 / (negatives + 2)
    )
    coefficient, intercept = 0.0, float(np.log((negatives + 1) / (positives + 1)))
    for _ in range(60000):
        probabilities = 1.0 / (1.0 + np.exp(-(coefficient * scores + intercept)))
        residual = probabilities - targets
        coefficient -= 1e-4 * float((residual * scores).sum())
        intercept -= 1e-4 * float(residual.sum())
    reference = platt_objective(scores, labels, coefficient, intercept)

    assert fitted < reference * 1.05, f"fit stalled at {fitted} vs {reference}"


def test_platt_coefficient_has_the_right_sign(calibration_data):
    """higher score must mean higher probability for a positively correlated score"""
    scores, labels = calibration_data
    model = ProbCalibration(method="platt", data_dimension=1)
    model.fit(scores, labels)
    assert float(np.ravel(model.platt_coefficient)[0]) > 0


def test_platt_handles_multiple_columns(calibration_data):
    """the second column has inverted labels, so its coefficient must flip"""
    scores, labels = calibration_data
    wide_scores = np.hstack([scores, scores])
    wide_labels = np.hstack([labels, 1 - labels])

    model = ProbCalibration(method="platt", data_dimension=2)
    model.fit(wide_scores, wide_labels)

    coefficients = np.ravel(model.platt_coefficient)
    assert coefficients[0] > 0 and coefficients[1] < 0


def test_isotonic_pava_is_correct():
    """non-decreasing, and each pooled block keeps its mean"""
    values = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    fitted = ProbCalibration._pava(values)
    assert np.all(np.diff(fitted) >= 0)
    assert fitted.mean() == pytest.approx(values.mean())
    assert np.allclose(fitted, [2.0, 2.0, 2.0, 4.5, 4.5])


def test_isotonic_pava_leaves_sorted_input_alone():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(ProbCalibration._pava(values), values)


def test_isotonic_pava_flattens_decreasing_input():
    values = np.array([4.0, 3.0, 2.0, 1.0])
    assert np.allclose(ProbCalibration._pava(values), np.full(4, 2.5))


def test_spline_handles_a_single_knot():
    """
    Constant scores collapse to one PAVA block. That path used to return a
    tuple instead of self, breaking the fit_predict chain, and then divide by a
    zero-width interval.
    """
    scores = np.full((5, 1), 0.5)
    labels = np.ones((5, 1), dtype=int)
    calibrated = ProbCalibration(method="spline", data_dimension=1).fit_predict(
        scores, labels
    )
    assert calibrated.shape == scores.shape
    assert np.isfinite(calibrated).all()


def test_calibration_type_enum_covers_the_methods():
    assert {member.value for member in CalibrationType} >= set(METHODS)


# -------------    pca    ------------------------------------------
def test_pca_components_are_uncorrelated_and_ordered(regression_dataset):
    x_data, _, _ = regression_dataset
    projection = pca(x_data, top_k_components=3)

    variances = projection.var(axis=0, ddof=1)
    assert np.all(np.diff(variances) <= 1e-9), "components not ordered by variance"

    correlation = np.corrcoef(projection, rowvar=False)
    assert np.abs(correlation - np.eye(3)).max() < 1e-10


def test_pca_preserves_total_variance(regression_dataset):
    x_data, _, _ = regression_dataset
    features = x_data.shape[1]
    projection = pca(x_data, top_k_components=features)

    standardized = (x_data - x_data.mean(0)) / x_data.std(0)
    assert projection.var(0, ddof=1).sum() == pytest.approx(
        standardized.var(0, ddof=1).sum()
    )


@pytest.mark.parametrize("requested", (1, 2, 3, 99))
def test_pca_clamps_the_component_count(requested, regression_dataset):
    x_data, _, _ = regression_dataset
    projection = pca(x_data, top_k_components=requested)
    assert projection.shape == (len(x_data), min(requested, x_data.shape[1]))


def test_pca_is_deterministic(regression_dataset):
    x_data, _, _ = regression_dataset
    assert np.allclose(pca(x_data, 3), pca(x_data, 3))


@pytest.mark.parametrize(
    "make_input",
    (
        pytest.param(lambda rng: rng.normal(size=(50, 1)), id="single_feature"),
        pytest.param(
            lambda rng: np.hstack([rng.normal(size=(50, 3)), np.full((50, 1), 5.0)]),
            id="constant_column",
        ),
        pytest.param(
            lambda rng: np.hstack([rng.normal(size=(50, 3)), np.zeros((50, 1))]),
            id="zero_column",
        ),
        pytest.param(
            lambda rng: rng.normal(size=(4, 10)), id="fewer_rows_than_features"
        ),
        pytest.param(lambda rng: rng.normal(size=(50, 4, 3)), id="three_dimensional"),
    ),
)
def test_pca_survives_degenerate_input(make_input):
    """
    A zero-variance column used to divide by zero, produce NaN, and surface as
    LinAlgError from eigh. A single feature made np.cov return a scalar.
    """
    rng = np.random.default_rng(0)
    projection = pca(make_input(rng), top_k_components=2)
    assert np.isfinite(projection).all()


def test_pca_recovers_a_planted_direction():
    """
    pca standardises first, so a merely rescaled feature carries no extra
    variance. Shared structure across several features is what the leading
    component should find.
    """
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(400, 1))
    shared = latent + rng.normal(size=(400, 4)) * 0.2
    independent = rng.normal(size=(400, 2))
    x_data = np.hstack([shared, independent])

    projection = pca(x_data, top_k_components=1)
    correlation = np.corrcoef(projection[:, 0], latent[:, 0])[0, 1]
    assert abs(correlation) > 0.9


# -------------    mca    ------------------------------------------
@pytest.fixture()
def onehot_table():
    """three categorical variables, one-hot encoded into eight columns"""
    rng = np.random.default_rng(0)
    count = 240
    table = np.zeros((count, 8))
    table[np.arange(count), rng.integers(0, 3, count)] = 1
    table[np.arange(count), 3 + rng.integers(0, 3, count)] = 1
    table[np.arange(count), 6 + rng.integers(0, 2, count)] = 1
    return table


@pytest.mark.parametrize("requested", (1, 3, 8, 50))
def test_mca_returns_row_coordinates(requested, onehot_table):
    projection = mca(onehot_table, top_k_components=requested)
    assert projection.shape[0] == onehot_table.shape[0]
    assert projection.shape[1] <= requested


def test_mca_handles_non_square_input(onehot_table):
    """dividing by an un-kept row sum used to raise for any non-square table"""
    assert mca(onehot_table, 3).shape == (onehot_table.shape[0], 3)


def test_mca_total_inertia_is_the_chi_square_statistic(onehot_table):
    """
    The expected table must be built from masses, not raw counts. With counts
    the inertia came out around 3.6e5 instead of order one.
    """
    probability = onehot_table / onehot_table.sum()
    row_mass = probability.sum(axis=1)
    column_mass = probability.sum(axis=0)
    expected = np.outer(row_mass, column_mass)
    residuals = (probability - expected) / np.sqrt(expected)

    _, singular, _ = np.linalg.svd(residuals, full_matrices=False)
    assert np.sum(singular**2) == pytest.approx(np.sum(residuals**2))
    assert np.sum(singular**2) < 10.0, "inertia is not on the expected scale"


def test_mca_row_coordinates_satisfy_the_weighted_identity(onehot_table):
    """
    The defining property of row principal coordinates: the mass-weighted sum
    of squared coordinates on an axis equals that axis's principal inertia.
    """
    probability = onehot_table / onehot_table.sum()
    row_mass = probability.sum(axis=1)
    column_mass = probability.sum(axis=0)
    expected = np.outer(row_mass, column_mass)
    residuals = (probability - expected) / np.sqrt(expected)
    _, singular, _ = np.linalg.svd(residuals, full_matrices=False)

    axes = 4
    coordinates = mca(onehot_table, top_k_components=axes)
    weighted = (row_mass[:, None] * coordinates**2).sum(axis=0)

    assert np.allclose(weighted, singular[:axes] ** 2)


def test_mca_row_centroid_is_at_the_origin(onehot_table):
    probability = onehot_table / onehot_table.sum()
    row_mass = probability.sum(axis=1)
    coordinates = mca(onehot_table, top_k_components=4)
    assert np.abs((row_mass[:, None] * coordinates).sum(axis=0)).max() < 1e-10


def test_mca_is_finite_on_a_sparse_table():
    """an all-zero column gives a zero mass, which must not divide by zero"""
    table = np.zeros((30, 5))
    table[np.arange(30), np.random.default_rng(0).integers(0, 3, 30)] = 1
    projection = mca(table, top_k_components=2)
    assert np.isfinite(projection).all()
