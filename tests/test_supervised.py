"""
Supervised models: GradientDescent (SCG) and the tree-based models.

Behavioral tests -- each model actually has to recover the fixture's planted
structure (R^2 / accuracy above a real threshold), not just run without
raising. Calling conventions here mirror each model's own __main__ demo.
"""

import numpy as np
import pytest

from polyergalio.generators.data_generators import to_onehot
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.supervised.scg_regression import GradientDescent
from polyergalio.models.supervised.trees.tree_models import (
    ExplainableBoostedTreeModel,
    SupervisedTreeModel,
)


def _ols_r_square(x_data, y_data) -> float:
    """closed-form OLS R^2, in-sample -- the ceiling a correct linear fit can reach."""
    design = np.hstack([np.ones((len(x_data), 1)), x_data])
    beta, *_ = np.linalg.lstsq(design, y_data, rcond=None)
    predicted = design @ beta
    residual = np.sum((y_data - predicted) ** 2)
    total = np.sum((y_data - y_data.mean()) ** 2)
    return 1 - residual / total


# -------------    GradientDescent    --------------------------------
def test_gradient_descent_fits_regression(regression_dataset):
    """
    regression_dataset is deliberately noisy (noise_scale=1.5 against
    roughly unit-scale signal), so an R^2 near 1 is not achievable here --
    what a correctly working linear fit CAN do is close in on the
    closed-form OLS optimum for this same data.
    """
    x_data, y_data, _ = regression_dataset
    ceiling = _ols_r_square(x_data, y_data)

    model = GradientDescent(task="regression")
    model.fit(x_data=x_data, y_data=y_data.reshape(-1, 1), iterations=60)

    assert model.r_square > 0.7 * ceiling


def test_gradient_descent_fits_binary(binary_dataset):
    x_data, y_data, _ = binary_dataset
    model = GradientDescent(task=ClassificationTask.BINARY)
    targets = to_onehot(y_data, 2)
    model.fit(x_data=x_data, y_data=targets, iterations=60)

    predicted = np.argmax(model.predict(x_data), axis=-1)
    accuracy = np.mean(predicted == y_data)
    assert accuracy > 0.7


def test_gradient_descent_fits_multiclass(multiclass_dataset):
    x_data, y_data, _ = multiclass_dataset
    model = GradientDescent(task=ClassificationTask.MULTINOMIAL)
    targets = to_onehot(y_data, 4)
    model.fit(x_data=x_data, y_data=targets, iterations=60)

    predicted = np.argmax(model.predict(x_data), axis=-1)
    accuracy = np.mean(predicted == y_data)
    assert accuracy > 0.5


def test_gradient_descent_fits_multilabel(multilabel_dataset):
    x_data, y_data, _ = multilabel_dataset
    model = GradientDescent(task=ClassificationTask.MULTILABEL)
    model.fit(x_data=x_data, y_data=y_data, iterations=60)

    predicted = np.where(model.predict(x_data) > 0.5, 1, 0)
    accuracy = np.mean(predicted == y_data)
    assert accuracy > 0.7


# -------------    ExplainableBoostedTreeModel    ---------------------
def test_ebm_fits_regression(regression_dataset):
    """
    Same noisy fixture as test_gradient_descent_fits_regression, so the
    absolute R^2 that matters is relative, not near 1: a working additive
    boosted model, with its per-feature bins, should comfortably beat the
    closed-form OLS ceiling a plain linear fit is stuck at on this data.
    """
    x_data, y_data, _ = regression_dataset
    ceiling = _ols_r_square(x_data, y_data)

    model = ExplainableBoostedTreeModel(
        input_dimension=x_data.shape[1], output_dimension=1, task=None,
        num_bins=24, learning_rate=0.1, num_rounds=60,
    )
    model.fit(x_data, y_data)

    predicted = model.predict(x_data)
    residual = np.sum((y_data - predicted) ** 2)
    total = np.sum((y_data - y_data.mean()) ** 2)
    r_square = 1 - residual / total
    assert r_square > 3 * ceiling


def test_ebm_fits_binary(binary_dataset):
    x_data, y_data, _ = binary_dataset
    model = ExplainableBoostedTreeModel(
        input_dimension=x_data.shape[1], output_dimension=1,
        task=ClassificationTask.BINARY, num_bins=16, learning_rate=0.05,
        num_rounds=40,
    )
    model.fit(x_data, y_data)

    predicted = model.predict(x_data)
    accuracy = np.mean(predicted == y_data)
    assert accuracy > 0.7


def test_ebm_fits_multiclass(multiclass_dataset):
    x_data, y_data, _ = multiclass_dataset
    targets = to_onehot(y_data, 4)
    model = ExplainableBoostedTreeModel(
        input_dimension=x_data.shape[1], output_dimension=4,
        task=ClassificationTask.MULTINOMIAL, num_bins=16, learning_rate=0.05,
        num_rounds=40,
    )
    model.fit(x_data, targets)

    predicted = model.predict(x_data)
    accuracy = np.mean(predicted == y_data)
    assert accuracy > 0.5


# -------------    SupervisedTreeModel    ------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="SupervisedTreeModel has no implementation yet -- its class body "
    "is just `pass`, so it doesn't satisfy BasalModel's abstract interface "
    "(forward/predict/fit/fit_predict/calculate_loss) and can't even be "
    "instantiated",
)
def test_supervised_tree_model_fits_and_predicts(binary_dataset):
    x_data, y_data, _ = binary_dataset
    model = SupervisedTreeModel()
    model.fit(x_data, y_data)
    predicted = model.predict(x_data)
    assert predicted.shape[0] == x_data.shape[0]
