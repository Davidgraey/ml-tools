"""
Loss functions.

The property that matters most here is that backward() is the derivative of
forward() -- including its reduction. A loss that averages over every element
but divides its gradient by the batch size only is wrong by a factor of the
feature count, which is invisible until you compare against finite differences.
"""

import numpy as np
import pytest

from ml_tools.models.constants import ClassificationTask
from ml_tools.models.model_loss import (
    CosineLoss,
    CrossEntropyLoss,
    MAELoss,
    MSELoss,
    MultiHeadLoss,
    RMSELoss,
)
from conftest import GRADIENT_TOLERANCE, numeric_gradient, relative_error


SCALAR_LOSSES = (MSELoss, RMSELoss, MAELoss, CosineLoss)


def loss_gradient_error(loss, prediction, targets) -> float:
    prediction = np.array(prediction, dtype=np.float64)
    loss.forward(prediction, targets)
    analytic = np.asarray(loss.backward(), dtype=np.float64)
    numeric = numeric_gradient(
        lambda: float(loss.forward(prediction, targets)), prediction
    )
    return relative_error(analytic, numeric)


@pytest.mark.slow
@pytest.mark.parametrize("loss_class", SCALAR_LOSSES)
@pytest.mark.parametrize("shape", ((6, 1), (6, 4)))
def test_scalar_loss_gradient(loss_class, shape):
    """
    Both shapes matter: a loss whose reduction is wrong often still passes on
    single-output data and only breaks when there are several columns.
    """
    if loss_class is CosineLoss and shape[1] == 1:
        pytest.skip("cosine of one-dimensional vectors is +-1, so the gradient "
                    "is identically zero and the comparison is vacuous")

    rng = np.random.default_rng(0)
    prediction = rng.normal(size=shape) * 2
    targets = rng.normal(size=shape)
    assert loss_gradient_error(loss_class(), prediction, targets) < GRADIENT_TOLERANCE


@pytest.mark.parametrize("loss_class", SCALAR_LOSSES)
def test_scalar_loss_returns_a_scalar(loss_class):
    rng = np.random.default_rng(0)
    value = loss_class().forward(rng.normal(size=(5, 3)), rng.normal(size=(5, 3)))
    assert np.isscalar(value) or np.asarray(value).ndim == 0


def test_mse_is_zero_at_a_perfect_fit():
    values = np.linspace(0, 1, 12).reshape(4, 3)
    assert MSELoss().forward(values, values) == pytest.approx(0.0)


def test_mse_reduction_covers_every_element():
    """mean over all elements, so widening the target must not change the value"""
    rng = np.random.default_rng(0)
    difference = rng.normal(size=(8, 5))
    loss = MSELoss().forward(difference, np.zeros_like(difference))
    assert loss == pytest.approx((difference ** 2).mean())


def test_mse_gradient_scales_with_size():
    """the 2/size factor, which was previously 2/batch"""
    rng = np.random.default_rng(0)
    prediction = rng.normal(size=(6, 4))
    targets = rng.normal(size=(6, 4))
    loss = MSELoss()
    loss.forward(prediction, targets)
    expected = 2 * (prediction - targets) / prediction.size
    assert np.allclose(loss.backward(), expected)


@pytest.mark.slow
@pytest.mark.parametrize(
    "task",
    (
        ClassificationTask.MULTINOMIAL,
        ClassificationTask.BINARY,
        ClassificationTask.MULTILABEL,
    ),
)
def test_cross_entropy_gradient(task):
    """
    forward() consumes logits and applies its own squashing, so backward() has
    to apply the matching softmax or sigmoid. Omitting it leaves the gradient
    pointing somewhere else entirely.
    """
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 4)) * 2
    if task is ClassificationTask.MULTINOMIAL:
        targets = np.eye(4)[rng.integers(0, 4, 5)]
    else:
        targets = (rng.random((5, 4)) > 0.5).astype(float)

    loss = CrossEntropyLoss(task)
    assert loss_gradient_error(loss, logits, targets) < GRADIENT_TOLERANCE


def test_cross_entropy_prefers_the_true_class():
    """confident and correct must score below confident and wrong"""
    targets = np.eye(3)[[0, 1]]
    correct = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    wrong = np.array([[0.0, 0.0, 6.0], [6.0, 0.0, 0.0]])
    loss = CrossEntropyLoss(ClassificationTask.MULTINOMIAL)
    assert loss.forward(correct, targets) < loss.forward(wrong, targets)


def test_cross_entropy_is_stable_at_extreme_logits():
    targets = np.eye(3)[[0]]
    extreme = np.array([[900.0, -900.0, 0.0]])
    loss = CrossEntropyLoss(ClassificationTask.MULTINOMIAL)
    assert np.isfinite(loss.forward(extreme, targets))
    assert np.isfinite(loss.backward()).all()


def test_multihead_loss_weights_and_splits():
    rng = np.random.default_rng(0)
    first = rng.normal(size=(4, 2))
    second = rng.normal(size=(4, 3))
    targets = [np.zeros((4, 2)), np.zeros((4, 3))]

    combined = MultiHeadLoss([MSELoss(), MSELoss()], weights=[1.0, 0.0])
    total = combined.forward([first, second], targets)

    assert total == pytest.approx(MSELoss().forward(first, targets[0]))
    gradients = combined.backward()
    assert len(gradients) == 2
    assert np.allclose(gradients[1], 0.0), "a zero weight must zero its gradient"


def test_rmse_matches_root_of_mse():
    rng = np.random.default_rng(0)
    prediction = rng.normal(size=(7, 3))
    targets = rng.normal(size=(7, 3))
    assert RMSELoss().forward(prediction, targets) == pytest.approx(
        np.sqrt(MSELoss().forward(prediction, targets))
    )


def test_mae_is_the_mean_absolute_difference():
    prediction = np.array([[1.0, -2.0]])
    targets = np.array([[0.0, 0.0]])
    assert MAELoss().forward(prediction, targets) == pytest.approx(1.5)


def test_cosine_loss_is_zero_for_parallel_vectors():
    vectors = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    assert CosineLoss().forward(vectors, vectors * 2.0) == pytest.approx(0.0, abs=1e-8)
