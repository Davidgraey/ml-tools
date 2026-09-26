"""
Loss functions: backward() is the derivative of forward(), including its reduction.
"""

import numpy as np
import pytest

from polyergalio.models.constants import ClassificationTask
from polyergalio.models.model_loss import (
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


@pytest.mark.parametrize("loss_class", SCALAR_LOSSES)
@pytest.mark.parametrize("shape", ((6, 1), (6, 4)))
def test_scalar_loss_gradient(loss_class, shape):
    """a wrong reduction often passes on one column and breaks on several"""
    if loss_class is CosineLoss and shape[1] == 1:
        pytest.skip("cosine of one-dimensional vectors has a zero gradient")

    rng = np.random.default_rng(0)
    prediction = rng.normal(size=shape) * 2
    targets = rng.normal(size=shape)
    assert loss_gradient_error(loss_class(), prediction, targets) < GRADIENT_TOLERANCE


@pytest.mark.parametrize(
    "task",
    (
        ClassificationTask.MULTINOMIAL,
        ClassificationTask.BINARY,
        ClassificationTask.MULTILABEL,
    ),
)
def test_cross_entropy_gradient(task):
    """forward() squashes logits, so backward() must apply the matching softmax or sigmoid"""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 4)) * 2
    if task is ClassificationTask.MULTINOMIAL:
        targets = np.eye(4)[rng.integers(0, 4, 5)]
    else:
        targets = (rng.random((5, 4)) > 0.5).astype(float)

    loss = CrossEntropyLoss(task)
    assert loss_gradient_error(loss, logits, targets) < GRADIENT_TOLERANCE


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
