from enum import Enum
import numpy as np
from numpy.typing import NDArray

EPSILON = 1e-15

# TOLERANCES AND LIMITS FOR SCALED CONJUGATE GRADIENT DESCENT (scg_regression)
FLOAT_TOLERANCE = 1e-30
SIGMA_ZERO = 1e-6
LAMBDA_MAX = 1e24
LAMBDA_MIN =  1e-20


class ClassificationTask(Enum):
    BINARY = "binary"
    MULTINOMIAL = "multinomial"
    MULTILABEL = "multilabel"


class Reductions(Enum):
    MEAN = 'mean'
    SUM = 'sum'
    NONE = None


def determine_classification_task(targets: NDArray) -> ClassificationTask:
    """
    Takes one-hot targets
    Parameters
    ----------
    targets :

    Returns
    -------

    """
    if len(targets.shape) == 1:
        targets = targets.reshape(-1, 1)

    num_numeric_classes = np.ptp(targets)
    last_dim = targets.shape[-1]
    num_classes = max(num_numeric_classes, last_dim)
    class_per_sample = max(np.sum(targets, axis=-1))

    if class_per_sample == 1:
        if num_classes <= 2:
            task = ClassificationTask("binary")

        elif num_classes > 2:
            task = ClassificationTask("multinomial")

    elif class_per_sample > 1:
        task = ClassificationTask("multilabel")

    return task

