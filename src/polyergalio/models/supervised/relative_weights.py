"""
Heuristic Method for Estimating the Relative Weight of Predictor Variables in Multiple Regression
https://www.researchgate.net/publication
/247721163_Determining_the_Relative_Importance_of_Predictors_in_Logistic_Regression_An_Extension_of_Relative_Weight_Analysis
https://arxiv.org/pdf/2106.14095.pdf

Relative Weight Analysis returns "importance scores" whose sum equals tothe overall R2 of a model; it’s normalized form allows us to say
“Feature _X _accounts for _Z% _of variance in target variable Y.
"""

import numpy as np
from numpy.typing import NDArray
import scipy.stats as ss
from polyergalio.models.supervised import log
from polyergalio.utilities import standardize_data
from polyergalio.models.supervised.scg_regression import GradientDescent
from polyergalio.models.constants import (
    ClassificationTask,
    EPSILON,
    determine_classification_task,
)


def relative_weights(x: NDArray, y: NDArray, logistic: bool = True) -> dict:
    """
    # Extension of RWA to logistic regressions:
    https://www.researchgate.net/publication/247721163_Determining_the_Relative_Importance_of_Predictors_in_Logistic_Regression_An_Extension_of_Relative_Weight_Analysis
    # applied logistic RWA
    # https://arxiv.org/pdf/2106.14095.pdf

    Parameters
    ----------
    x : ndarray - design matrix, our input data as an np array shaped
            (observations, n_variables)
    y : ndarray - target values
    logistic : Bool - if conducting logistic regression or just numeric regression

    Returns
    -------
    Results Dictionary:

    """
    num_samples, num_features = x.shape

    if logistic is True:
        task = determine_classification_task(y)
    else:
        task = "regression"
    print(f"targeting {task}")
    # standardize our raw design matrix
    d = ss.zscore(x)

    # q is already transposed in linalg.svd --
    #  U, s, Vh = svd(A, lapack_driver='gesvd')
    p, _delta, q = np.linalg.svd(d, full_matrices=False)

    z = p @ q
    z_std = ss.zscore(z)

    # Classification ----
    if task == ClassificationTask.BINARY:
        logit_model = GradientDescent(
            task=ClassificationTask.BINARY, use_elastic_reg=False
        )
        logit_model.fit(
            x_data=x,
            y_data=y.reshape(num_samples, -1),
            iterations=25,
            add_constant=True,
        )
        logits = logit_model.forward(x, has_bias_present=False)
        predict = logit_model.predict(x)

        # Regress the predicted log‐odds on Z to get bZ (OLS or standard linear regression in papers)
        grad_model = GradientDescent(
            task=ClassificationTask.BINARY, use_elastic_reg=False
        )
        grad_model.fit(
            x_data=z_std,
            y_data=y.reshape(num_samples, -1),
            iterations=25,
            add_constant=True,
        )
        unstd_beta = grad_model.weights[1:]

    elif task == ClassificationTask.MULTINOMIAL:
        logit_model = GradientDescent(
            task=ClassificationTask.MULTINOMIAL, use_elastic_reg=False
        )
        logit_model.fit(
            x_data=x,
            y_data=y.reshape(num_samples, -1),
            iterations=25,
            add_constant=True,
        )
        logits = logit_model.forward(x, has_bias_present=False)
        predict = logit_model.predict(x)

        # Regress the predicted log‐odds on Z to get bZ (OLS or standard linear regression in papers)
        grad_model = GradientDescent(
            task=ClassificationTask.MULTINOMIAL, use_elastic_reg=False
        )
        grad_model.fit(
            x_data=z_std,
            y_data=y.reshape(num_samples, -1),
            iterations=25,
            add_constant=True,
        )
        unstd_beta = grad_model.weights[1:]

    elif task == ClassificationTask.MULTILABEL:
        logit_model = GradientDescent(
            task=ClassificationTask.MULTILABEL, use_elastic_reg=False
        )
        logit_model.fit(
            x_data=x,
            y_data=y.reshape(num_samples, -1),
            iterations=25,
            add_constant=True,
        )
        logits = logit_model.forward(x, has_bias_present=False)
        predict = logit_model.predict(x)

        # Regress the predicted log‐odds on Z to get bZ (OLS or standard linear regression in papers)
        grad_model = GradientDescent(
            task=ClassificationTask.MULTILABEL, use_elastic_reg=False
        )
        grad_model.fit(
            x_data=z_std,
            y_data=y.reshape(num_samples, -1),
            iterations=25,
            add_constant=True,
        )
        unstd_beta = grad_model.weights[1:]

    # Regression ----
    else:
        # Regress Y on Z to get bZ (OLS or standard linear regression in papers)
        # np.linalg.lstsq()
        grad_model = GradientDescent(task="regression", use_elastic_reg=False)
        grad_model.fit(
            x_data=z_std,
            y_data=y.reshape(num_samples, -1),
            iterations=100,
            add_constant=True,
        )
        predict = grad_model.predict(x)
        unstd_beta = grad_model.weights[1:]

    log.info(f"Link y^ to y: {grad_model.weights}")

    r2 = np.abs(grad_model.r_square)
    # r2_adj = grad_model.adjusted_r_square
    # residuals = grad_model.get_residuals()
    # Lambda_star = z_std.T @ d

    if logistic:
        # use the y_hat (logits -- not probability / sigmoid!)
        std_logit = np.std(logits)
        # estimate standardized coefficients (betastar)
        # np.std(z_std, axis=0)  # should all be 1.0, so we can skip s_Z in the paper
        beta = (unstd_beta * np.sqrt(r2 + EPSILON)) / (std_logit + EPSILON)

    else:
        # beta is just our raw coefficients since we are using linear model -
        # we'll call it beta for simplicity
        beta = unstd_beta

    signs = [np.sign(beta) for beta in beta]

    # Link funciton ------
    lambda_star = np.linalg.inv(z_std.T @ z_std) @ (z_std.T @ d)
    # back-project our coefficients into x-space
    # beta_projected = lambda_star @ beta

    relative_w = (lambda_star**2) @ (beta**2)

    # epsilon - our relative weight value
    # relative_w = lambda_star ** 2 @ beta ** 2
    if relative_w.shape[-1] > 2:
        _max = np.max(relative_w, axis=0)
        _min = np.min(relative_w, axis=0)
    else:
        _max = np.max(relative_w)
        _min = np.min(relative_w)
    normalized_weights = (relative_w - _min) / (_max - _min + EPSILON)
    # logging.info(f'rwa completed')

    return {
        "rwa": signs * relative_w,
        "norm_rwa": normalized_weights,
        "sign_norm_rwa": normalized_weights * signs,
        "betas": beta,
        "r2": r2,
        "model_prediction": predict,
    }
