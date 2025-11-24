# https://www.researchgate.net/publication/247721163_Determining_the_Relative_Importance_of_Predictors_in_Logistic_Regression_An_Extension_of_Relative_Weight_Analysis
# https://arxiv.org/pdf/2106.14095.pdf

# Relative Weight Analysis returns "importance scores" whose sum equals to
# the overall R2 of a model; it’s normalized form allows us to say
# “Feature _X _accounts for _Z% _of variance in target variable Y.

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import scipy.stats as ss
from .scg_regression import GradientDescent
from .constants import ClassificationTask, EPSILON, determine_classification_task


def standardize_data(design_matrix: NDArray, axis=0):
    """
    zero tge mean and variance = 1 along axis
    Parameters
    ----------
    design_matrix :
    axis :

    Returns
    -------

    """
    array_mean = np.mean(design_matrix, axis=axis)
    array_std = np.std(design_matrix, axis=axis)
    return (design_matrix - array_mean) / array_std


def relative_weights(x: NDArray, y: NDArray, logistic: bool=True) -> dict:
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

    # d = standardize_data(x, axis=0)


    # q is already transposed in linalg.svd --
    #  U, s, Vh = svd(A, lapack_driver='gesvd')
    p, _delta, q = np.linalg.svd(d, full_matrices=False)

    z = p @ q
    z_std = ss.zscore(z)

    # Classification ----
    if task == ClassificationTask.BINARY:
        logit_model = GradientDescent(task=ClassificationTask.BINARY, use_elastic_reg=False)
        logit_model.fit(x_data=x, y_data=y.reshape(num_samples, -1), iterations=25, add_constant=True)
        logits = logit_model.forward(x, has_bias_present=False)
        predict = logit_model.predict(x)

        # Regress the predicted log‐odds on Z to get bZ (OLS or standard linear regression in papers)
        grad_model = GradientDescent(task=ClassificationTask.BINARY, use_elastic_reg=False)
        grad_model.fit(x_data=z_std, y_data=y.reshape(num_samples, -1), iterations=25, add_constant=True)
        unstd_beta = grad_model.weights[1:]

    elif task == ClassificationTask.MULTINOMIAL:
        logit_model = GradientDescent(task=ClassificationTask.MULTINOMIAL, use_elastic_reg=False)
        logit_model.fit(x_data=x, y_data=y.reshape(num_samples, -1), iterations=25, add_constant=True)
        logits = logit_model.forward(x, has_bias_present=False)
        predict = logit_model.predict(x)

        # Regress the predicted log‐odds on Z to get bZ (OLS or standard linear regression in papers)
        grad_model = GradientDescent(task=ClassificationTask.MULTINOMIAL, use_elastic_reg=False)
        grad_model.fit(x_data=z_std, y_data=y.reshape(num_samples, -1), iterations=25, add_constant=True)
        unstd_beta = grad_model.weights[1:]

    elif task == ClassificationTask.MULTILABEL:
        logit_model = GradientDescent(task=ClassificationTask.MULTILABEL, use_elastic_reg=False)
        logit_model.fit(x_data=x, y_data=y.reshape(num_samples, -1), iterations=25, add_constant=True)
        logits = logit_model.forward(x, has_bias_present=False)
        predict = logit_model.predict(x)

        # Regress the predicted log‐odds on Z to get bZ (OLS or standard linear regression in papers)
        grad_model = GradientDescent(task=ClassificationTask.MULTILABEL, use_elastic_reg=False)
        grad_model.fit(x_data=z_std, y_data=y.reshape(num_samples, -1), iterations=25, add_constant=True)
        unstd_beta = grad_model.weights[1:]


    # Regression ----
    else:
        # Regress Y on Z to get bZ (OLS or standard linear regression in papers)
        # np.linalg.lstsq()
        grad_model = GradientDescent(task="regression", use_elastic_reg=False)
        grad_model.fit(x_data=z_std, y_data=y.reshape(num_samples, -1), iterations=100, add_constant=True)
        predict = grad_model.predict(x)
        unstd_beta = grad_model.weights[1:]


    log.info('Link y^ to y:', grad_model.weights)

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

    relative_w = (lambda_star ** 2) @ (beta ** 2)

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

    return {'rwa': signs * relative_w,
            'norm_rwa': normalized_weights,
            'sign_norm_rwa': normalized_weights * signs,
            'betas': beta,
            'r2': r2,
            'model_prediction': predict,
            }


if __name__ == "__main__":
    from data_generators import RandomDatasetGenerator, to_onehot

    num_samples = 2000
    num_features = 8
    NOISE = 0.05
    N_steps = 100

    gen = RandomDatasetGenerator(random_seed=123)
    # ========= regression dataset ========

    x, y, meta = gen.generate(task="regression",
                              num_samples=num_samples,
                              num_features=num_features,
                              noise_scale=NOISE)
    coef = np.abs(meta["weights"])
    lin_res = relative_weights(x, y, logistic=False)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("pred v target")
    plt.scatter(lin_res['model_prediction'], y)
    plt.subplot(1, 2, 2)
    plt.title("regression RWA")
    to_plot = (coef - np.min(coef)) / (np.max(coef) - np.min(coef) + EPSILON)
    plt.barh(range(len(to_plot)), to_plot, alpha=0.75, color="blue")
    to_plot = lin_res['sign_norm_rwa'].ravel() / np.max(lin_res['sign_norm_rwa'].ravel())
    plt.barh(range(len(to_plot)), to_plot, alpha=0.75, color="orange")
    plt.legend(["data_beta", "RWA"])
    plt.show()

    # ========= classificaiton dataset ========
    num_classes = 2
    x, y, meta = gen.generate(task="binary",
                              num_samples=num_samples,
                              num_classes=num_classes,
                              num_features=num_features,
                              noise_scale=NOISE)

    coef = np.abs(meta["weights"])
    _y = to_onehot(y)
    print(_y.shape)
    log_res = relative_weights(x, _y, logistic=True)

    pred = log_res['model_prediction']
    pred = np.argmax(pred, axis=-1)
    print("correct: ", np.sum(pred == y))
    print("wrong: ", np.sum(pred != y))
    print("accuracy: ", np.sum(pred == y) / num_samples)
    print(y.shape, pred.shape, )
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("pred v target")
    spread = list(range(len(x)))
    diff = np.abs(y - pred)
    noise = np.random.uniform(0.5, 1, size=num_samples)
    diff = ((diff * noise) / num_classes * num_samples)
    for idx in range(num_samples):
        sign = np.random.choice([-1, 1])
        plt.scatter(spread[idx], (spread[idx] + sign * diff[idx]), alpha=0.5, color="blue")
    plt.subplot(1, 2, 2)
    plt.title("Classificaiton RWA")
    to_plot = coef / np.max(coef)
    plt.barh(range(len(to_plot)), to_plot, alpha=0.75, color="blue")

    to_plot = np.sum(log_res['norm_rwa'], axis=-1)
    plt.barh(range(len(to_plot)), to_plot, alpha=0.5, color="orange")
    plt.legend(["data_beta", "RWA"])
    plt.tight_layout()
    plt.show()

    # ========= multinomial dataset ========
    num_classes = 6
    x, y, meta = gen.generate(task="multiclass",
                              num_samples=num_samples,
                              num_classes=num_classes,
                              num_features=num_features,
                              noise_scale=NOISE)

    coef = np.abs(meta["weights"])
    _y = to_onehot(y)
    print(_y.shape)
    log_res = relative_weights(x, _y, logistic=True)

    pred = log_res['model_prediction']
    pred = np.argmax(pred, axis=-1)
    print("correct: ", np.sum(pred == y))
    print("wrong: ", np.sum(pred != y))
    print("accuracy: ", np.sum(pred == y) / num_samples)
    print(y.shape, pred.shape, )

    plt.figure()
    # to_plot = np.sum(log_res['sign_norm_rwa'], axis=-1)
    plt.subplot(1, 2, 1)
    plt.title("pred v target")
    spread = list(range(len(x)))
    diff = np.abs(y - pred)
    noise = np.random.uniform(0.5, 1, size=num_samples)
    diff = ((diff * noise) / num_classes * num_samples)
    for idx in range(num_samples):
        sign = np.random.choice([-1, 1])
        plt.scatter(spread[idx], (spread[idx] + sign*diff[idx]), alpha=0.1, color="blue")
    # plt.scatter(spread, spread, alpha=0.2, color="orange")
    plt.show()

    plt.figure()
    plt.title("Weights & Betas - multiclass")
    coef = np.sum(coef, axis=-1)
    rows = 3
    cols = int(np.ceil(num_classes / rows))
    for class_idx in range(num_classes):
        plt.subplot(rows, cols, class_idx+ 1)
        plt.title(f"class {class_idx}")
        to_plot = coef / np.max(coef)
        plt.barh(range(len(to_plot)), np.abs(to_plot), alpha=0.75, color="blue")

        to_plot = log_res['norm_rwa'][:, class_idx]
        plt.barh(range(len(to_plot)), to_plot, alpha=0.5, color="orange")

    plt.legend(["data_beta", "RWA"])
    plt.tight_layout()
    plt.show()

    # ========= Multiclass, Multilabel dataset ========
    num_classes = 5
    x, y, meta = gen.generate(task="multilabel",
                              num_samples=num_samples,
                              num_classes=num_classes,
                              num_features=num_features,
                              noise_scale=NOISE)

    coef = np.abs(meta["weights"])
    # y is already one-hot since it's multilabel.
    _y = y
    print(_y.shape)
    log_res = relative_weights(x, _y, logistic=True)

    pred = log_res['model_prediction'].round()

    print("correct: ", np.sum(pred == y))
    print("wrong: ", np.sum(pred != y))
    print("accuracy: ", np.sum(pred == y) / (num_samples * num_classes))
    print(y.shape, pred.shape, )

    plt.figure()
    # to_plot = np.sum(log_res['sign_norm_rwa'], axis=-1)
    # plt.subplot(1, 2, 1)
    plt.title("pred v target")
    spread = list(range(len(x)))
    diff = np.abs(y - pred)
    noise = np.random.uniform(0.5, 1, size=(num_samples, num_classes))
    diff = ((diff * noise) / num_classes * num_samples)
    for class_idx in range(num_classes):
        for idx in range(num_samples):
            sign = np.random.choice([-1, 1])
            plt.scatter(spread[idx], (spread[idx] + sign * diff[idx, class_idx]), alpha=0.05)
    # plt.scatter(spread, spread, alpha=0.2, color="orange")
    plt.show()

    plt.figure()
    plt.title("Weights & Betas - multilabel class")
    rows = 3
    coef = coef.sum(axis=-1)
    cols = int(np.ceil(num_classes / rows))
    for class_idx in range(num_classes):
        plt.subplot(rows, cols, class_idx + 1)
        plt.title(f"class {class_idx}")
        to_plot = coef / np.max(coef)
        plt.barh(range(len(to_plot)), np.abs(to_plot), alpha=0.75, color="blue")

        to_plot = log_res['norm_rwa'][:, class_idx]
        plt.barh(range(len(to_plot)), to_plot,  alpha=0.5, color="orange")

    plt.legend(["data_beta", "RWA"])
    plt.tight_layout()
    plt.show()
