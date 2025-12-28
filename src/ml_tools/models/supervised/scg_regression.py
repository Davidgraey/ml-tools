'''
Regression: Gradient Descent with Scaled Conjugate Gradient
SCG lets us avoid hyperparameter optimization step; makes it much more 'plug n play'
Scaled Conjugate Gradient adapted from implementation by Prof Charles
Anderson, CSU

Including regularization via elasticnet to extend usability a bit
'''
import numpy as np
import copy
from numpy.typing import NDArray
from ml_tools.models.constants import EPSILON, SIGMA_ZERO, LAMBDA_MAX, LAMBDA_MIN
from ml_tools.generators.data_generators import to_onehot, to_int_classes, RandomDatasetGenerator
from typing import Optional
from ml_tools.models.constants import ClassificationTask, determine_classification_task
from ml_tools.models.model_loss import cross_entropy, cross_entropy_derivative, mse, mse_derivative
from ml_tools.models.activations import (
    softmax_activation,
    sigmoid_activation,
    linear_activation
)
from ml_tools.models.supervised import log
from ml_tools.types import BasalModel


class GradientDescent(BasalModel):
    def __init__(self,
                 task: str|ClassificationTask = "regression",
                 divisi: int = 3,
                 reg_lambda: float = 0.1,
                 reg_alpha: float = 0.5,
                 use_elastic_reg: bool = False,
                 early_termination: bool = True):
        """

        Parameters
        ----------
        task : Union[str, constants.ClassificationTask]
        divisi : int the divisor to slow down learning rate - higher = slower
        reg_lambda : float  overall regularization strength
        reg_alpha : float balance between lasso L1 (0.0) ridge L2 (1.0)
        use_elastic_reg : bool enable elastic regularization
        """
        super().__init__(seed=42)
        self.full_init = False
        self.early_termination = early_termination

        # hyperparameters for elasticnet regularization
        self.divisi = divisi
        self.use_elastic_reg = use_elastic_reg
        self.reg_lambda: float = reg_lambda # overall regularization strength
        self.reg_alpha: float = reg_alpha  # balance between lasso L1 (0.0) ridge L2 (1.0)

        self.task: ClassificationTask = task
        if task == "regression":
            self.func = {"activation": linear_activation,
                         "loss": mse,
                         "loss_derivative": mse_derivative}

        elif task == ClassificationTask.BINARY:
            self.func = {"activation": sigmoid_activation,
                         "loss": cross_entropy,
                         "loss_derivative": cross_entropy_derivative}

        elif task == ClassificationTask.MULTINOMIAL:
            self.func = {"activation": softmax_activation,
                         "loss": cross_entropy,
                         "loss_derivative": cross_entropy_derivative}

        elif task == ClassificationTask.MULTILABEL:
            self.func = {"activation": sigmoid_activation,
                         "loss": cross_entropy,
                         "loss_derivative": cross_entropy_derivative}
        else:
            raise ValueError(f"task must be one of regression, binary, multinomial, multilabel; got {task}")

    def init_weights(self,  n_outputs: int = 1) -> None:
        """
        Initalize the beta values (coefficients / weights) using Kaiming mechanism
        Parameters
        ----------
        n_outputs : the number of output dimensions

        Returns
        -------
        None
        """
        # init with kaiming:
        ws = self.RNG.normal(
            loc=0.0,
            scale=np.sqrt(2 / self.input_dimension),
            size=(self.input_dimension + 1, n_outputs)
        )

        # add zero for has_bias_presentbias / intercept value
        ws[0, :] = 0
        # log.info(f'kaiming init built {ws.shape} coefficients')
        self.weights = ws

    def standardize(self,
                    data_array: NDArray,
                    mean: NDArray,
                    stds: NDArray
                    ) -> NDArray:
        """ standardize our data to 0 mean 1 std"""
        assert data_array.shape[-1] == mean.shape[0]

        return (data_array - mean) / (stds + EPSILON)

    def unstandardize(self,
                      data_array: NDArray,
                      mean: NDArray,
                      stds: NDArray
                      ) -> NDArray:
        """ reverse the standardize process - restore the original data space """
        assert data_array.shape[-1] == mean.shape[0]

        return (stds - EPSILON) * data_array + mean

    def init_standardize(self, x_data: NDArray, y_data: Optional[NDArray]) -> None:
        """ update the tracking means and standards """
        self.x_means = np.mean(x_data, axis=0)
        self.x_stds = np.std(x_data, axis=0)

        if self.task == "regression":
            self.y_means = np.mean(y_data, axis=0)
            self.y_stds = np.std(y_data, axis=0)

    def _add_intercept(self, data_array: NDArray):
        """ insert an intercept at 0th index """
        return np.insert(data_array, 0, 1, -1)

    def log_likelihood(self, prediction: NDArray, targets: NDArray) -> NDArray | float:
        """
        Calc the LLH given the prediction in prob space, and our targets. (classification tasks only)
        Parameters
        ----------
        prediction : prediction with activation applied (probability values)
        targets : y data

        Returns
        -------
        log likelihood, float
        """
        _p = np.clip(prediction, EPSILON, 1 - EPSILON)
        if self.task == ClassificationTask.MULTINOMIAL:
            llh = np.mean(np.sum(targets * np.log(_p), axis=-1))
        elif self.task in (ClassificationTask.BINARY, ClassificationTask.MULTILABEL):
            llh = np.mean(
                np.sum(
                    targets * np.log(_p) + (1 - targets) * np.log(1 - prediction + EPSILON),
                    axis=-1
                )
            )
        return llh

    def _calculate_pseudo_r_squared(self, prediction: NDArray, targets: NDArray) -> tuple[float, float]:
        """ doing McFadden's R-squared for classificaiton """
        # LLH of the model
        log_likelihood = self.log_likelihood(prediction, targets)

        # Log-likelihood of the null model (using the mean of y as intercept)
        p_null = np.mean(targets, axis=0)
        null_log_likelihood = self.log_likelihood(p_null, targets)

        # McFadden's R-squared
        self.r_square = 1 - (log_likelihood / null_log_likelihood)

        # no standardized adjusted version; so let's just... -1
        self.adjusted_r_square = -1

        return self.r_square, self.adjusted_r_square

    def _calculate_r_square(self, prediction: NDArray, targets: NDArray) -> tuple[float, float]:
        if self.task == "regression":
            self.residuals = prediction - targets
            residual_sum_squares = np.sum(self.residuals ** 2)
            total_sum_squares = np.sum((targets - np.mean(targets)) ** 2)
            self.r_square = 1 - (residual_sum_squares / total_sum_squares)
            self.adjusted_r_square = 1- (
                ((1 - self.r_square) * (self.num_samples - 1))
                / (self.num_samples - self.input_dimension - 1))

        # ------- classification----- using mcfadden's R2 and no adjusted mechanism
        else:
            self.r_square, self.adjusted_r_square = self._calculate_pseudo_r_squared(prediction, targets)

        return self.r_square, self.adjusted_r_square

    def scaled_conjugate_gradient(self,
                                  x_data: NDArray,
                                  y_data: NDArray,
                                  iterations: int
                                  ) -> NDArray:
        """
        This is the SCG multi-step search; 2nd order derivatives lead us to a step that is less sensitive to local
        minima.  However, it's expensive to compute, and can overfit/converge too quickly; so we alternate between SCG
        and gradient descent.  It shares some stages with the Matlab implementation

        Parameters
        ----------
        x_data : numpy.ndarray
        y_data : numpy.ndarray
        iterations : int - number of substeps to take for this SCG update

        Returns
        -------
        numpy.ndarray
        """
        lamb = 1e-6
        lamb_ = 0

        vector = self.get_weights().reshape(-1, 1)
        grad_new, _ = self._calculate_gradients(x_data, y_data)
        grad_new = -1 * grad_new.reshape(-1, 1)
        r_new = grad_new.copy()
        success = True

        for _i in range(iterations):
            r = r_new.copy()
            grad = grad_new.copy()
            mu = grad.T @ grad

            if success:
                success = False
                sigma = SIGMA_ZERO / np.sqrt(mu)

                grad_old, _ = self._calculate_gradients(x_data, y_data)
                grad_old = grad_old.reshape(-1, 1)
                self.weights = (vector + (sigma * grad)).reshape(self._weight_shape)
                grad_step, _ = self._calculate_gradients(x_data, y_data)

                step = (grad_old - grad_step.reshape(-1, 1)) / sigma
                delta = grad.T @ step

            # increase the curvature / scale
            zeta = lamb - lamb_
            step += zeta * grad
            delta += zeta * mu

            if delta <= 0:
                step += (lamb - 2 * delta / mu) * grad
                lamb_ = 2 * (lamb - delta / mu)
                delta -= lamb * mu
                delta *= -1
                lamb = lamb_

            phi = grad.T @ r
            alpha = phi / delta

            vector_new = vector + alpha * grad
            loss_old = self.calculate_loss(x_data, y_data)
            self.weights = vector_new.copy().reshape(self._weight_shape)
            loss_new = self.calculate_loss(x_data, y_data)

            comparison = 2 * delta * (loss_old - loss_new) / (phi ** 2)

            if comparison >= 0:
                # break condition?
                vector = vector_new.copy()
                loss_old = loss_new
                self.weights = vector_new.copy().reshape(self._weight_shape)
                r_new, _ = self._calculate_gradients(x_data, y_data)
                r_new = -1 * r_new.reshape(-1, 1)
                success = True
                lamb_ = 0

                if _i % self._weight_shape[0] == 0:
                    grad_new = r_new
                else:
                    beta = ((r_new.T @ r_new) - (r_new.T @ r)) / phi
                    grad_new = r_new + beta * grad

                if comparison > 0.75:
                    lamb = max(0.5 * lamb, LAMBDA_MIN)
            else:
                lamb_ = lamb

            if comparison < 0.25:
                lamb = min(4 * lamb, LAMBDA_MAX)

        return vector_new.reshape(self._weight_shape)

    def fit(self,
            x_data: NDArray,
            y_data: NDArray,
            iterations: int = 100,
            add_constant: bool = True
            ):

        self.num_samples = x_data.shape[0]
        self.num_outputs = y_data.shape[-1]

        # if we've never trained before, we need to take additional init steps
        self.input_dimension = x_data.shape[-1]

        if not self.full_init:
            self.init_weights(n_outputs=self.num_outputs)
            self.init_standardize(x_data=x_data, y_data=y_data)
            self.full_init = True

        # standardize our data ------------------------------------------
        xs = self.standardize(x_data, self.x_means, self.x_stds)
        if self.task == "regression":
            ys = self.standardize(y_data, self.y_means, self.y_stds)
        else:
            ys = y_data.copy()

        # if we need to add intercept ------------------------------------
        if add_constant:
            xs = self._add_intercept(data_array=xs)
        else:
            xs = xs.copy()

        # training loop ------------------------------------------------
        errors = []
        for i in range(iterations):
            shuffler = self.RNG.permutation(self.num_samples)
            xs = xs[shuffler]
            ys = ys[shuffler]

            loss = self.calculate_loss(xs, ys)
            # deepcopy - so we can exploit the SCG steps
            W = copy.deepcopy(self.weights)

            # weight updates -------------------------------------------------
            if i % 2 == 0:
                # Scaled Conjugate Gradient - take a partial step towards the SCG results
                # between current weights and SCG weights
                scg_weights = self.scaled_conjugate_gradient(xs, ys, iterations // 2)
                self.weights = (W + scg_weights) / (self.divisi)

            else:
                # standard gradient descent approach ---------
                delta_ws, prediction = self._calculate_gradients(xs, ys)
                self.weights = W - (delta_ws * (0.001/self.divisi))

            epoch_loss = np.mean(loss)
            errors.append(epoch_loss)

            if i <= 5:
                continue
            if self.early_termination:
                loss_delta = (np.array(errors[-min(i, 5):]) - epoch_loss)
                loss_new = np.mean(self.calculate_loss(xs, ys))
                if (np.mean(loss_delta) < 0.01) or ((epoch_loss - loss_new) >= 0):
                    self.divisi += 1

                if (np.sum(loss_delta < 0) > 10):
                    print(f"early termination at {i} with error {epoch_loss}")
                    log.info(f"early termination at {i} with error {epoch_loss}")
                    break
                # not elif -- if cond
                if np.mean(loss_delta) < 1e-12:
                    print(f"early termination, loss stopped decreasing at {i}")
                    log.info(f"early termination, loss stopped decreasing at {i}")
                    break

        r_squared = self._calculate_r_square(self.predict(x_data), y_data)
        log.info(f"fitted r_squared: {r_squared}")
        log.info(f"divisi {self.divisi} ")
        return errors

    def fit_predict(self,
                    x_data: NDArray,
                    y_data: NDArray,
                    iterations: int = 100,
                    add_constant: bool = True ):
        _error = self.fit(x_data, y_data, iterations, add_constant)
        return self.predict(x_data)

    def forward(self, x_data: NDArray, has_bias_present=True) -> NDArray:
        """
        If we already introduced bias into the X data (training), or if we need to add it as in prediction

        Parameters
        ----------
        x_data : numpy.ndarray
        has_bias_present : bool if we have the bias / intercept already present in our x_data array

        Returns
        -------
        numpy.ndarray
        """
        if has_bias_present:
            logit = x_data @ self.weights

        else:  # add intercept to our data
            x_data = self._add_intercept(x_data)
            logit = x_data @ self.weights

        return logit

    def _calculate_gradients(self, x_data: NDArray, y_data: NDArray) -> tuple[NDArray, NDArray]:
        """ calculate gradients for weight updates"""
        logits = self.forward(x_data)
        prediction = self.func["activation"](logits)
        delta_grad = self.func["loss_derivative"](targets=y_data, prediction=prediction, task=self.task)

        if self.use_elastic_reg:
            reg_gradient = self.elasticnet_grad()
            gradient = ((x_data.T @ (delta_grad)) + reg_gradient)
        else:
            gradient = (x_data.T @ (delta_grad))

        return gradient, prediction

    def calculate_loss(self, xs: NDArray, ys: NDArray) -> float:
        """ calculate loss for current weights and inputs """
        logits = self.forward(xs)
        if self.use_elastic_reg:
            regularization_loss = self.elasticnet_loss()
        else:
            regularization_loss = 0

        return self.func["loss"](prediction=logits, targets=ys, task=self.task, reduction="mean") + regularization_loss

    def predict(self, x_data: NDArray) -> NDArray:
        """ forward pass, with final activation in place and the unstandardization (if we have a regression task)"""
        _x = self.standardize(x_data, self.x_means, self.x_stds)
        prediction = self.forward(_x, has_bias_present=False)

        if self.task == "regression":
            prediction =  self.unstandardize(prediction, self.y_means, self.y_stds)
        else:
            prediction =  prediction

        return self.func["activation"](prediction)

    def elasticnet_loss(self) -> NDArray:
        """ implementing elasticnet factor on loss """
        lasso_penalty = np.sum(np.abs(self.weights))
        ridge_penalty = np.sum(self.weights ** 2)
        reg_loss = (
            self.reg_lambda * (
            (self.reg_alpha * lasso_penalty)
            + ((1 - self.reg_alpha) * ridge_penalty)
            )
        )
        return reg_loss

    def elasticnet_grad(self) -> NDArray:
        """ implementing elasticnet factor on gradients """
        reg_grad = self.reg_lambda * (
            (self.reg_alpha * np.sign(self.weights))
            + ((1 - self.reg_alpha) * 2*self.weights)
        )
        return reg_grad

    def get_residuals(self):
        """ get the residuals from the last fit """
        return self.residuals

    def get_weights(self):
        """ return a copy of the weights """
        self._weight_shape = self.weights.shape
        return copy.deepcopy(self.weights)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    num_samples = 2000
    num_features = 8
    NOISE = 0.33
    N_steps = 25

    # ============================ REGRESSION ==============================
    print("============ REGRESSION ============")
    generator = RandomDatasetGenerator(random_seed=44)
    x, y, meta = generator.generate(task="regression",
                                    num_samples=num_samples,
                                    num_features=num_features,
                                    noise_scale=NOISE
                                    )
    coef = meta["weights"]
    ws = []
    for reg in [False, True]:
        model = GradientDescent(task='regression',
                                use_elastic_reg=reg,
                                early_termination=True)
        print(x.shape, y.shape)
        errors = model.fit(x_data=x,
                           y_data=y.reshape(-1, 1),
                           iterations=N_steps)
        print(model.use_elastic_reg)
        print("R2: ", model.r_square, model.adjusted_r_square)


        plt.title("regression loss")
        plt.subplot(2, 1, 1)
        plt.plot(errors)
        plt.subplot(2, 1, 2)
        plt.plot(errors[-20:])
        plt.show()

        pred = model.predict(x)

        plt.figure(figsize=(7, 10))
        plt.title(f"regression info for {reg}")
        ax1 = plt.subplot(2, 3, 1, label="ys")
        plt.plot(y[:], 'r')
        ax1.set_title("targets")

        ax2 = plt.subplot(2, 3, 2, label="both")
        ax2.set_title("pred/targets")
        plt.plot(y, alpha=0.8, c='r')
        plt.plot(pred, alpha=0.8, c='b')

        ax3 = plt.subplot(2, 3, 3, label="prediction")
        ax3.set_title("prediction")
        plt.plot(pred[:], 'b')

        ax4 = plt.subplot(2, 3, 4)
        ax4.set_title("Target coefficients")
        _xs = np.arange(num_features + 1)
        _coef = np.insert(coef, 0, meta["bias"], axis=0)
        plt.scatter(_xs, _coef, label="ceoffs")
        ax5 = plt.subplot(2, 3, 5)
        ax5.set_title("coeff - Weights DIFF")
        plt.plot(_xs, np.abs(_coef - model.weights.squeeze()), label="diff", alpha=0.66, c="r")
        plt.scatter(_xs, _coef.T, label="T Betas", alpha=0.66, c="y")
        plt.scatter(_xs, model.weights.squeeze(), label="T Betas", alpha=0.66, c="b")

        ax6 = plt.subplot(2, 3, 6)
        ax6.set_title("Model Weights")
        plt.scatter(_xs, model.weights.T, label="weights", c="b")

        plt.show()

    # ============================ CLASSIFICATION ==============================
    print("============ BINARY CLASSIFICATION ============")
    x, y, meta = generator.generate(task="binary",
                                    num_samples=num_samples,
                                    num_features=num_features,
                                    num_classes=2,
                                    noise_scale=NOISE
                                    )

    model = GradientDescent(task=ClassificationTask.BINARY,
                            use_elastic_reg=False,
                            early_termination=True)
    _y = to_onehot(y)
    errors = model.fit(x_data=x,
                       y_data=_y,
                       iterations=N_steps)

    coef = meta["weights"]
    plt.title("binary classification loss")
    plt.subplot(2, 1, 1)
    plt.plot(errors)
    plt.subplot(2, 1, 2)
    plt.plot(errors[-20:])
    plt.show()

    pred = model.predict(x)
    pred = np.argmax(pred, axis=-1)

    print("correct: ", np.sum(pred == y))
    print("wrong: ", np.sum(pred != y))
    print(np.unique(y, return_counts=True)[-1])
    print("accuracy: ", np.sum(pred == y) / num_samples)
    print("R2: ", model.r_square, model.adjusted_r_square)

    plt.figure(figsize=(7, 10))
    _xs = np.arange(num_samples)
    plt.title(f"Model details for binary classification")
    # to_int_classes
    ax1 = plt.subplot(2, 3, 1, label="targets")
    plt.scatter(_xs, y, c="r")
    ax1.set_title("targets")
    ax2 = plt.subplot(2, 3, 2, label="both")
    ax2.set_title("pred/targets")
    plt.scatter(_xs, y, alpha=0.66, c='r')
    plt.scatter(_xs, pred, alpha=0.33, c='b')
    ax3 = plt.subplot(2, 3, 3, label="prediction")
    ax3.set_title("prediction")
    plt.scatter(_xs, pred[:], c='b')
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title("Target coefficients")
    _xs = np.arange(num_features + 1)
    _coef = np.insert(coef, 0, meta["bias"], axis=0)
    plt.scatter(_xs, _coef.T, label="T Betas", c="y")
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title("Beta - Weights DIFF")
    to_plot = np.abs(model.weights).sum(axis=-1)
    plt.plot(_xs, np.abs(_coef - to_plot), label="diff", alpha=0.66, c="r")
    plt.scatter(_xs, _coef, label="T Betas", alpha=0.66, c="y")
    plt.scatter(_xs, to_plot, label="T Betas", alpha=0.66, c="b")
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title("Model Weights")
    plt.scatter(_xs, to_plot, label="weights", c="b")

    plt.show()


    # ======================= multinomial CLASSIFICATION ===========================
    print("============ MULTINOMIAL CLASSIFICATION ============")
    x, y, meta = generator.generate(task="multiclass",
                                    num_samples=num_samples,
                                    num_features=num_features,
                                    num_classes=6,
                                    noise_scale=NOISE,
                                    )

    model = GradientDescent(task=ClassificationTask.MULTINOMIAL,
                            use_elastic_reg=False,
                            early_termination=False)
    _y = to_onehot(y)
    errors = model.fit(x_data=x,
                       y_data=_y,
                       iterations=N_steps)

    coef = meta["weights"]
    plt.title("multinomial classification loss")
    plt.subplot(2, 1, 1)
    plt.plot(errors)
    plt.subplot(2, 1, 2)
    plt.plot(errors[-20:])
    plt.show()

    pred = model.predict(x)
    pred = np.argmax(pred, axis=-1)
    print("correct: ", np.sum(pred == y))
    print("wrong: ", np.sum(pred != y))
    print("accuracy: ", np.sum(pred == y) / num_samples)
    print("R2: ", model.r_square, model.adjusted_r_square)

    # sanity_check = LogisticRegression(penalty="elasticnet", max_iter = N_steps, )
    # sanity_check.fit(x, y)
    # sanity_y = sanity_check.predict(x)

    plt.figure(figsize=(7, 10))
    _xs = np.arange(num_samples)
    plt.title(f"Model details for multinomial classification")
    # to_int_classes
    ax1 = plt.subplot(2, 3, 1, label="ys")
    plt.scatter(_xs, y, c="r")
    ax1.set_title("targets")
    ax2 = plt.subplot(2, 3, 2, label="both")
    ax2.set_title("pred/targets")
    plt.scatter(_xs, y, alpha=0.66, c='r')
    plt.scatter(_xs, pred, alpha=0.33, c='b')
    ax3 = plt.subplot(2, 3, 3, label="prediction")
    ax3.set_title("prediction")
    plt.scatter(_xs, pred, c='b')
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title("Target coefficients")
    _xs = np.arange(num_features + 1)
    _coef = np.insert(coef, 0, meta["bias"], axis=0)
    _coef = np.sum(_coef, axis=-1)
    ws = np.sum(model.weights, axis=-1)
    plt.scatter(_xs, _coef, label="T Betas", c="y")
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title("Beta - Weights DIFF")
    plt.plot(_xs, np.abs(_coef - ws), label="diff", alpha=0.66, c="r")
    plt.scatter(_xs, _coef, label="T Betas", alpha=0.66, c="y")
    plt.scatter(_xs, ws, label="T Betas", alpha=0.66, c="b")
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title("Model Weights")
    plt.scatter(_xs, ws, label="weights", c="b")

    plt.show()


    # ============================ MULTILABEL CLASSIFICATION ==============================
    print("============ MULTILABEL CLASSIFICATION ============")
    num_classes = 5
    x, y, meta = generator.generate(task="multilabel",
                                    num_samples=num_samples,
                                    num_features=num_features,
                                    num_classes=num_classes,
                                    noise_scale=NOISE
                                    )

    model = GradientDescent(task=ClassificationTask.MULTILABEL,
                            use_elastic_reg=False,
                            early_termination=True)

    print(x.shape, y.shape)
    errors = model.fit(x_data=x,
                       y_data=y,
                       iterations=N_steps
                       )

    coef = meta["weights"]
    plt.title("multilabel classification loss")
    plt.subplot(2, 1, 1)
    plt.plot(errors)
    plt.subplot(2, 1, 2)
    plt.plot(errors[-20:])
    plt.show()

    pred = model.predict(x)
    pred = np.where(pred > 0.5, 1, 0)
    print("correct: ", np.sum(pred == y))
    print("wrong: ", np.sum(pred != y))
    print("accuracy: ", np.sum(pred == y) / (num_samples*num_classes))
    print("R2: ", model.r_square, model.adjusted_r_square)

    plt.figure(figsize=(7, 10))
    _xs = np.tile(np.arange(num_samples), num_classes).reshape(num_samples, -1)

    plt.title(f"Model details for multinomial, multilabel classification")
    # to_int_classes
    plotable_y = y * np.arange(0, num_classes)
    plotable_pred = pred * np.arange(0, num_classes)
    ax1 = plt.subplot(2, 3, 1, label="targets")
    plt.scatter(_xs, plotable_y, c="r")
    ax1.set_title("targets")
    ax2 = plt.subplot(2, 3, 2, label="both")
    ax2.set_title("pred/targets")
    plt.scatter(_xs, plotable_y, alpha=0.66, c='r')
    plt.scatter(_xs, plotable_pred, alpha=0.33, c='b')
    ax3 = plt.subplot(2, 3, 3, label="prediction")
    ax3.set_title("prediction")
    plt.scatter(_xs, plotable_pred, c='b')
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title("Target coefficients")
    _xs = np.arange(num_features + 1)
    _coef = np.sum(np.insert(coef, 0, meta["bias"], axis=0), axis=-1)
    ws = np.sum(model.weights, axis=-1)
    plt.scatter(_xs, _coef.T, label="T Betas", c="y")
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title("Beta - Weights DIFF")
    plt.plot(_xs, np.abs(_coef - ws), label="diff", alpha=0.66, c="r")
    plt.scatter(_xs, _coef, label="T Betas", alpha=0.66, c="y")
    plt.scatter(_xs, ws, label="T Betas", alpha=0.66, c="b")
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title("Model Weights")
    plt.scatter(_xs, ws, label="weights", c="b")

    plt.legend()
    plt.show()
