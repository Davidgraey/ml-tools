import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional, Tuple
from ml_tools.types import BasalModel
from ml_tools.models.activations import sigmoid, softmax
import copy
from enum import Enum


class CalibrationType(Enum):
    platt = "platt"
    isotonic = "isotonic"
    spline = "spline"


class ProbCalibration(BasalModel):
    """
    Probability calibration - takes in logits as a prediction, and, using CalibrationType methods, projects those
    logist into a calibrated probability space

    Platt Scaling fits a sigmoid: P(y=1|f) = 1 / (1 + exp(A*f + B))
    Isotonic Regression (PAVA) fits a non-decreasing step function.
    Temperature Scaling divides logits by learned T: P = sigmoid(f/T) or softmax(f/T)
    """

    def __init__(self, data_dimension: int, method: CalibrationType = "platt"):

        self.input_dimension: int = data_dimension
        self.output_dimension: int = data_dimension

        self.method = CalibrationType(method)
        # Platt parameters
        self.platt_coefficient = None
        self.platt_intercept = None

        # Isotonic parameters
        self.isotonic_scores: Optional[NDArray] = None
        self.isotonic_values: Optional[NDArray] = None

        # Spline parameters
        self.spline_x = None
        self.spline_y = None
        self.spline_d = None

        self._is_fitted: bool = False

    def fit(self, logit_data: NDArray, y_data: NDArray):
        """
        Fit the calibration model-- logits (x) are compared to the actual labels(y)
        # TODO: insure one-hot shape matching.

        Parameters
        ----------
        logit_data : array-like of shape (n_samples, input_dimension)

        y_true : array-like of shape (n_samples, input_dimension) - ground truth labels

        Returns
        -------

        """
        x_data = np.asarray(logit_data)
        y_data = np.asarray(y_data)

        assert x_data.shape == y_data.shape, "you fucked it up, shapes must match"


        if self.method == CalibrationType.platt:
            fitted = self._fit_platt(x_data, y_data)
        elif self.method == CalibrationType.isotonic:
            fitted = self._fit_isotonic(x_data, y_data)
        elif self.method == CalibrationType.spline:
            fitted = self._fit_monotone_spline(scores=x_data, labels=y_data)
        else:
            raise ValueError(f"Unknown method: {self.method}.")

        self._is_fitted = True
        return fitted

    def forward(self, x_data: NDArray, **kwargs):
        pass

    def calculate_loss(self, x_data: NDArray, y_data, **kwargs):
        return y_data - x_data

    def predict(self, logits: NDArray) -> NDArray:
        """
        Apply calibration to raw scores.

        Parameters
        ----------
        logits : array-like of shape (n_samples, num_classes)
            Raw predicted scores, logits or uncalibrated probabilities.

        Returns
        -------
        calibrated : ndarray of shape (n_samples, num_classes)
            Calibrated probabilities .
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        logits = np.asarray(logits)

        if self.method == CalibrationType.platt:
            return self._platt_forward(logits)
        elif self.method == CalibrationType.isotonic:
            return self._predict_isotonic(logits)
        elif self.method == CalibrationType.spline:
            return self._predict_spline(logits)

    def fit_predict(self, y_score: NDArray, y_true: NDArray) -> NDArray:
        """Fit and transform in one step."""
        return self.fit(y_score, y_true).predict(y_score)

    # Platt Scaling # ------------------------------------------------------------------------
    def _platt_forward(self, logits: NDArray) -> NDArray:
        z = (self.platt_coefficient * logits) + self.platt_intercept
        return sigmoid(z)

    def _fit_platt(self,
                   logits: NDArray,
                   labels: NDArray,
                   max_iter: int = 100,
                   tolerance: float = 1e-25,
                   ) -> BasalModel:
        """
        Platt scaling via Gauss-Newtonian (L-Sum-of-Squares) / line search

        Minimizes the negative log-likelihood:
        derived from label smoothing (Platt's original formulation).

        sigmoid(A*s +B)

        Parameters
        ----------
        logits -
        labels
        max_iter
        tolerance

        Returns
        -------
        A, B : float
            Fitted sigmoid parameters.
        """
        num_samples = len(logits)

        # n_pos = positive counts per label (if multiclass)
        n_pos = np.sum(labels == 1, axis=0)
        # n_neg = negative counts per label (if multiclass)
        n_neg = num_samples - n_pos

        # Target probabilities (Platt's label smoothing)
        targets = np.where(labels == 1,
                           (n_pos + 1) / (n_pos + 2),
                           1.0 / (n_neg + 2))

        # Initialize A, B
        self.platt_coefficient = 1.0
        self.platt_intercept = np.log((n_neg + 1) / (n_pos + 1)) # valid init?

        for step_i in range(max_iter):
            # Compute the forward
            prob_scores = self._platt_forward(logits)

            loss = prob_scores - targets  # shape (n,)
            # gradients
            delta_a = logits.T @ logits
            delta_b = np.sum(loss)

            # hessian diagonal approximation: H = p*(1-p))
            w = prob_scores * (1.0 - prob_scores)  # weights
            _h11 = w.T @ (logits ** 2)
            _h22 = np.sum(w)
            _h12 = w.T @ logits

            # 2x2 determinate
            determinate = (_h11 * _h22) - (_h12 * _h12)
            if np.any(np.abs(determinate)) < 1e-15: # TODO: EPSILON
                print('nope')
                break

            # Newton step (solve H * delta = -grad)
            delta_a = -((_h22 * delta_a - _h12 * delta_b) / determinate).ravel()
            delta_b = -((_h11 * delta_b - _h12 * delta_a) / determinate).ravel()

            # Line search with halving
            step = 1.0
            old_nll = self._nll(scores=logits,
                                targets=targets,
                                coef=self.platt_coefficient,
                                intercept=self.platt_intercept)

            new_coef = copy.copy(self.platt_coefficient)
            new_intercept = copy.copy(self.platt_intercept)

            # N-steps down this line---
            for _ in range(10):
                new_coef = new_coef + (step * delta_a)
                new_intercept = new_intercept + (step * delta_b)
                new_nll = self._nll(scores=logits,
                                    targets=targets,
                                    coef=new_coef,
                                    intercept=new_intercept)

                # reduce delta by half
                step *= 0.5
                if new_nll < old_nll:
                    break


            # adjust weights
            self.platt_coefficient -= (step * delta_a)
            self.platt_intercept -= (step * delta_b)

            # Convergence check
            if (abs(step * delta_a) < tolerance) and (abs(step * delta_b) < tolerance):
                print(f'halted at convergence {step_i}')
                break

        return self

    def _nll(self, scores: NDArray, targets: NDArray, coef: Optional = None, intercept: Optional = None) -> float:
        """Negative log-likelihood for Platt scaling."""
        if (coef is not None) and (intercept is not None):
            p = sigmoid(coef * scores + intercept)
        else:
            p = sigmoid(scores)
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        return -np.sum(targets * np.log(p) + (1 - targets) * np.log(1 - p))


    # Isotonic Regression (PAVA) # ──────────────────────────────────────────────

    @staticmethod
    def _pava(y: NDArray) -> NDArray:
        """
        Pool Adjacent Violators Algorithm.

        Finds the isotonic (non-decreasing) regression of y that minimizes
        the weighted least squares: sum w_i * (y_i - y_hat_i)^2
        subject to y_hat being non-decreasing.

        Parameters
        ----------
        y : ndarray, shape (n,)
            Input values.

        Returns
        -------
        result : ndarray, shape (n,)
            Isotonic (non-decreasing) fitted values.
        """
        n = len(y)
        if n <= 1:
            return y.copy()

        # each block: [value_sum, weight, start_index, end_index]
        result = y.astype(np.float64).copy()

        all_blocks = []

        for i in range(n):
            all_blocks.append([result[i], 1.0, i, i])

            # pool adjacent violators: merge with previous block if it violates the monotonic
            while len(all_blocks) > 1:
                curr = all_blocks[-1]
                prev = all_blocks[-2]

                # check monotonicity: prev_mean <= curr_mean
                prev_mean = prev[0] / prev[1]
                curr_mean = curr[0] / curr[1]

                if prev_mean > curr_mean:
                    prev[0] += curr[0]  # sum values
                    prev[1] += curr[1]  # sum weights
                    prev[3] = curr[3]  # extend end index
                    all_blocks.pop()
                else:
                    break

        # Reconstruct result from blocks
        for block in all_blocks:
            mean_val = block[0] / block[1]
            start = int(block[2])
            end = int(block[3])
            result[start: end + 1] = mean_val

        return result

    def _fit_isotonic(self,
                      logits: NDArray,
                      labels: NDArray,
                      ) -> BasalModel:
        """
        Fit isotonic regression using the Pool Adjacent Violators Algorithm (PAVA).

        Produces a non-decreasing mapping from scores to calibrated probabilities.
        Collapses each constant-value PAVA block into a single knot at the block's
        centroid score, enabling smooth linear interpolation between steps.

        Parameters
        ----------
        logits : ndarray, shape (n_samples,) or (n_samples, 1)
        labels : ndarray, shape (n_samples,) or (n_samples, 1) with values in {0, 1}

        Returns
        -------
        self
        """
        # Flatten to 1D for robust processing
        scores_flat = logits.ravel()
        labels_flat = labels.ravel()
        num_samples = len(scores_flat)

        # Sort by score
        order = np.argsort(scores_flat)
        sorted_scores = scores_flat[order]
        sorted_labels = labels_flat[order]

        # Run PAVA on 1D sorted labels → non-decreasing step function
        isotonic_values = self._pava(sorted_labels)

        # Collapse to one knot per PAVA block (centroid of scores in each block)
        # Identify block boundaries: where the isotonic value changes
        change_mask = np.diff(isotonic_values) != 0
        change_points = np.where(change_mask)[0] + 1

        block_starts = np.concatenate([[0], change_points])
        block_ends = np.concatenate([change_points, [num_samples]])

        n_blocks = len(block_starts)
        self.isotonic_scores = np.empty(n_blocks, dtype=np.float64)
        self.isotonic_values = np.empty(n_blocks, dtype=np.float64)

        for i, (start, end) in enumerate(zip(block_starts, block_ends)):
            # Centroid score for this block → representative x position
            self.isotonic_scores[i] = np.mean(sorted_scores[start:end])
            # Isotonic value is constant within the block
            self.isotonic_values[i] = isotonic_values[start]

        return self


    def _predict_isotonic(self, y_score: NDArray) -> NDArray:
        """
        Apply isotonic calibration via linear interpolation.

        For each input score, linearly interpolate between the fitted
        (isotonic_scores -> isotonic_values) knot pairs.
        Scores outside the fitted range are clamped to boundary values.

        Parameters
        ----------
        y_score : ndarray of shape (n_samples,) or (n_samples, 1)

        Returns
        -------
        calibrated : ndarray, same shape as y_score
        """
        original_shape = y_score.shape
        scores_flat = y_score.ravel()

        calibrated_flat = np.interp(
            scores_flat,
            self.isotonic_scores,
            self.isotonic_values,
            left=self.isotonic_values[0],
            right=self.isotonic_values[-1],
        )

        return calibrated_flat.reshape(original_shape)

    # Monotone Spline (Fritsch-Carlson PCHIP) ----------
    def _fit_monotone_spline(self,
                             scores: NDArray,
                             labels: NDArray
                             ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Fit a monotone cubic Hermite spline (Fritsch-Carlson method).

        Steps:
          1. Apply PAVA to get monotone (score, probability) pairs.
          2. Deduplicate tied scores.
          3. Compute derivatives at each knot that guarantee monotonicity.

        Parameters
        ----------
        scores : ndarray, shape (n,)
        labels : ndarray, shape (n,) with values in {0, 1}

        Returns
        -------
        x : ndarray - knot positions (sorted unique scores)
        y : ndarray - knot values (isotonic probabilities)
        d : ndarray - derivatives at each knot (monotonicity-preserving)
        """
        # Get isotonic step function first
        self._fit_isotonic(scores, labels)

        # Need at least 2 points for a spline
        if len(self.isotonic_scores) < 2:
            return (self.isotonic_scores,
                    self.isotonic_values,
                    np.zeros_like(self.isotonic_values)
                    )

        # Compute monotone Hermite derivatives (Fritsch-Carlson)
        self.spline_d = self._fritsch_carlson_derivatives(self.isotonic_scores, self.isotonic_values)
        self.spline_x, self.spline_y = self.isotonic_scores, self.isotonic_values
        return self

    @staticmethod
    def _fritsch_carlson_derivatives(x: NDArray, y: NDArray) -> NDArray:
        """
        Compute Fritsch-Carlson monotone cubic Hermite derivatives.

        Guarantees the interpolant is monotone between each pair of data points.
        Reference: Fritsch & Carlson (1980), "Monotone Piecewise Cubic Interpolation"

        Parameters
        ----------
        x : ndarray, shape (n,) - strictly increasing knot positions
        y : ndarray, shape (n,) - non-decreasing knot values

        Returns
        -------
        d : ndarray, shape (n,) - derivatives at each knot
        """
        n = len(x)
        d = np.zeros(n)

        if n < 2:
            return d

        # Compute secants (slopes between adjacent points)
        # h[k] = x[k+1] - x[k], shape (n-1,)
        h = np.diff(x)
        # delta[k] = (y[k+1] - y[k]) / h[k], shape (n-1,)
        delta = np.diff(y) / h

        if n == 2:
            d[0] = delta[0]
            d[1] = delta[0]
            return d

        # interior points: initialize with harmonic mean of adjacent secants
        # (three-point formula for initial derivatives)
        for k in range(1, n - 1):
            if delta[k - 1] * delta[k] <= 0:
                # Sign change or zero => flat spot, set derivative to 0
                d[k] = 0.0
            else:
                # Harmonic mean weighted by interval lengths
                w1 = 2.0 * h[k] + h[k - 1]
                w2 = h[k] + 2.0 * h[k - 1]
                d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

        # Endpoint derivatives: one-sided shape-preserving
        d[0] = _edge_derivative(h[0], h[1], delta[0], delta[1])
        d[-1] = _edge_derivative(h[-1], h[-2], delta[-1], delta[-2])

        # Fritsch-Carlson monotonicity correction
        for k in range(n - 1):
            if abs(delta[k]) < 1e-30:
                # Flat segment: both endpoint derivatives must be zero
                d[k] = 0.0
                d[k + 1] = 0.0
            else:
                alpha = d[k] / delta[k]
                beta = d[k + 1] / delta[k]

                #If violated, scale down to the boundary of the monotonic region
                tau = alpha ** 2 + beta ** 2
                if tau > 3.0:
                    s = 3.0 / np.sqrt(tau)
                    d[k] = s * alpha * delta[k]
                    d[k + 1] = s * beta * delta[k]

        return d

    def _predict_spline(self, y_score: NDArray) -> NDArray:
        """
        Evaluate the monotone cubic Hermite spline at given points.

        Scores outside the fitted range are clamped to boundary values.
        Vectorized evaluation over all input scores.

        Parameters
        ----------
        y_score : ndarray of shape (n_samples,) or (n_samples, 1)

        Returns
        -------
        calibrated : ndarray, same shape as y_score
        """
        original_shape = y_score.shape
        scores_flat = y_score.ravel()

        s = np.clip(scores_flat, self.spline_x[0], self.spline_x[-1])

        # find interval index for each score
        k = np.searchsorted(self.spline_x, s, side="right") - 1
        k = np.clip(k, 0, len(self.spline_x) - 2)

        # local coordinates within each interval
        h = self.spline_x[k + 1] - self.spline_x[k]
        t = (s - self.spline_x[k]) / h  # t in [0, 1]

        # hermite basis functions (vectorized)
        h00 = (1.0 + 2.0 * t) * (1.0 - t) ** 2
        h10 = t * (1.0 - t) ** 2
        h01 = t ** 2 * (3.0 - 2.0 * t)
        h11 = t ** 2 * (t - 1.0)

        # interpolate
        result = (
            h00 * self.spline_y[k]
            + h10 * h * self.spline_d[k]
            + h01 * self.spline_y[k + 1]
            + h11 * h * self.spline_d[k + 1]
        )

        return np.clip(result, 0.0, 1.0).reshape(original_shape)


def _edge_derivative(h0: float, h1: float, delta0: float, delta1: float) -> float:
    """
    Non-centered three-point derivative for endpoint, shape-preserving.
    """
    d = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
    # If sign differs from adjacent secant, set to zero
    if d * delta0 < 0:
        d = 0.0
    # If sign of adjacent secants differ and magnitude is too large, clamp
    elif delta0 * delta1 < 0 and abs(d) > 3.0 * abs(delta0):
        d = 3.0 * delta0
    return d


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # synthetic uncalibrated scores and labels
    RNG = np.random.default_rng(42)
    count = 800

    y_true = RNG.choice([1, 0], size=(count, 1), replace=True)
    y_score = y_true * RNG.uniform(low=0.2, high=0.9999, size=(count,1))
    y_score += RNG.uniform(low=-0.2, high=0.2, size=(count,1))

    sorted = np.argsort(y_score, axis=0)

    def normalize(x):
        x -= x.min()
        return (x / x.max())

    ideal = np.linspace(0, 1, num=count)
    y_score_norm = normalize(y_score)
    logits = np.take(y_score_norm, sorted)
    plt.plot(logits, label="raw logit (normalized)", alpha=1.0, linewidth=2, color="black")
    plt.plot(ideal, label="ideal", alpha=0.5, linewidth=2, color="black")

    # Platt scaling
    platt_cal = ProbCalibration(method="platt", data_dimension=1)
    platt_cal.fit(y_score, y_true)
    calibrated_platt = platt_cal.predict(np.take(y_score, sorted))
    plt.plot(normalize(calibrated_platt), color="blue", alpha=0.66, label="platt_calibration")

    # isotonic regression
    iso_model = ProbCalibration(method="isotonic", data_dimension=1)
    iso_model.fit(y_score, y_true)
    calibrated_iso = iso_model.predict(np.take(y_score, sorted))
    plt.plot(calibrated_iso, label="isotonic mapped logit", alpha=0.66, color="orange")

    # spline
    spline_model = ProbCalibration(method="spline", data_dimension=1)
    spline_model.fit(y_score, y_true)
    calibrated_spline = spline_model.predict(np.take(y_score, sorted))
    plt.plot(calibrated_spline, label="spline mapped logit", alpha=0.66, color="purple")

    plt.legend()
    plt.show()


    plt.plot(ideal - logits.ravel(), alpha=0.5, color="black", label="logit diff")
    plt.plot(ideal - calibrated_platt.ravel(), alpha=0.5, color="blue", label="platt diff")
    plt.plot(ideal - calibrated_iso.ravel(), alpha=0.5, color="orange", label="isotonic diff")
    plt.plot(ideal - calibrated_spline.ravel(), alpha=0.5, color="purple", label="spline diff")
    plt.legend()
    plt.show()
