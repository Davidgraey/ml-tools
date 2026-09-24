"""
Relative Weight Analysis (RWA) example.

Runs relative_weights() -- the heuristic estimator of each predictor's
share of a model's R² -- against RandomDatasetGenerator's regression,
binary, multiclass, and multilabel tasks, and compares its estimated
importance to the generator's planted |weight| magnitudes.

Plotting lives in ml_tools.visuals.supervised_visuals: plot_relative_weights
(true vs RWA importance, one bar chart or a per-class grid) and
plot_prediction_scatter (pred vs target, regression only -- RWA's
classification predictions are hard labels, not a useful scatter).

Run: python relative_weights_example.py
"""

import numpy as np
from ml_tools.generators.data_generators import RandomDatasetGenerator, to_onehot
from ml_tools.models.supervised.relative_weights import relative_weights
from ml_tools.visuals.supervised_visuals import (
    plot_prediction_scatter,
    plot_relative_weights,
)

NUM_SAMPLES = 2000
NUM_FEATURES = 8
NOISE = 0.33


def run_regression(generator: RandomDatasetGenerator) -> None:
    x, y, meta = generator.generate(
        task="regression",
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES,
        noise_scale=NOISE,
    )
    result = relative_weights(x, y, logistic=False)

    plot_prediction_scatter(y, result["model_prediction"], title="Regression: pred vs target")
    plot_relative_weights(
        np.abs(meta["weights"]),
        result["sign_norm_rwa"].ravel(),
        title="Regression RWA",
    )


def run_binary(generator: RandomDatasetGenerator) -> None:
    x, y, meta = generator.generate(
        task="binary",
        num_samples=NUM_SAMPLES,
        num_classes=2,
        num_features=NUM_FEATURES,
        noise_scale=NOISE,
    )
    result = relative_weights(x, to_onehot(y), logistic=True)

    pred = np.argmax(result["model_prediction"], axis=-1)
    print(f"binary accuracy: {np.mean(pred == y):.3f}")
    plot_relative_weights(
        np.abs(meta["weights"]),
        np.sum(result["norm_rwa"], axis=-1),
        title="Binary Classification RWA",
    )


def run_multiclass(generator: RandomDatasetGenerator) -> None:
    num_classes = 6
    x, y, meta = generator.generate(
        task="multiclass",
        num_samples=NUM_SAMPLES,
        num_classes=num_classes,
        num_features=NUM_FEATURES,
        noise_scale=NOISE,
    )
    result = relative_weights(x, to_onehot(y), logistic=True)

    pred = np.argmax(result["model_prediction"], axis=-1)
    print(f"multiclass accuracy: {np.mean(pred == y):.3f}")
    plot_relative_weights(
        np.sum(np.abs(meta["weights"]), axis=-1),
        result["norm_rwa"],
        title="Multiclass RWA",
    )


def run_multilabel(generator: RandomDatasetGenerator) -> None:
    num_classes = 5
    x, y, meta = generator.generate(
        task="multilabel",
        num_samples=NUM_SAMPLES,
        num_classes=num_classes,
        num_features=NUM_FEATURES,
        noise_scale=NOISE,
    )
    result = relative_weights(x, y, logistic=True)

    pred = result["model_prediction"].round()
    print(f"multilabel elementwise accuracy: {np.mean(pred == y):.3f}")
    plot_relative_weights(
        np.sum(np.abs(meta["weights"]), axis=-1),
        result["norm_rwa"],
        title="Multilabel RWA",
    )


if __name__ == "__main__":
    generator = RandomDatasetGenerator(random_seed=123)
    run_regression(generator)
    run_binary(generator)
    run_multiclass(generator)
    run_multilabel(generator)
