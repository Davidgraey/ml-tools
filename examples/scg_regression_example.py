"""
Scaled Conjugate Gradient regression example.

Fits GradientDescent (regression, binary, multinomial, multilabel) against
RandomDatasetGenerator tasks and renders the unified diagnostic dashboard --
loss curve, pred-vs-target, and true-vs-learned weight comparison -- from
polyergalio.visuals.supervised_visuals.plot_model_diagnostics for each.

Run: python scg_regression_example.py
"""

import numpy as np
from polyergalio.generators.data_generators import RandomDatasetGenerator, to_onehot
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.supervised.scg_regression import GradientDescent
from polyergalio.visuals.supervised_visuals import plot_model_diagnostics

NUM_SAMPLES = 2000
NUM_FEATURES = 8
NOISE = 0.33
NUM_STEPS = 25


def run_regression(generator: RandomDatasetGenerator) -> None:
    x, y, meta = generator.generate(
        task="regression",
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES,
        noise_scale=NOISE,
    )
    for use_elastic_reg in (False, True):
        model = GradientDescent(
            task="regression", use_elastic_reg=use_elastic_reg, early_termination=True
        )
        errors = model.fit(x_data=x, y_data=y.reshape(-1, 1), iterations=NUM_STEPS)
        print(f"regression (elastic={use_elastic_reg}) R2: {model.r_square:.3f}")

        plot_model_diagnostics(
            model, x, y, meta,
            task_label="Regression",
            errors=errors,
            filename_prefix=f"scg_regression_elastic{use_elastic_reg}",
        )


def run_binary(generator: RandomDatasetGenerator) -> None:
    x, y, meta = generator.generate(
        task="binary",
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES,
        num_classes=2,
        noise_scale=NOISE,
    )
    model = GradientDescent(
        task=ClassificationTask.BINARY, use_elastic_reg=False, early_termination=True
    )
    errors = model.fit(x_data=x, y_data=to_onehot(y), iterations=NUM_STEPS)

    pred = np.argmax(model.predict(x), axis=-1)
    print(f"binary accuracy: {np.mean(pred == y):.3f}")
    plot_model_diagnostics(
        model, x, y, meta,
        task_label="Binary Classification",
        errors=errors,
        filename_prefix="scg_binary",
    )


def run_multinomial(generator: RandomDatasetGenerator) -> None:
    x, y, meta = generator.generate(
        task="multiclass",
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES,
        num_classes=6,
        noise_scale=NOISE,
    )
    model = GradientDescent(
        task=ClassificationTask.MULTINOMIAL, use_elastic_reg=False, early_termination=True
    )
    errors = model.fit(x_data=x, y_data=to_onehot(y), iterations=NUM_STEPS)

    pred = np.argmax(model.predict(x), axis=-1)
    print(f"multinomial accuracy: {np.mean(pred == y):.3f}")
    plot_model_diagnostics(
        model, x, y, meta,
        task_label="Multinomial Classification",
        errors=errors,
        filename_prefix="scg_multinomial",
    )


def run_multilabel(generator: RandomDatasetGenerator) -> None:
    num_classes = 5
    x, y, meta = generator.generate(
        task="multilabel",
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES,
        num_classes=num_classes,
        noise_scale=NOISE,
    )
    model = GradientDescent(
        task=ClassificationTask.MULTILABEL, use_elastic_reg=False, early_termination=True
    )
    errors = model.fit(x_data=x, y_data=y, iterations=NUM_STEPS)

    pred = (model.predict(x) > 0.5).astype(int)
    print(f"multilabel elementwise accuracy: {np.mean(pred == y):.3f}")
    plot_model_diagnostics(
        model, x, y, meta,
        task_label="Multilabel Classification",
        errors=errors,
        filename_prefix="scg_multilabel",
        multilabel=True,
    )


if __name__ == "__main__":
    generator = RandomDatasetGenerator(random_seed=44)
    run_regression(generator)
    run_binary(generator)
    run_multinomial(generator)
    run_multilabel(generator)
