"""
Explainable Boosted Tree Model (EBM) example.

Fits ExplainableBoostedTreeModel (binary, regression, multiclass) against
RandomDatasetGenerator tasks, renders the unified diagnostic dashboard for
each via polyergalio.visuals.supervised_visuals.plot_model_diagnostics, and
plots the learned shape function of the top-2 most important features via
plot_shape_functions.

Run: python tree_models_example.py
"""

import numpy as np
from polyergalio.generators.data_generators import RandomDatasetGenerator, to_onehot
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.supervised.trees.tree_models import ExplainableBoostedTreeModel
from polyergalio.visuals.supervised_visuals import plot_model_diagnostics, plot_shape_functions


def ebm_weights_vector(model: ExplainableBoostedTreeModel, num_features: int) -> np.ndarray:
    """
    Assemble [intercept, importance_0, ..., importance_{D-1}].

    EBM importance is unsigned and keyed by feature index rather than
    stored as a weight vector; this shapes it to match what
    plot_model_diagnostics expects from `learned_weights` (intercept first).
    """
    importance = model.get_feature_importance()
    intercept = model.intercept if np.isscalar(model.intercept) else float(np.mean(model.intercept))
    return np.array([float(intercept)] + [importance[i] for i in range(num_features)])


def top_features(importance: dict, count: int = 2) -> list:
    """The `count` feature indices with the largest importance, most important first."""
    return [feature for feature, _ in sorted(importance.items(), key=lambda kv: -kv[1])[:count]]


def run_binary(generator: RandomDatasetGenerator) -> dict:
    x, y, meta = generator.generate("binary", num_samples=500, num_features=10, noise_scale=0.8)
    model = ExplainableBoostedTreeModel(
        input_dimension=10, output_dimension=1, task=ClassificationTask.BINARY,
        num_bins=16, learning_rate=0.05, num_rounds=50, max_interaction_pairs=5,
    )
    errors = model.fit(x, y)
    pred = model.predict(x)
    print(f"binary accuracy: {np.mean(pred == y):.3f}, final loss: {errors[-1]:.3f}")

    importance = model.get_feature_importance()
    true_weights = np.abs(meta["weights"])
    plot_model_diagnostics(
        model, x, y, meta,
        task_label="Binary (EBM)",
        errors=errors,
        filename_prefix="ebm_binary",
        learned_weights=ebm_weights_vector(model, 10),
    )
    plot_shape_functions(
        model, top_features(importance), true_weights=true_weights,
        title_prefix="Binary shape", color="steelblue",
        save_path="ebm_shape_functions_binary.png",
    )
    return dict(model=model, true_weights=true_weights)


def run_regression(generator: RandomDatasetGenerator) -> dict:
    x, y, meta = generator.generate("regression", num_samples=500, num_features=10, noise_scale=0.8)
    model = ExplainableBoostedTreeModel(
        input_dimension=10, output_dimension=1, task=None,
        num_bins=24, learning_rate=0.1, num_rounds=80, max_interaction_pairs=4,
    )
    errors = model.fit(x, y)
    pred = model.predict(x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_square = 1 - ss_res / ss_tot
    print(f"regression R2: {r_square:.3f}, final loss: {errors[-1]:.3f}")

    importance = model.get_feature_importance()
    true_weights = np.abs(meta["weights"])
    plot_model_diagnostics(
        model, x, y, meta,
        task_label="Regression",
        errors=errors,
        filename_prefix="ebm_regression",
        learned_weights=ebm_weights_vector(model, 10),
        r_square=r_square,
    )
    plot_shape_functions(
        model, top_features(importance), true_weights=true_weights,
        title_prefix="Regression shape", color="teal",
        save_path="ebm_shape_functions_regression.png",
    )
    return dict(model=model, true_weights=true_weights)


def run_multiclass(generator: RandomDatasetGenerator) -> dict:
    num_classes = 4
    x, y_int, meta = generator.generate(
        "multiclass", num_samples=600, num_features=10, num_classes=num_classes, noise_scale=0.8
    )
    y = to_onehot(y_int, num_classes)
    model = ExplainableBoostedTreeModel(
        input_dimension=10, output_dimension=num_classes, task=ClassificationTask.MULTINOMIAL,
        num_bins=16, learning_rate=0.05, num_rounds=60, max_interaction_pairs=5,
    )
    errors = model.fit(x, y)
    pred = model.predict(x)
    print(f"multiclass accuracy: {np.mean(pred == y_int):.3f}, final loss: {errors[-1]:.3f}")

    # weight matrix is (features, classes); row-wise L2 norm collapses to per-feature magnitude
    true_weights = np.linalg.norm(meta["weights"], axis=1)
    plot_model_diagnostics(
        model, x, y, meta,
        task_label="Multiclass (EBM)",
        errors=errors,
        filename_prefix="ebm_multiclass",
        learned_weights=ebm_weights_vector(model, 10),
    )
    return dict(model=model, true_weights=true_weights)


if __name__ == "__main__":
    gen = RandomDatasetGenerator(random_seed=42)
    run_binary(gen)
    run_regression(gen)
    run_multiclass(gen)
    print("\nAll plots displayed and saved.")
