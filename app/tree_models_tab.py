"""Explainable Boosted Tree Model tab."""

import numpy as np
import streamlit as st
from common import run_panel, show_diagram
from diagrams import plot_graph
from model_plots import diagnostics
from polyergalio.generators.data_generators import to_onehot
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.supervised.trees.tree_models import ExplainableBoostedTreeModel
from polyergalio.visuals.supervised_visuals import plot_shape_functions
from supervised_data import select_data
from tree_models_example import ebm_weights_vector, top_features

TASKS = ("binary", "regression", "multiclass")

DIAGRAM = [
    ("Features X", "Quantile binning"),
    ("Quantile binning", "Main effects: one shape function per feature"),
    ("Quantile binning", "Pairwise interactions"),
    ("Main effects: one shape function per feature", "Additive score"),
    ("Pairwise interactions", "Additive score"),
    ("Intercept", "Additive score"),
    ("Additive score", "Task activation"),
    ("Task activation", "Prediction"),
]


def fit_model(data, num_bins: int, learning_rate: float, num_rounds: int, interaction_pairs: int) -> None:
    """Fit the EBM to the data, print the score, and draw diagnostics and shape functions."""
    num_features = data.x.shape[1]
    options = dict(
        input_dimension=num_features,
        num_bins=num_bins,
        learning_rate=learning_rate,
        num_rounds=num_rounds,
        max_interaction_pairs=interaction_pairs,
    )
    if data.task == "binary":
        model = ExplainableBoostedTreeModel(output_dimension=1, task=ClassificationTask.BINARY, **options)
        target = data.y
    elif data.task == "regression":
        model = ExplainableBoostedTreeModel(output_dimension=1, task=None, **options)
        target = data.y
    else:
        model = ExplainableBoostedTreeModel(
            output_dimension=data.num_classes, task=ClassificationTask.MULTINOMIAL, **options
        )
        target = to_onehot(data.y, data.num_classes)

    errors = model.fit(data.x, target)
    prediction = model.predict(data.x)
    r_square = None
    if data.task == "regression":
        r_square = 1 - np.sum((data.y - prediction) ** 2) / np.sum((data.y - np.mean(data.y)) ** 2)
        print(f"regression R2: {r_square:.3f}, final loss: {errors[-1]:.3f}")
    else:
        print(f"{data.task} accuracy: {np.mean(prediction == data.y):.3f}, final loss: {errors[-1]:.3f}")

    importance = model.get_feature_importance()
    diagnostics(
        model,
        data,
        f"{data.task.capitalize()} (EBM)",
        errors,
        f"ebm_{data.task}",
        y_true=target,
        y_pred=prediction,
        learned_weights=ebm_weights_vector(model, num_features),
        r_square=r_square,
    )
    if data.task != "multiclass":
        true = None if data.meta is None else np.abs(data.meta["weights"])
        plot_shape_functions(model, top_features(importance), true_weights=true, title_prefix=f"{data.task.capitalize()} shape")


def render() -> None:
    left, right = st.columns([1, 2])
    with left:
        st.write(
            "An additive model with pairwise interactions, fit by cyclic gradient boosting on binned features. "
            "Each feature's shape function shows its contribution directly."
        )
        with st.expander("Model structure", expanded=True):
            show_diagram(plot_graph(DIAGRAM, "Explainable Boosted Tree Model"))
            st.caption(
                "Boosting runs in two phases: main effects first, then up to the chosen number "
                "of interaction pairs fit to the remaining residuals."
            )
        st.subheader("Settings")
        task = st.selectbox("Task", TASKS, key="ebm_task")
        num_bins = int(st.number_input("Bins", 4, 64, 16, key="ebm_bins"))
        learning_rate = float(st.number_input("Learning rate", 0.001, 1.0, 0.05, step=0.01, format="%.3f", key="ebm_lr"))
        num_rounds = int(st.number_input("Rounds", 5, 300, 50, key="ebm_rounds"))
        pairs = int(st.number_input("Interaction pairs", 0, 20, 5, key="ebm_pairs"))
    with right:
        st.subheader("Data")
        data = select_data("ebm", task, samples=500, features=10, settings=left)
        run_panel("ebm", fit_model, data, num_bins, learning_rate, num_rounds, pairs)
