"""Relative Weight Analysis tab."""

import numpy as np
import pandas as pd
import streamlit as st
from common import emit_table, run_panel, show_diagram
from diagrams import plot_graph
from model_plots import plot_importance
from polyergalio.generators.data_generators import to_onehot
from polyergalio.models.supervised.relative_weights import relative_weights
from polyergalio.visuals.supervised_visuals import (
    plot_prediction_scatter,
    plot_relative_weights,
)
from supervised_data import TASKS, select_data

DIAGRAM = [
    ("Predictors X", "z-score"),
    ("z-score", "SVD: orthogonal predictors Z"),
    ("SVD: orthogonal predictors Z", "Regress y on Z: beta"),
    ("Target y", "Regress y on Z: beta"),
    ("SVD: orthogonal predictors Z", "Regress X on Z: lambda"),
    ("z-score", "Regress X on Z: lambda"),
    ("Regress y on Z: beta", "Raw weight: lambda^2 . beta^2"),
    ("Regress X on Z: lambda", "Raw weight: lambda^2 . beta^2"),
    ("Raw weight: lambda^2 . beta^2", "Normalized importance"),
]


def analyse(data) -> None:
    """Run RWA on the data, plot importance against the planted weights when known, print accuracy."""
    x, y, meta = data.x, data.y, data.meta
    if data.task == "regression":
        result = relative_weights(x, y, logistic=False)
        importance = result["sign_norm_rwa"].ravel()
        true = None if meta is None else np.abs(meta["weights"])
        plot_prediction_scatter(y, result["model_prediction"], title="Regression: pred vs target")
    elif data.task == "binary":
        result = relative_weights(x, to_onehot(y), logistic=True)
        print(f"binary accuracy: {np.mean(np.argmax(result['model_prediction'], axis=-1) == y):.3f}")
        importance = np.sum(result["norm_rwa"], axis=-1)
        true = None if meta is None else np.abs(meta["weights"])
    elif data.task == "multiclass":
        result = relative_weights(x, to_onehot(y), logistic=True)
        print(f"multiclass accuracy: {np.mean(np.argmax(result['model_prediction'], axis=-1) == y):.3f}")
        importance = result["norm_rwa"]
        true = None if meta is None else np.sum(np.abs(meta["weights"]), axis=-1)
    else:
        result = relative_weights(x, y, logistic=True)
        print(f"multilabel elementwise accuracy: {np.mean(result['model_prediction'].round() == y):.3f}")
        importance = result["norm_rwa"]
        true = None if meta is None else np.sum(np.abs(meta["weights"]), axis=-1)

    title = f"{data.task.capitalize()} RWA"
    if true is None:
        plot_importance(importance, title)
    else:
        plot_relative_weights(true, importance, title=title)

    summed = importance if np.ndim(importance) == 1 else np.sum(np.abs(importance), axis=-1)
    emit_table(
        "Normalized importance per feature",
        pd.DataFrame({"feature": [f"F{i}" for i in range(len(summed))], "importance": summed}),
    )


def render() -> None:
    left, right = st.columns([1, 2])
    with left:
        st.write(
            "Relative Weight Analysis estimates each predictor's share of a model's R-squared "
            "by rotating correlated predictors into orthogonal ones."
        )
        with st.expander("Model structure", expanded=True):
            show_diagram(plot_graph(DIAGRAM, "Relative weight analysis"))
        st.subheader("Settings")
        task = st.selectbox("Task", TASKS, key="rwa_task")
    with right:
        st.subheader("Data")
        data = select_data("rwa", task, samples=1000, features=8, settings=left)
        run_panel("rwa", analyse, data)
