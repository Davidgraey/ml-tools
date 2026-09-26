"""Scaled Conjugate Gradient regression tab."""

import numpy as np
import streamlit as st
from common import run_panel, show_diagram
from diagrams import plot_graph
from model_plots import diagnostics
from polyergalio.generators.data_generators import to_onehot
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.supervised.scg_regression import GradientDescent
from supervised_data import TASKS, select_data

DIAGRAM = [
    ("Inputs X", "Standardize"),
    ("Standardize", "Linear map: X . W"),
    ("Weights W (Kaiming init)", "Linear map: X . W"),
    ("Linear map: X . W", "Activation: linear, sigmoid or softmax"),
    ("Activation: linear, sigmoid or softmax", "Loss: MSE or cross-entropy, optional elastic net"),
    ("Loss: MSE or cross-entropy, optional elastic net", "SCG step, alternating with gradient descent"),
    ("SCG step, alternating with gradient descent", "Updated weights W"),
]

CLASSIFICATION = {
    "binary": ClassificationTask.BINARY,
    "multiclass": ClassificationTask.MULTINOMIAL,
    "multilabel": ClassificationTask.MULTILABEL,
}


def fit_model(data, elastic: bool, early: bool, steps: int, reg_lambda: float, reg_alpha: float) -> None:
    """Fit GradientDescent to the data, print the score, and draw its diagnostics."""
    options = dict(use_elastic_reg=elastic, reg_lambda=reg_lambda, reg_alpha=reg_alpha, early_termination=early)
    if data.task == "regression":
        model = GradientDescent(task="regression", **options)
        target = data.y.reshape(-1, 1)
    else:
        model = GradientDescent(task=CLASSIFICATION[data.task], **options)
        target = data.y if data.task == "multilabel" else to_onehot(data.y)
    errors = model.fit(x_data=data.x, y_data=target, iterations=steps)
    prediction = model.predict(data.x)

    if data.task == "regression":
        print(f"regression R2: {model.r_square:.3f}")
    elif data.task == "multilabel":
        print(f"multilabel elementwise accuracy: {np.mean((prediction > 0.5).astype(int) == data.y):.3f}")
    else:
        print(f"{data.task} accuracy: {np.mean(np.argmax(prediction, axis=-1) == data.y):.3f}")

    diagnostics(
        model,
        data,
        data.task.capitalize(),
        errors,
        f"scg_{data.task}",
        y_true=target,
        y_pred=prediction,
        multilabel=data.task == "multilabel",
    )


def render() -> None:
    left, right = st.columns([1, 2])
    with left:
        st.write(
            "Linear and logistic models fit with Scaled Conjugate Gradient, "
            "which needs no learning rate, plus optional elastic net regularization."
        )
        with st.expander("Model structure", expanded=True):
            show_diagram(plot_graph(DIAGRAM, "GradientDescent (SCG)"))
        st.subheader("Settings")
        task = st.selectbox("Task", TASKS, key="scg_task")
        steps = int(st.number_input("Iterations", 1, 500, 25, key="scg_steps"))
        early = st.checkbox("Early termination", value=True, key="scg_early")
        elastic = st.checkbox("Elastic net", key="scg_elastic")
        reg_lambda = float(st.number_input("Reg strength", 0.0, 10.0, 0.1, step=0.05, key="scg_lambda"))
        reg_alpha = st.slider("Elastic net mix (0 lasso, 1 ridge)", 0.0, 1.0, 0.5, key="scg_alpha")
    with right:
        st.subheader("Data")
        data = select_data("scg", task, samples=1000, features=8, settings=left)
        run_panel("scg", fit_model, data, elastic, early, steps, reg_lambda, reg_alpha)
