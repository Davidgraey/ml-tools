"""Diagnostic figures shared by the supervised tabs."""

import matplotlib.pyplot as plt
import numpy as np
from common import scratch_path
from polyergalio.visuals.supervised_visuals import plot_model_diagnostics


def plot_fit_summary(errors, y_true, y_pred, weights, title: str) -> None:
    """Loss curve, prediction vs target, and learned weight magnitudes, without needing true weights."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(errors)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Iteration")

    y_true = np.ravel(y_true).astype(float)
    y_pred = np.ravel(y_pred).astype(float)
    axes[1].scatter(y_true, y_pred, alpha=0.4, s=12, color="steelblue", edgecolors="none")
    limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(limits, limits, "k--", alpha=0.5, linewidth=1)
    axes[1].set_title("Prediction vs target")
    axes[1].set_xlabel("Target")
    axes[1].set_ylabel("Prediction")

    weights = np.abs(np.asarray(weights).squeeze())
    if weights.ndim > 1:
        weights = np.linalg.norm(weights, axis=-1)
    labels = ["bias"] + [f"F{i}" for i in range(len(weights) - 1)]
    axes[2].bar(labels, weights, color="coral")
    axes[2].set_title("Learned weight magnitude")
    axes[2].tick_params(axis="x", labelsize=8)

    fig.suptitle(title)
    fig.tight_layout()


def diagnostics(model, data, task_label, errors, prefix, y_true, y_pred, learned_weights=None, r_square=None, multilabel=False) -> None:
    """The library's full diagnostic figure when true weights exist, otherwise plot_fit_summary."""
    if data.meta is not None:
        plot_model_diagnostics(
            model,
            data.x,
            data.y,
            data.meta,
            task_label=task_label,
            errors=errors,
            filename_prefix=scratch_path(prefix),
            learned_weights=learned_weights,
            r_square=r_square,
            multilabel=multilabel,
        )
        return
    weights = learned_weights if learned_weights is not None else model.weights
    plot_fit_summary(errors, y_true, y_pred, weights, task_label)


def plot_importance(importance, title: str) -> None:
    """Bar chart of per-feature importance; class columns are summed by magnitude."""
    importance = np.abs(np.asarray(importance))
    if importance.ndim > 1:
        importance = importance.sum(axis=-1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"F{i}" for i in range(len(importance))], importance, color="coral")
    ax.set_title(title)
    fig.tight_layout()
