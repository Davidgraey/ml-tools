from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from polyergalio.models.constants import EPSILON


def plot_model_diagnostics(
    model,
    x,
    y,
    meta,
    task_label: str,
    errors: list,
    filename_prefix: str,
    learned_weights=None,
    r_square=None,
    multilabel: bool = False,
):
    """
    Unified diagnostic plot combining:
      - Loss curve (full + zoomed tail)
      - Normalised true-weight vs learned-weight bar chart
      - Prediction vs target overlay
      - Signed weight error (stem plot)

    Parameters
    ----------
    learned_weights : array-like, optional
        Override for model weights (useful for models like EBM that expose
        feature importance differently). Should include intercept at index 0.
    r_square : float, optional
        Override for R² value shown in regression pred-vs-target plot.
    multilabel : bool
        True when y carries independent per-class labels rather than one
        label per row (e.g. sigmoid-thresholded output, not argmax). Uses
        elementwise accuracy and a flattened pred/target scatter instead of
        the single-label argmax comparison.
    """
    true_coef = meta["weights"].squeeze()
    bias = meta["bias"]

    # Build comparable weight vectors (with intercept at index 0)
    _true = np.insert(true_coef, 0, bias, axis=0)

    if learned_weights is not None:
        learned = np.asarray(learned_weights).squeeze()
    else:
        learned = model.weights.squeeze()

    # For multi-output, collapse to per-feature magnitude
    if _true.ndim > 1:
        _true = np.linalg.norm(_true, axis=-1)
    if learned.ndim > 1:
        learned = np.linalg.norm(learned, axis=-1)

    n_coef = len(_true)
    feature_labels = ["bias"] + [f"F{i}" for i in range(n_coef - 1)]

    # Normalise to [0, 1] for visual comparison
    abs_true = np.abs(_true)
    abs_learned = np.abs(learned)
    norm_true = abs_true / (abs_true.max() + EPSILON)
    norm_learned = abs_learned / (abs_learned.max() + EPSILON)

    # ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"{task_label} — Model Diagnostics", fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(3, 3)

    # ── Row 1: Loss curve (full) | Loss curve (tail) | Pred vs Target
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(errors, color="steelblue", linewidth=1.2)
    ax_loss.set_title("Loss (full)")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")

    ax_loss_tail = fig.add_subplot(gs[0, 1])
    tail = errors[max(0, len(errors) - 20) :]
    ax_loss_tail.plot(
        range(len(errors) - len(tail), len(errors)),
        tail,
        color="steelblue",
        linewidth=1.2,
    )
    ax_loss_tail.set_title("Loss (last 20 epochs)")
    ax_loss_tail.set_xlabel("epoch")

    ax_pred = fig.add_subplot(gs[0, 2])
    pred = model.predict(x)
    if task_label == "Regression":
        # scatter subset for clarity
        idx = np.random.default_rng(0).choice(len(y), min(500, len(y)), replace=False)
        ax_pred.scatter(
            y[idx], pred[idx], alpha=0.4, s=12, c="steelblue", edgecolors="none"
        )
        lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
        ax_pred.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax_pred.set_xlabel("Target")
        ax_pred.set_ylabel("Prediction")
        _r2 = r_square if r_square is not None else getattr(model, "r_square", None)
        if _r2 is not None:
            ax_pred.set_title(f"Pred vs Target  (R²={_r2:.3f})")
        else:
            ax_pred.set_title("Pred vs Target")
    elif multilabel:
        pred_binary = (pred > 0.5).astype(int)
        acc = np.mean(pred_binary == y)
        flat_true = y.ravel()
        flat_pred = pred_binary.ravel()
        idx = np.random.default_rng(0).choice(
            len(flat_true), min(1000, len(flat_true)), replace=False
        )
        ax_pred.scatter(idx, flat_true[idx], alpha=0.6, s=10, c="coral", label="target")
        ax_pred.scatter(
            idx, flat_pred[idx], alpha=0.3, s=10, c="steelblue", label="pred"
        )
        ax_pred.set_title(f"Pred vs Target  (elementwise acc={acc:.3f})")
        ax_pred.legend(fontsize=8)
        ax_pred.set_xlabel("(sample, class) index")
        ax_pred.set_ylabel("active")
    else:
        pred_classes = np.argmax(pred, axis=-1) if pred.ndim > 1 else pred.ravel()
        y_classes = y if y.ndim == 1 else np.argmax(y, axis=-1)
        acc = np.mean(pred_classes == y_classes)
        idx = np.random.default_rng(0).choice(
            len(y_classes), min(500, len(y_classes)), replace=False
        )
        ax_pred.scatter(idx, y_classes[idx], alpha=0.6, s=14, c="coral", label="target")
        ax_pred.scatter(
            idx, pred_classes[idx], alpha=0.3, s=14, c="steelblue", label="pred"
        )
        ax_pred.set_title(f"Pred vs Target  (acc={acc:.3f})")
        ax_pred.legend(fontsize=8)
        ax_pred.set_xlabel("sample")
        ax_pred.set_ylabel("class")

    # ── Row 2: Normalised importance bar chart (full width)
    ax_bar = fig.add_subplot(gs[1, :])
    x_pos = np.arange(n_coef)
    width = 0.35
    ax_bar.bar(
        x_pos - width / 2, norm_true, width, label="True |weight|", color="coral"
    )
    ax_bar.bar(
        x_pos + width / 2,
        norm_learned,
        width,
        label="Learned |weight|",
        color="steelblue",
    )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(feature_labels, fontsize=9)
    ax_bar.set_ylabel("Normalised magnitude")
    ax_bar.set_title("True vs Learned Feature Importance (normalised)")
    ax_bar.legend()
    ax_bar.axhline(0, color="grey", linewidth=0.5)

    # ── Row 3: Signed weight comparison | Absolute error (stem) | Raw weights
    ax_signed = fig.add_subplot(gs[2, 0])
    ax_signed.scatter(
        x_pos, _true, marker="D", s=40, c="coral", label="True β", zorder=3
    )
    ax_signed.scatter(
        x_pos, learned, marker="o", s=40, c="steelblue", label="Learned β", zorder=3
    )
    for i in range(n_coef):
        ax_signed.plot(
            [i, i], [_true[i], learned[i]], color="grey", linewidth=0.8, alpha=0.6
        )
    ax_signed.set_xticks(x_pos)
    ax_signed.set_xticklabels(feature_labels, fontsize=8)
    ax_signed.set_title("Signed Weights (true vs learned)")
    ax_signed.legend(fontsize=8)
    ax_signed.axhline(0, color="grey", linewidth=0.5)

    ax_err = fig.add_subplot(gs[2, 1])
    errors_vec = learned - _true
    colors = ["coral" if e < 0 else "steelblue" for e in errors_vec]
    ax_err.bar(x_pos, errors_vec, color=colors, alpha=0.7)
    ax_err.set_xticks(x_pos)
    ax_err.set_xticklabels(feature_labels, fontsize=8)
    ax_err.set_title("Weight Error (learned − true)")
    ax_err.axhline(0, color="grey", linewidth=0.5)

    ax_abs = fig.add_subplot(gs[2, 2])
    markerline, stemlines, baseline = ax_abs.stem(x_pos, np.abs(errors_vec))
    plt.setp(stemlines, linewidth=1.2, color="grey")
    plt.setp(markerline, markersize=5, color="coral")
    ax_abs.set_xticks(x_pos)
    ax_abs.set_xticklabels(feature_labels, fontsize=8)
    ax_abs.set_title("|Weight Error|")
    ax_abs.set_ylabel("absolute error")

    plt.savefig(f"{filename_prefix}_diagnostics.png", dpi=120)
    plt.show()
    plt.close(fig)


def plot_prediction_scatter(
    y_true: NDArray, y_pred: NDArray, title: str = "Prediction vs Target"
) -> None:
    """
    Scatter of predicted vs true values with a y=x reference line.

    Parameters
    ----------
    y_true : (samples,) or (samples, outputs) targets
    y_pred : same shape, model predictions
    title : plot title
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.4, s=12, color="steelblue", edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_relative_weights(
    true_importance: NDArray,
    rwa_importance: NDArray,
    class_labels: Optional[list] = None,
    title: str = "Relative Weight Analysis",
) -> None:
    """
    True vs RWA-estimated feature importance, each normalised to [0, 1] by
    its own max magnitude.

    One bar chart when rwa_importance is 1D (a single target). A grid of
    one subplot per class -- all sharing the same true_importance reference
    bars -- when rwa_importance carries a class axis (features, classes),
    e.g. multiclass or multilabel RWA.

    Parameters
    ----------
    true_importance : (features,) reference |weight| magnitudes
    rwa_importance : (features,) or (features, classes) RWA-estimated importance
    class_labels : label per class, defaults to "class {i}"; only used
        when rwa_importance is 2D
    title : figure title
    """
    true_importance = np.abs(np.ravel(true_importance))
    rwa_importance = np.asarray(rwa_importance)
    n_features = len(true_importance)
    feature_labels = [f"F{i}" for i in range(n_features)]
    norm_true = true_importance / (true_importance.max() + EPSILON)

    def _draw(ax, rwa_column, subtitle):
        norm_rwa = np.abs(rwa_column) / (np.abs(rwa_column).max() + EPSILON)
        ax.barh(range(n_features), norm_true, alpha=0.75, color="steelblue", label="true |weight|")
        ax.barh(range(n_features), norm_rwa, alpha=0.5, color="coral", label="RWA")
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(feature_labels, fontsize=8)
        ax.set_title(subtitle)

    if rwa_importance.ndim == 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        _draw(ax, rwa_importance, title)
        ax.legend()
    else:
        n_classes = rwa_importance.shape[-1]
        class_labels = class_labels or [f"class {i}" for i in range(n_classes)]
        cols = min(3, n_classes)
        rows = int(np.ceil(n_classes / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
        for class_idx, ax in zip(range(n_classes), axes.flat):
            _draw(ax, rwa_importance[:, class_idx], class_labels[class_idx])
        for ax in axes.flat[n_classes:]:
            ax.axis("off")
        axes.flat[0].legend()
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def plot_shape_functions(
    model,
    feature_indices: list,
    true_weights: Optional[NDArray] = None,
    title_prefix: str = "",
    color: str = "steelblue",
    save_path: Optional[str] = None,
) -> None:
    """
    An EBM's per-feature shape function (binned contribution to the
    prediction) for a handful of features, side by side.

    Parameters
    ----------
    model : a fitted model exposing get_shape_function(feature_index) -> (centers, contributions)
    feature_indices : which features to plot, one subplot each
    true_weights : optional (features,) reference magnitudes, noted in each title
    title_prefix : prefix for every subplot title, e.g. "Binary shape"
    color : bar color
    save_path : if given, the figure is saved there before being shown
    """
    fig, axes = plt.subplots(
        1, len(feature_indices), figsize=(6 * len(feature_indices), 4), squeeze=False
    )
    for ax, feature_index in zip(axes.flat, feature_indices):
        _, contributions = model.get_shape_function(feature_index)
        ax.bar(range(len(contributions)), contributions, color=color)
        label = f"Feature {feature_index}"
        if true_weights is not None:
            label += f"  (true |w|={true_weights[feature_index]:.3f})"
        ax.set_title(f"{title_prefix} — {label}" if title_prefix else label)
        ax.set_xlabel("Bin index")
        ax.set_ylabel("Contribution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()
    plt.close(fig)
