import numpy as np
import matplotlib.pyplot as plt

def plot_model_diagnostics(
        model, x, y, meta, task_label: str, errors: list, filename_prefix: str,
        learned_weights=None, r_square=None,
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
    from ml_tools.models.constants import EPSILON
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
    tail = errors[max(0, len(errors) - 20):]
    ax_loss_tail.plot(range(len(errors) - len(tail), len(errors)), tail,
                      color="steelblue", linewidth=1.2)
    ax_loss_tail.set_title("Loss (last 20 epochs)")
    ax_loss_tail.set_xlabel("epoch")

    ax_pred = fig.add_subplot(gs[0, 2])
    pred = model.predict(x)
    if task_label == "Regression":
        # scatter subset for clarity
        idx = np.random.default_rng(0).choice(len(y), min(500, len(y)), replace=False)
        ax_pred.scatter(y[idx], pred[idx], alpha=0.4, s=12, c="steelblue", edgecolors="none")
        lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
        ax_pred.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax_pred.set_xlabel("Target")
        ax_pred.set_ylabel("Prediction")
        _r2 = r_square if r_square is not None else getattr(model, "r_square", None)
        if _r2 is not None:
            ax_pred.set_title(f"Pred vs Target  (R²={_r2:.3f})")
        else:
            ax_pred.set_title("Pred vs Target")
    else:
        pred_classes = np.argmax(pred, axis=-1) if pred.ndim > 1 else pred.ravel()
        y_classes = y if y.ndim == 1 else np.argmax(y, axis=-1)
        acc = np.mean(pred_classes == y_classes)
        idx = np.random.default_rng(0).choice(len(y_classes), min(500, len(y_classes)), replace=False)
        ax_pred.scatter(idx, y_classes[idx], alpha=0.6, s=14, c="coral", label="target")
        ax_pred.scatter(idx, pred_classes[idx], alpha=0.3, s=14, c="steelblue", label="pred")
        ax_pred.set_title(f"Pred vs Target  (acc={acc:.3f})")
        ax_pred.legend(fontsize=8)
        ax_pred.set_xlabel("sample")
        ax_pred.set_ylabel("class")

    # ── Row 2: Normalised importance bar chart (full width)
    ax_bar = fig.add_subplot(gs[1, :])
    x_pos = np.arange(n_coef)
    width = 0.35
    ax_bar.bar(x_pos - width / 2, norm_true, width, label="True |weight|", color="coral")
    ax_bar.bar(x_pos + width / 2, norm_learned, width, label="Learned |weight|", color="steelblue")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(feature_labels, fontsize=9)
    ax_bar.set_ylabel("Normalised magnitude")
    ax_bar.set_title("True vs Learned Feature Importance (normalised)")
    ax_bar.legend()
    ax_bar.axhline(0, color="grey", linewidth=0.5)

    # ── Row 3: Signed weight comparison | Absolute error (stem) | Raw weights
    ax_signed = fig.add_subplot(gs[2, 0])
    ax_signed.scatter(x_pos, _true, marker="D", s=40, c="coral", label="True β", zorder=3)
    ax_signed.scatter(x_pos, learned, marker="o", s=40, c="steelblue", label="Learned β", zorder=3)
    for i in range(n_coef):
        ax_signed.plot([i, i], [_true[i], learned[i]], color="grey", linewidth=0.8, alpha=0.6)
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
