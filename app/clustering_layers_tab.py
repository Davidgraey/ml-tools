"""Clustering layers tab: prototype layers trained alone and inside an encoder -> layer -> classifier network."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from clustering_example import SCENARIOS
from clustering_tab import generate_frame
from common import edit_table, flush_figures, run_panel, show_diagram
from NNet_clustering_layers_example import (
    ANNEAL,
    ENERGY_WEIGHT,
    HIDDEN_DIM,
    LAYER_NAMES,
    LEARNING_RATE,
    PROTOTYPE_RATE,
    SUPERVISED_BATCH,
    SUPERVISED_EPOCHS,
    TEMPERATURE,
    UNSUPERVISED_BATCH,
    UNSUPERVISED_EPOCHS,
    build_classifier,
    classify_supervised,
    cluster_unsupervised,
)
from polyergalio.visuals.cluster_visuals import plot_prototype_layouts
from polyergalio.visuals.nnet_visuals import plot_network


def standardize(x: np.ndarray) -> np.ndarray:
    """Zero mean and unit spread per feature; constant features are only centered."""
    spread = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(spread > 0, spread, 1.0)


def plot_training_curves(results: dict) -> None:
    """Quantization error and prototype count per epoch, one line per layer."""
    fig, (errors, counts) = plt.subplots(1, 2, figsize=(10, 3.5))
    for name, result in results.items():
        history = result["history"]
        errors.plot([snapshot["error"] for snapshot in history], label=name)
        counts.plot([snapshot["num_prototypes"] for snapshot in history], label=name)
    errors.set_title("Quantization error")
    counts.set_title("Prototypes")
    for ax in (errors, counts):
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()


def plot_classifier_curves(results: dict) -> None:
    """Training loss and prototype count per epoch, one line per layer."""
    fig, (losses, counts) = plt.subplots(1, 2, figsize=(10, 3.5))
    for name, result in results.items():
        losses.plot(result["losses"], label=name)
        counts.plot(result["counts"], label=name)
    losses.set_title("Training loss")
    counts.set_title("Prototypes")
    for ax in (losses, counts):
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()


def train(x, truth, scenario, layers, num_clusters, unsupervised, supervised) -> None:
    """Train each chosen layer alone, then optionally inside a classifier, printing metrics and plotting results."""
    x = standardize(x)
    if x.shape[1] > 2:
        print(f"{x.shape[1]} features: layout plots show the first two dimensions only")

    print("unsupervised (clustering energy only):")
    results = {
        name: cluster_unsupervised(
            name, x, truth, num_clusters, unsupervised["epochs"], unsupervised["batch"], unsupervised["anneal"],
            temperature=unsupervised["temperature"],
        )
        for name in layers
    }
    plot_prototype_layouts(
        scenario,
        x,
        truth,
        {
            name: (result["layer"].weights, (result["layer"].height, result["layer"].width) if hasattr(result["layer"], "height") else None)
            for name, result in results.items()
        },
    )
    plot_training_curves(results)
    flush_figures()

    if supervised is None:
        return
    keep = truth >= 0
    _, classes = np.unique(truth[keep], return_inverse=True)
    print(f"\nsupervised (cross entropy through the layer; {int(np.sum(~keep))} outlier rows left out):")
    trained = {
        name: classify_supervised(
            name, x[keep], classes, supervised["epochs"], supervised["batch"], supervised["learning_rate"],
            supervised["hidden"], supervised["energy_weight"], supervised["prototype_rate"],
            anneal=unsupervised["anneal"], temperature=unsupervised["temperature"],
        )
        for name in layers
    }
    plot_classifier_curves(trained)
    flush_figures()


def render() -> None:
    left, right = st.columns([1, 2])
    with left:
        st.write(
            "Differentiable prototype layers (soft k-means, PLSOM, growing PLSOM and neural-gas PLSOM). "
            "Each layer is first trained alone on its own clustering energy, then placed between an encoder "
            "and a classifier, where the cross-entropy loss trains the encoder and the prototypes together."
        )
        diagram = st.container()
        st.subheader("Settings")
        layers = st.multiselect("Layers", LAYER_NAMES, default=list(LAYER_NAMES), key="layers_layers")
        st.caption("Unsupervised")
        epochs = int(st.number_input("Epochs", 1, 500, UNSUPERVISED_EPOCHS, step=10, key="layers_epochs"))
        batch = int(st.number_input("Batch size", 10, 2000, UNSUPERVISED_BATCH, step=10, key="layers_batch"))
        temperature = float(st.number_input("Assignment temperature", 0.01, 5.0, TEMPERATURE, step=0.05, key="layers_temperature"))
        anneal = float(st.number_input("Anneal per epoch", 0.5, 1.0, ANNEAL, step=0.01, key="layers_anneal"))
        supervised_on = st.checkbox("Also train inside a classifier", value=True, key="layers_supervised")
        hidden = int(st.number_input("Hidden dim", 2, 64, HIDDEN_DIM, key="layers_hidden"))
        supervised_epochs = int(st.number_input("Classifier epochs", 1, 300, SUPERVISED_EPOCHS, step=10, key="layers_sup_epochs"))
        supervised_batch = int(st.number_input("Classifier batch size", 10, 2000, SUPERVISED_BATCH, step=10, key="layers_sup_batch"))
        learning_rate = float(st.number_input("Adam learning rate", 0.0005, 0.5, LEARNING_RATE, step=0.005, format="%.4f", key="layers_lr"))
        energy_weight = float(st.number_input("Energy weight", 0.0, 2.0, ENERGY_WEIGHT, step=0.05, key="layers_energy"))
        prototype_rate = float(st.number_input("Prototype learning rate", 0.001, 1.0, PROTOTYPE_RATE, step=0.01, format="%.3f", key="layers_prototype_rate"))
        st.subheader("Data settings")
        scenario = st.selectbox("Scenario", list(SCENARIOS), key="layers_scenario")
        preset = SCENARIOS[scenario]
        samples = int(st.number_input("Samples", 100, 5000, 900, step=100, key="layers_samples"))
        features = int(st.number_input("Features", 2, 30, preset["num_features"], key="layers_features"))
        clusters = int(st.number_input("Clusters", 2, 20, preset["num_clusters"], key="layers_clusters"))
        seed = int(st.number_input("Seed", 0, 9999, 123, key="layers_seed"))
    config = dict(preset, num_samples=samples, num_features=features, num_clusters=clusters)

    with right:
        st.subheader("Data")
        frame, meta, unchanged = edit_table(
            f"layers_{scenario}",
            (scenario, samples, features, clusters, seed),
            lambda: generate_frame(config, seed),
        )
        st.caption("Columns F0, F1, ... are features; label is the true cluster (-1 for outliers) and is used for the classifier.")

        frame = frame.dropna(subset=[c for c in frame.columns if c != "label"])
        features_only = frame[[c for c in frame.columns if c != "label"]]
        x = features_only.to_numpy(dtype=float)
        truth = frame["label"].fillna(-1).to_numpy(dtype=int) if "label" in frame else np.zeros(len(x), dtype=int)
        if not layers:
            st.warning("Choose at least one layer.")
        elif supervised_on and np.sum(truth >= 0) == 0:
            st.warning("The classifier needs at least one labelled row.")
        else:
            supervised = (
                dict(
                    epochs=supervised_epochs, batch=supervised_batch, learning_rate=learning_rate, hidden=hidden,
                    energy_weight=energy_weight, prototype_rate=prototype_rate,
                )
                if supervised_on
                else None
            )
            unsupervised = dict(epochs=epochs, batch=batch, temperature=temperature, anneal=anneal)
            run_panel("layers", train, x, truth, scenario, layers, clusters, unsupervised, supervised)

    with diagram:
        with st.expander("Model structure", expanded=True):
            name = st.selectbox("Layer", LAYER_NAMES, key="layers_diagram")
            classes = max(2, int(truth.max()) + 1) if len(truth) else 2
            net = build_classifier(name, x.shape[1], classes, hidden, energy_weight, prototype_rate)
            show_diagram(plot_network(net, figsize=(5, 5)).figure)
