"""Dataset selection and editing shared by the supervised tabs."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from common import edit_table
from polyergalio.generators.data_generators import RandomDatasetGenerator

TASKS = ("regression", "binary", "multiclass", "multilabel")


@dataclass
class SupervisedData:
    task: str
    x: np.ndarray
    y: np.ndarray
    meta: Optional[dict]
    num_classes: int


def to_frame(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Feature columns F0.. followed by target columns y, or y0.. for several targets."""
    features = pd.DataFrame(x, columns=[f"F{i}" for i in range(x.shape[1])])
    if y.ndim == 1:
        targets = pd.DataFrame({"y": y})
    else:
        targets = pd.DataFrame(y, columns=[f"y{j}" for j in range(y.shape[1])])
    return pd.concat([features, targets], axis=1)


def from_frame(frame: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray]:
    """Split a table back into x and y, target columns being those starting with y."""
    frame = frame.dropna()
    targets = [name for name in frame.columns if str(name).startswith("y")]
    features = [name for name in frame.columns if name not in targets]
    x = frame[features].to_numpy(dtype=float)
    y = frame[targets].to_numpy(dtype=float)
    if task == "regression":
        return x, y[:, 0]
    if task == "multilabel":
        return x, y.astype(int)
    return x, y[:, 0].astype(int)


def generate_frame(task, num_samples, num_features, noise, num_classes, seed):
    """Generator output as (table, meta)."""
    options = dict(task=task, num_samples=num_samples, num_features=num_features, noise_scale=noise, verbose=False)
    if task != "regression":
        options["num_classes"] = num_classes
    x, y, meta = RandomDatasetGenerator(random_seed=seed).generate(**options)
    return to_frame(x, y), meta


def select_data(key: str, task: str, samples: int = 300, features: int = 8, classes: int = 4, settings=st) -> SupervisedData:
    """
    Generator controls and an editable table for one task.

    Parameters
    ----------
    key : unique widget and state prefix
    task : one of TASKS
    samples, features, classes : starting generator settings
    settings : container that receives the generator controls

    Returns
    -------
    SupervisedData; meta is None once the table differs from what was generated
    """
    num_samples = int(settings.number_input("Samples", 20, 5000, samples, step=20, key=f"{key}_samples"))
    num_features = int(settings.number_input("Features", 2, 30, features, key=f"{key}_features"))
    noise = float(settings.number_input("Noise", 0.0, 5.0, 0.33, step=0.05, key=f"{key}_noise"))
    if task in ("multiclass", "multilabel"):
        num_classes = int(settings.number_input("Classes", 3, 12, classes, key=f"{key}_classes"))
    else:
        num_classes = 2
    seed = int(settings.number_input("Seed", 0, 9999, 42, key=f"{key}_seed"))

    signature = (task, num_samples, num_features, noise, num_classes, seed)
    frame, meta, unchanged = edit_table(
        f"{key}_{task}",
        signature,
        lambda: generate_frame(task, num_samples, num_features, noise, num_classes, seed),
    )
    meta = meta if unchanged else None
    st.caption(
        "Feature columns first, then target columns named y (or y0, y1, ... for several targets). "
        + ("True weights are available for comparison." if meta is not None else "The table was modified, so true-weight comparisons are hidden.")
    )
    x, y = from_frame(frame, task)
    if task == "multiclass":
        num_classes = int(y.max()) + 1
    elif task == "multilabel":
        num_classes = y.shape[1]
    return SupervisedData(task, x, y, meta, num_classes)
