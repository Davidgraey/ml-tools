"""Shared helpers for the Streamlit example app."""

import contextlib
import io
import os
import tempfile
import traceback
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


@dataclass
class Result:
    """Everything one run produced: printed text, figures, tables, images."""

    text: str = ""
    figures: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    images: list = field(default_factory=list)
    error: str = ""


current = Result()
close_figure = plt.close
layout = {}


def scratch_path(name: str) -> str:
    """Path for a temporary output file, creating its folder."""
    folder = os.path.join(tempfile.gettempdir(), "polyergalio_app")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def flush_figures() -> None:
    """Move open matplotlib figures into the current result."""
    current.figures.extend(plt.figure(number) for number in plt.get_fignums())
    close_figure("all")


def discard_figures() -> None:
    """Close open matplotlib figures without keeping them."""
    close_figure("all")


def emit_table(title: str, frame: pd.DataFrame) -> None:
    """Add a table to the current result."""
    current.tables.append((title, frame))


def emit_image(path: str, caption: str) -> None:
    """Add an image file, such as a GIF, to the current result."""
    with open(path, "rb") as handle:
        current.images.append((caption, handle.read()))


def capture(action, *args, **kwargs) -> Result:
    """
    Run action, collecting its printed text, figures, tables and images.

    plt.close is disabled while the action runs so figures the library
    closes itself are still collected.
    """
    global current
    current = Result()
    close_figure("all")
    buffer = io.StringIO()
    plt.close = lambda *args, **kwargs: None
    try:
        with contextlib.redirect_stdout(buffer):
            action(*args, **kwargs)
    except Exception:
        current.error = traceback.format_exc()
    finally:
        plt.close = close_figure
    flush_figures()
    current.text = buffer.getvalue()
    return current


def show_result(result: Result) -> None:
    """Render a stored result."""
    if result.error:
        st.error("The run failed.")
        st.code(result.error)
    if result.text:
        st.code(result.text, language="text")
    for title, frame in result.tables:
        st.caption(title)
        st.dataframe(frame)
    for figure in result.figures:
        st.pyplot(figure)
    for caption, data in result.images:
        st.image(data, caption=caption)


def run_panel(key: str, action, *args, label: str = "Run", **kwargs) -> None:
    """A run button plus the last stored result for key."""
    slot = f"{key}_result"
    if layout.pop("run", st).button(label, key=f"{key}_run"):
        with st.spinner("Running"):
            st.session_state[slot] = capture(action, *args, **kwargs)
    if slot in st.session_state:
        show_result(st.session_state[slot])


def show_diagram(figure, width: int = 300) -> None:
    """Render a diagram figure as a small fixed-width image and release it."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    st.image(buffer.getvalue(), width=width)


def edit_table(
    key: str,
    signature: tuple,
    generate,
    allow_upload: bool = True,
    dynamic_rows: bool = True,
    disabled: list = None,
):
    """
    Generated data in an editable table, with optional CSV upload.

    Parameters
    ----------
    key : unique widget and state prefix
    signature : generator settings; a change regenerates the table
    generate : callable returning (frame, meta)
    allow_upload : offer a CSV upload that replaces the table
    dynamic_rows : allow adding and deleting rows
    disabled : columns the user may not edit

    Returns
    -------
    edited frame, the generator meta (None for uploads), and whether the
    table still matches what was generated
    """
    state = st.session_state
    frame_key, meta_key, version_key = f"{key}_frame", f"{key}_meta", f"{key}_version"

    def load(frame, meta):
        state[frame_key] = frame
        state[meta_key] = meta
        state[version_key] = state.get(version_key, 0) + 1

    if allow_upload:
        upload = st.file_uploader("Upload a CSV to replace the table", type="csv", key=f"{key}_upload")
        if upload is not None:
            upload_id = (upload.name, upload.size)
            if state.get(f"{key}_upload_id") != upload_id:
                state[f"{key}_upload_id"] = upload_id
                load(pd.read_csv(upload), None)
    table = st.expander("Edit data")
    regenerate_column, run_column, _ = st.columns([1, 1, 3])
    layout["run"] = run_column
    regenerate = regenerate_column.button("Regenerate", key=f"{key}_regenerate")
    if regenerate or key + "_signature" not in state or state[key + "_signature"] != signature:
        state[key + "_signature"] = signature
        load(*generate())

    original = state[frame_key]
    with table:
        edited = st.data_editor(
            original,
            num_rows="dynamic" if dynamic_rows else "fixed",
            disabled=disabled or False,
            key=f"{key}_editor_{state[version_key]}",
            height=280,
        )
    unchanged = state[meta_key] is not None and edited.reset_index(drop=True).equals(original.reset_index(drop=True))
    return edited, state[meta_key], unchanged
