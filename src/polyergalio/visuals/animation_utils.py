"""
Generic support for matplotlib animations
"""

import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def thin_indices(length: int, stride: int) -> list[int]:
    """
    Indices of every `stride`-th snapshot, always including the last one.

    Parameters
    ----------
    length : number of snapshots available
    stride : keep every stride-th one

    Returns
    -------
    indices to keep
    """
    if stride <= 1 or length <= 1:
        return list(range(length))
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def thin_history(history: list, stride: int) -> list:
    """Keep every `stride`-th snapshot of a history list, always including the final one."""
    return [history[i] for i in thin_indices(len(history), stride)]


def save_and_show(
    anim: FuncAnimation, save_path: str, interval_ms: int, show: bool
) -> None:
    """
    Write an animation to disk as a GIF and/or open its interactive window.

    Parameters
    ----------
    anim : the animation to save/show
    save_path : GIF path to write to, via matplotlib's Pillow writer; skipped if falsy
    interval_ms : the animation's frame interval, used to derive the GIF's frame rate
    show : whether to also open the interactive window
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        print(f"  saving animation -> {save_path}")
        anim.save(save_path, writer=PillowWriter(fps=max(1, round(1000 / interval_ms))))
    if show:
        plt.show()
