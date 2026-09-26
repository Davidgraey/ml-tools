"""
Encoders: the audio feature pipeline and the shared utilities.

The audio tests lean on Parseval's identity -- the one-sided power spectrum has
to carry the same energy as the time-domain frame -- because that catches both
a missing conjugate-pair correction and a wrong normalisation, at either
sequence-length parity.
"""

import numpy as np
import pytest

from polyergalio.encoders.audio_encoders import (
    build_windows,
    from_decibels,
    overlap_add,
    to_decibels,
    to_power,
    windowed_spectrum,
)
from polyergalio.encoders.encoder_utils import find_outliers


WINDOW_LENGTHS = (8, 9, 16, 17, 64, 65)


# -------------    spectral identities    --------------------------
@pytest.mark.parametrize("window_size", WINDOW_LENGTHS)
def test_one_sided_power_conserves_energy(window_size):
    """
    Parseval: the summed one-sided power equals the frame energy times the
    window length. Only the paired bins double, and Nyquist exists only for
    even lengths -- inferring that from the bin count alone is impossible.
    """
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(5, window_size))
    spectrum = windowed_spectrum(frames, window_kernel=np.ones(window_size))
    power = to_power(spectrum, window_size=window_size)

    assert np.allclose(
        power.sum(axis=-1), window_size * (frames ** 2).sum(axis=-1)
    )


def test_decibels_round_trip():
    power = np.array([1e-6, 1.0, 500.0])
    assert np.allclose(from_decibels(to_decibels(power)), power)


# -------------    framing and reconstruction    -------------------
def test_overlap_add_inverts_framing():
    rng = np.random.default_rng(0)
    waveform = rng.normal(size=500)
    window_size, overlap = 40, 10
    frames = build_windows(waveform, window_size, overlap)
    rebuilt = overlap_add(frames, overlap)

    covered = min(len(rebuilt), len(waveform))
    assert np.allclose(rebuilt[window_size:covered - window_size],
                       waveform[window_size:covered - window_size])


# -------------    encoder utilities    ----------------------------
def test_find_outliers_flags_an_extreme_value():
    pandas = pytest.importorskip("pandas")
    values = pandas.Series(np.concatenate([np.zeros(50), np.array([1e6])]))
    indices, _ = find_outliers(values, idxs_to_impute=set())
    assert 50 in set(indices)
