"""
Encoders: the audio feature pipeline, the shared utilities, and the state of
the tabular encoder modules.

The audio tests lean on Parseval's identity -- the one-sided power spectrum has
to carry the same energy as the time-domain frame -- because that catches both
a missing conjugate-pair correction and a wrong normalisation, at either
sequence-length parity.
"""

import numpy as np
import pytest

from polyergalio.encoders.audio_encoders import (
    AudioProcessor,
    build_windows,
    frequency_axis,
    from_decibels,
    overlap_add,
    to_decibels,
    to_power,
    window_size_from_ms,
    windowed_spectrum,
)
from polyergalio.encoders.encoder_utils import (
    calcualte_iqr_bounds,
    find_outliers,
    hash_string,
)
from polyergalio.encoders.encoders import Processor


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


def test_decibels_use_a_power_scale():
    power = np.array([1.0, 10.0, 100.0, 1000.0])
    assert np.allclose(to_decibels(power), [0.0, 10.0, 20.0, 30.0])


def test_decibels_round_trip():
    power = np.array([1e-6, 1.0, 500.0])
    assert np.allclose(from_decibels(to_decibels(power)), power)


def test_decibels_floor_at_zero_power():
    assert np.isfinite(to_decibels(np.zeros(4))).all()


def test_frequency_axis_spans_to_nyquist():
    axis = frequency_axis(window_size=64, sample_rate=8000)
    assert axis[0] == 0.0
    assert axis[-1] == pytest.approx(4000.0)
    assert axis.size == 64 // 2 + 1


def test_window_size_from_milliseconds():
    assert window_size_from_ms(sample_rate=8000, window_ms=20.0) == 160


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


# -------------    the Processor contract    -----------------------
def test_audio_processor_satisfies_the_processor_contract():
    assert issubclass(AudioProcessor, Processor)


def test_audio_processor_tracks_fitted_state():
    rng = np.random.default_rng(0)
    processor = AudioProcessor(sample_rate=1000, window_ms=20.0)
    assert not processor.is_fitted
    processor.fit(rng.normal(size=500))
    assert processor.is_fitted


# -------------    encoder utilities    ----------------------------
def test_iqr_bounds_bracket_the_middle_of_the_data():
    """these helpers call .dropna(), so they take a pandas Series not an array"""
    pandas = pytest.importorskip("pandas")
    values = pandas.Series(np.arange(100, dtype=float))
    low, high = calcualte_iqr_bounds(values)
    assert low < 25 and high > 75


def test_find_outliers_flags_an_extreme_value():
    pandas = pytest.importorskip("pandas")
    values = pandas.Series(np.concatenate([np.zeros(50), np.array([1e6])]))
    indices, _ = find_outliers(values, idxs_to_impute=set())
    assert 50 in set(indices)


def test_iqr_helpers_reject_a_bare_array():
    """
    Documenting the dependency: passing a numpy array fails on .dropna(), so
    callers have to hand these a Series.
    """
    with pytest.raises(AttributeError):
        calcualte_iqr_bounds(np.arange(10, dtype=float))


def test_hash_string_is_stable_and_varies():
    assert hash_string("alpha") == hash_string("alpha")
    assert hash_string("alpha") != hash_string("beta")


# -------------    the tabular encoder modules    ------------------
@pytest.mark.parametrize(
    "module",
    (
        "polyergalio.encoders.numeric_encoders",
        "polyergalio.encoders.categorical_encoders",
        "polyergalio.encoders.chronologic_encoders",
        "polyergalio.encoders.pipeline",
    ),
)
@pytest.mark.xfail(
    reason="these modules do `from encoders import Processor`, which only "
    "resolves when run from inside the encoders directory, so they cannot be "
    "imported as part of the installed package",
    strict=True,
)
def test_tabular_encoder_modules_import(module):
    __import__(module)
