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

from ml_tools.encoders.audio_encoders import (
    AudioProcessor,
    frame_signal,
    frequency_axis,
    from_decibels,
    overlap_add,
    to_decibels,
    to_power,
    window_size_from_ms,
    windowed_spectrum,
)
from ml_tools.encoders.encoder_utils import (
    calcualte_iqr_bounds,
    find_outliers,
    hash_string,
)
from ml_tools.encoders.encoders import Processor


WINDOW_LENGTHS = (8, 9, 16, 17, 64, 65)


@pytest.fixture()
def tone_signal():
    """one second of three summed tones landing on exact bin centres"""
    sample_rate = 8000
    seconds = np.arange(sample_rate) / sample_rate
    waveform = sum(
        amplitude * np.sin(2 * np.pi * frequency * seconds)
        for frequency, amplitude in ((250.0, 1.0), (1000.0, 0.5), (2500.0, 0.25))
    )
    return sample_rate, waveform


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
def test_frames_and_features_describe_the_same_segments(tone_signal):
    """
    The spectrogram must be computed from the same frames as the time-domain
    data. Previously the two used different strides, so their rows did not
    correspond and the row counts differed by half.
    """
    sample_rate, waveform = tone_signal
    processor = AudioProcessor(sample_rate=sample_rate, window_ms=20.0)
    features = processor.fit_encode(waveform)
    frames = frame_signal(waveform, processor.window_size, processor.num_overlap)
    assert frames.shape[0] == features.shape[0]


@pytest.mark.parametrize("window_ms", (9.0, 20.0, 21.0))
def test_round_trip_reconstructs_the_waveform(window_ms):
    """
    encode caches the complex spectrum, so inverse can reuse the phase and
    reconstruct exactly. Both parities of window length are covered.
    """
    rng = np.random.default_rng(0)
    sample_rate = 1000
    waveform = rng.normal(size=2 * sample_rate)

    processor = AudioProcessor(sample_rate=sample_rate, window_ms=window_ms)
    features = processor.fit_encode(waveform)
    reconstruction = processor.inverse(features)

    overlap = min(len(reconstruction), len(waveform))
    interior = slice(processor.window_size, overlap - processor.window_size)
    assert np.abs(reconstruction[interior] - waveform[interior]).max() < 1e-9


def test_inverse_without_a_cached_phase_is_still_finite():
    """magnitude alone cannot determine a signal, but it must not blow up"""
    rng = np.random.default_rng(0)
    waveform = rng.normal(size=2000)
    fitted = AudioProcessor(sample_rate=1000, window_ms=20.0)
    features = fitted.fit_encode(waveform)

    fresh = AudioProcessor(sample_rate=1000, window_ms=20.0)
    fresh.fit(waveform)
    assert np.isfinite(fresh.inverse(features)).all()


def test_overlap_add_inverts_framing():
    rng = np.random.default_rng(0)
    waveform = rng.normal(size=500)
    window_size, overlap = 40, 10
    frames = frame_signal(waveform, window_size, overlap)
    rebuilt = overlap_add(frames, overlap)

    covered = min(len(rebuilt), len(waveform))
    assert np.allclose(rebuilt[window_size:covered - window_size],
                       waveform[window_size:covered - window_size])


# -------------    the encoder finds planted structure    ----------
def test_encoding_recovers_the_planted_tones(tone_signal):
    sample_rate, waveform = tone_signal
    processor = AudioProcessor(sample_rate=sample_rate, window_ms=20.0)
    features = processor.fit_encode(waveform)

    mean_power = from_decibels(features).mean(axis=0)
    for frequency in (250.0, 1000.0, 2500.0):
        nearby = np.abs(processor.freqs - frequency) <= 100.0
        detected = processor.freqs[nearby][np.argmax(mean_power[nearby])]
        assert detected == pytest.approx(frequency, abs=processor.freqs[1])


def test_amplitude_ordering_survives_encoding(tone_signal):
    """the 1.0, 0.5, 0.25 tones must come out in that order"""
    sample_rate, waveform = tone_signal
    processor = AudioProcessor(sample_rate=sample_rate, window_ms=20.0)
    features = processor.fit_encode(waveform)
    mean_power = from_decibels(features).mean(axis=0)

    peaks = []
    for frequency in (250.0, 1000.0, 2500.0):
        nearby = np.abs(processor.freqs - frequency) <= 100.0
        peaks.append(mean_power[nearby].max())
    assert peaks[0] > peaks[1] > peaks[2]


# -------------    the Processor contract    -----------------------
def test_audio_processor_satisfies_the_processor_contract():
    assert issubclass(AudioProcessor, Processor)


def test_audio_processor_reports_metadata():
    processor = AudioProcessor(sample_rate=8000, window_ms=20.0, variable_idx=1)
    metadata = processor.metadata
    assert metadata["sample_rate"] == 8000
    assert metadata["window_size"] == 160
    assert metadata["variable_idx"] == 1
    assert metadata["num_frequencies"] == 160 // 2 + 1


def test_audio_processor_tracks_fitted_state():
    rng = np.random.default_rng(0)
    processor = AudioProcessor(sample_rate=1000, window_ms=20.0)
    assert not processor.is_fitted
    processor.fit(rng.normal(size=500))
    assert processor.is_fitted


def test_power_mode_skips_the_decibel_conversion():
    rng = np.random.default_rng(0)
    waveform = rng.normal(size=2000)
    power = AudioProcessor(1000, 20.0, use_decibels=False).fit_encode(waveform)
    assert (power >= 0).all(), "power cannot be negative"


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
        "ml_tools.encoders.numeric_encoders",
        "ml_tools.encoders.categorical_encoders",
        "ml_tools.encoders.chronologic_encoders",
        "ml_tools.encoders.pipeline",
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
