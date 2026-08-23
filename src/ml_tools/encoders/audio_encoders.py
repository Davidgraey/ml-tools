"""
------------------------ Audio preprocessing / encoders ------------------------
Waveform to windowed spectral features and back again.

The stages are exposed as plain functions so each one can be checked on its
own, and AudioProcessor wraps them to satisfy the Processor contract used by
the rest of the encoders package.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from scipy.io import wavfile
import matplotlib.pyplot as plt

from ml_tools.encoders.encoders import Processor
from ml_tools.utilities import rolling_windows_nd, standardize_data


EPSILON = 1e-12


# -------------    waveform to frames    ---------------------------
# ------------------------------------------------------------------
def read_wav(
    file_path: str, channel: Optional[int] = 0, standardize: bool = True
) -> tuple[int, NDArray]:
    """
    Read a wav file and pull out one channel.

    Parameters
    ----------
    file_path : path to the wav file
    channel : channel index, or None to keep every channel
    standardize : zero mean and unit variance the samples

    Returns
    -------
    (sample_rate, waveform)
    """
    sample_rate, waveform = wavfile.read(file_path)

    if channel is not None and waveform.ndim > 1:
        waveform = waveform[:, channel]

    waveform = waveform.astype(np.float64)
    if standardize:
        waveform = standardize_data(waveform, axis=0)

    return sample_rate, waveform


def window_size_from_ms(sample_rate: int, window_ms: float) -> int:
    """number of samples spanned by a window of window_ms milliseconds"""
    return int(window_ms * sample_rate / 1000)


def frame_signal(
    waveform: NDArray, window_size: int, num_overlap: int = 0
) -> NDArray:
    """
    Cut the waveform into overlapping frames, shaped (num_frames, window_size).
    Every downstream stage works from these frames, so the spectrogram and the
    time domain frames always describe the same segments of signal.
    """
    return rolling_windows_nd(
        data=waveform, window_size=window_size, num_overlap=num_overlap, axis=0
    )


# -------------    frames to spectral features    ------------------
# ------------------------------------------------------------------
def windowed_spectrum(
    frames: NDArray, window_kernel: Optional[NDArray] = None
) -> NDArray:
    """
    Real FFT of each frame, tapered first to stop the frame edges ringing.

    Returns the complex spectrum, shaped (..., window_size // 2 + 1). Keep it
    if you intend to reconstruct, since the phase lives here and cannot be
    recovered from magnitude alone.
    """
    if window_kernel is None:
        window_kernel = np.blackman(frames.shape[-1])

    return np.fft.rfft(frames * window_kernel, axis=-1)


def _paired_bins(num_frequencies: int, window_size: Optional[int]) -> slice:
    """
    Which rfft bins stand for a conjugate pair, and so carry double the energy
    of a two sided spectrum.

    DC is never paired. Nyquist exists, and is likewise unpaired, only when the
    window length is even. The bin count alone cannot tell us which case we are
    in, since a window of 8 and a window of 9 both produce 5 bins, so the
    window size has to be supplied for odd length windows.
    """
    if window_size is None:
        window_size = 2 * (num_frequencies - 1)

    nyquist_present = window_size % 2 == 0
    return slice(1, -1 if nyquist_present else None)


def to_power(
    spectrum: NDArray, window_size: Optional[int] = None, one_sided: bool = True
) -> NDArray:
    """
    Power spectrum, |X|^2 for every bin.

    Parameters
    ----------
    spectrum : complex rfft output
    window_size : length of the window the spectrum came from. Defaults to the
        even case, so pass it explicitly whenever the window length is odd.
    one_sided : double the paired bins so the total matches the energy of the
        full two sided spectrum
    """
    power = np.abs(spectrum) ** 2

    if one_sided:
        power[..., _paired_bins(spectrum.shape[-1], window_size)] *= 2

    return power


def to_decibels(power: NDArray, epsilon: float = EPSILON) -> NDArray:
    """decibels from a power spectrum, 10 * log10, floored by epsilon"""
    return 10 * np.log10(power + epsilon)


def from_decibels(decibels: NDArray, epsilon: float = EPSILON) -> NDArray:
    """inverse of to_decibels"""
    return np.maximum(10 ** (decibels / 10) - epsilon, 0.0)


def frequency_axis(window_size: int, sample_rate: int) -> NDArray:
    """centre frequency in Hz of each rfft bin"""
    return np.fft.rfftfreq(window_size, d=1.0 / sample_rate)


# -------------    frames back to a waveform    --------------------
# ------------------------------------------------------------------
def overlap_add(
    frames: NDArray, num_overlap: int = 0, window_kernel: Optional[NDArray] = None
) -> NDArray:
    """
    Fold overlapping frames back into one signal, weighted so that the taper
    applied on the way in is divided out on the way back.
    """
    num_frames, window_size = frames.shape
    stride = window_size - num_overlap

    if window_kernel is None:
        window_kernel = np.ones(window_size)

    length = stride * (num_frames - 1) + window_size
    signal = np.zeros(length)
    weight = np.zeros(length)

    for i in range(num_frames):
        start = i * stride
        signal[start: start + window_size] += frames[i] * window_kernel
        weight[start: start + window_size] += window_kernel ** 2

    return signal / np.maximum(weight, EPSILON)


# -------------    plotting    -------------------------------------
# ------------------------------------------------------------------
def plot_waveform(waveform: NDArray, sample_rate: Optional[int] = None) -> None:
    """time domain view of one channel"""
    xs = np.arange(waveform.shape[0])
    label = "Sample"
    if sample_rate is not None:
        xs = xs / sample_rate
        label = "Time (seconds)"

    plt.figure(figsize=(10, 4))
    plt.plot(xs, waveform)
    plt.title("Audio waveform")
    plt.xlabel(label)
    plt.ylabel("Amplitude")
    plt.axis("tight")
    plt.show()


def plot_spectrogram(
    decibels: NDArray, freqs: NDArray, sample_rate: Optional[int] = None
) -> None:
    """decibel spectrogram, frames on x and frequency on y"""
    time_axis = np.arange(decibels.shape[0])
    label = "Frame"

    plt.figure(figsize=(15, 8))
    plt.pcolormesh(time_axis, freqs, decibels.T, shading="auto")
    plt.xlabel(label)
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    plt.colorbar(label="dB")
    plt.show()


def plot_window_spectra(
    spectrum: NDArray,
    freqs: NDArray,
    index: int = 0,
    window_size: Optional[int] = None,
) -> None:
    """
    Amplitude, power and decibel view of a single frame. Replaces the older
    plot_fft_windows, which assumed three named axes and a fixed subplot grid.
    """
    magnitude = np.abs(spectrum[index])
    power = to_power(spectrum[index: index + 1], window_size=window_size)[0]
    decibels = to_decibels(power)

    panels = (
        ("Amplitude spectrum", magnitude, "amplitude", plt.plot),
        ("Power spectrum", power, "power", plt.plot),
        ("Decibel spectrum", decibels, "dB", plt.plot),
    )

    plt.figure(figsize=(14, 10))
    for position, (title, values, ylabel, draw) in enumerate(panels, start=1):
        plt.subplot(len(panels), 1, position)
        draw(freqs, values)
        plt.title(f"{title}, frame {index}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(ylabel)
        plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
class AudioProcessor(Processor):
    """
    Waveform to a windowed spectral representation.

    variable_idx carries the channel to pull out of a multi channel file, so
    it keeps the same meaning it has for the tabular encoders: which column of
    the source the encoder is responsible for.
    """

    def __init__(
        self,
        sample_rate: int,
        window_ms: float = 20.0,
        overlap_ratio: float = 1 / 3,
        use_decibels: bool = True,
        target: str = "waveform",
        variable_idx: int = 0,
    ):
        """
        Parameters
        ----------
        sample_rate : samples per second of the source signal
        window_ms : frame length in milliseconds
        overlap_ratio : fraction of a frame shared with the next frame
        use_decibels : encode to decibels rather than raw power
        target : name of the signal being encoded
        variable_idx : channel index within a multi channel source
        """
        super().__init__(target=target, variable_idx=variable_idx)
        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.overlap_ratio = overlap_ratio
        self.use_decibels = use_decibels

        self.window_size: int = window_size_from_ms(sample_rate, window_ms)
        self.num_overlap: int = int(self.window_size * overlap_ratio)
        self.window_kernel: NDArray = np.blackman(self.window_size)
        self.freqs: NDArray = frequency_axis(self.window_size, sample_rate)

        self.spectrum: Optional[NDArray] = None

    def fit(self, values: NDArray) -> "AudioProcessor":
        """record the observed range, which inverse() needs to undo scaling"""
        self.obs_min_max = (float(np.min(values)), float(np.max(values)))
        self._fitted = True
        return self

    def encode(self, values: NDArray) -> NDArray:
        """
        Waveform to (num_frames, num_frequencies). The complex spectrum is
        cached on the way through so inverse() can reuse its phase.
        """
        frames = frame_signal(values, self.window_size, self.num_overlap)
        self.spectrum = windowed_spectrum(frames, self.window_kernel)

        power = to_power(self.spectrum, window_size=self.window_size)
        return to_decibels(power) if self.use_decibels else power

    def fit_encode(self, values: NDArray) -> NDArray:
        return self.fit(values).encode(values)

    def inverse(self, values: NDArray) -> NDArray:
        """
        Spectral features back to a waveform.

        Magnitude alone does not determine a signal, so the phase cached by the
        last encode() is reused when its shape matches. Without that cache the
        reconstruction is zero phase and only approximate.
        """
        power = from_decibels(values) if self.use_decibels else values

        two_sided = power.copy()
        two_sided[..., _paired_bins(power.shape[-1], self.window_size)] /= 2

        magnitude = np.sqrt(two_sided)

        if self.spectrum is not None and self.spectrum.shape == magnitude.shape:
            phase = np.exp(1j * np.angle(self.spectrum))
        else:
            phase = 1.0

        frames = np.fft.irfft(magnitude * phase, n=self.window_size, axis=-1)
        return overlap_add(frames, self.num_overlap, self.window_kernel)

    @property
    def metadata(self) -> dict:
        return {
            "target": self.target,
            "variable_idx": self.variable_idx,
            "sample_rate": self.sample_rate,
            "window_ms": self.window_ms,
            "window_size": self.window_size,
            "num_overlap": self.num_overlap,
            "num_frequencies": self.freqs.size,
            "use_decibels": self.use_decibels,
        }

    def __str__(self):
        units = "dB" if self.use_decibels else "power"
        return (
            f"AudioProcessor at {self.sample_rate} Hz, {self.window_size} sample "
            f"windows overlapping {self.num_overlap}, encoding to {units}"
        )


if __name__ == "__main__":
    SAMPLE_RATE = 8000
    DURATION_SECONDS = 1.0
    WINDOW_MS = 20.0

    # frequency, amplitude. All three land on bin centres at this window size,
    # so the encoded peaks can be checked against the tones that made them.
    TONES = ((250.0, 1.0), (1000.0, 0.5), (2500.0, 0.25))

    seconds = np.arange(int(SAMPLE_RATE * DURATION_SECONDS)) / SAMPLE_RATE
    waveform = sum(
        amplitude * np.sin(2 * np.pi * frequency * seconds)
        for frequency, amplitude in TONES
    )

    print("---- source signal ----")
    print(f"{waveform.size} samples at {SAMPLE_RATE} Hz")
    for frequency, amplitude in TONES:
        print(f"  tone {frequency:7.1f} Hz at amplitude {amplitude}")

    # ---- the stages, one at a time ----
    window_size = window_size_from_ms(SAMPLE_RATE, WINDOW_MS)
    num_overlap = window_size // 3

    frames = frame_signal(waveform, window_size, num_overlap)
    spectrum = windowed_spectrum(frames)
    power = to_power(spectrum, window_size=window_size)
    decibels = to_decibels(power)
    freqs = frequency_axis(window_size, SAMPLE_RATE)

    print("\n---- staged encoding ----")
    print(f"window_size      {window_size} samples ({WINDOW_MS} ms)")
    print(f"num_overlap      {num_overlap} samples")
    print(f"bin width        {freqs[1] - freqs[0]:.1f} Hz")
    print(f"frames           {frames.shape}")
    print(f"spectrum         {spectrum.shape}  {spectrum.dtype}")
    print(f"power, decibels  {power.shape}, {decibels.shape}")

    # each tone should show up as the loudest bin near its own frequency
    mean_power = power.mean(axis=0)
    print("\n---- recovered peaks ----")
    for frequency, amplitude in TONES:
        nearby = np.abs(freqs - frequency) <= 100.0
        detected = freqs[nearby][np.argmax(mean_power[nearby])]
        loudest = 10 * np.log10(mean_power[nearby].max() / mean_power.max())
        print(
            f"  expected {frequency:7.1f} Hz -> found {detected:7.1f} Hz"
            f"   {loudest:+6.2f} dB relative to the strongest tone"
        )

    # ---- the same thing through the Processor interface ----
    processor = AudioProcessor(
        sample_rate=SAMPLE_RATE, window_ms=WINDOW_MS, overlap_ratio=1 / 3
    )
    features = processor.fit_encode(waveform)

    print(f"\n---- {processor} ----")
    print(f"features   {features.shape}")
    print(f"is_fitted  {processor.is_fitted}")
    for key, value in processor.metadata.items():
        print(f"  {key:16s} {value}")

    reconstruction = processor.inverse(features)
    overlap = min(reconstruction.size, waveform.size)
    interior = slice(window_size, overlap - window_size)
    print(f"\nreconstruction   {reconstruction.shape}")
    print(
        "interior error   "
        f"{np.abs(reconstruction[interior] - waveform[interior]).max():.2e}"
    )

    # ---- plots ----
    visible = slice(0, int(0.05 * SAMPLE_RATE))
    plot_waveform(waveform[visible], SAMPLE_RATE)
    plot_spectrogram(features, processor.freqs)
    plot_window_spectra(
        processor.spectrum, processor.freqs, index=10, window_size=window_size
    )
