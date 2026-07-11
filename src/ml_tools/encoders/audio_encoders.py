import librosa

from encoders import Processor



class AudioProcessor(Processor):
    def __init__(
        self,
    ):
        super().__init__()


import numpy as np
from scipy import signal
import pywt

def dwt_single_level(x: np.ndarray, wavelet: str = "haar"):
    """
    Apply a single-level Discrete Wavelet Transform.

    Parameters
    ----------
    x : 1D array
    wavelet : str
        'haar', 'db2', 'db4', etc.

    Returns
    -------
    cA : approximation coefficients
    cD : detail coefficients
    """

    # Get wavelet filters
    # Ricker
    wavelet_obj = signal.wavelets._wavelets.Wavelet(wavelet)

    lo = wavelet_obj.dec_lo
    hi = wavelet_obj.dec_hi

    # Convolve and downsample
    cA = signal.convolve(x, lo, mode='same')[::2]
    cD = signal.convolve(x, hi, mode='same')[::2]

    return cA, cD


def dwt_multilevel(x: np.ndarray, levels: int = 3, wavelet: str = "haar"):
    """
    Multi-scale DWT decomposition.

    Returns:
        final approximation,
        list of detail coefficients per level
    """

    details = []
    current = x

    for _ in range(levels):
        cA, cD = dwt_single_level(current, wavelet)
        details.append(cD)
        current = cA

    return current, details


def dwt_windows(
    incoming_x: np.ndarray,
    levels: int = 3,
    wavelet: str = "haar"
):
    """
    Apply DWT to windowed input.

    Parameters
    ----------
    incoming_x : shape (num_windows, window_size)

    Returns
    -------
    List of tuples per window:
        (final_approx, details)
    """

    outputs = []

    for window in incoming_x:
        final_A, details = dwt_multilevel(
            window,
            levels=levels,
            wavelet=wavelet
        )
        outputs.append((final_A, details))

    return outputs

def idwt_multilevel(final_A, details, wavelet="haar"):
    """
    Reconstruct signal from multi-level DWT.
    """

    current = final_A

    wavelet_obj = signal.wavelets._wavelets.Wavelet(wavelet)
    lo = wavelet_obj.rec_lo
    hi = wavelet_obj.rec_hi

    for detail in reversed(details):
        up_A = np.zeros(len(detail) * 2)
        up_D = np.zeros(len(detail) * 2)

        up_A[::2] = current
        up_D[::2] = detail

        current = (
            signal.convolve(up_A, lo, mode='same') +
            signal.convolve(up_D, hi, mode='same')
        )

    return current