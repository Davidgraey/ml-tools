import numpy as np
from ml_tools.models.layers import Layer
import numpy as np
from ml_tools.utilities import rolling_windows_nd


def learnable_fft(X, G):
    """
    LM-WT-inspired learnable FFT block

    X : (T, d)
    G : (T, d) learnable frequency gates

    Returns
    -------
    Y : (T, d)
    """
    x_f = np.fft.fftshift(np.fft.fft2(x, axes=(-2, -1))).real

    Xf_mod = x_shift * G
    return np.real(np.fft.ifft(Xf_mod, axis=0))


def band_gated_fft(X, bands):
    Xf = np.fft.rfft(X, axis=0).real
    Yf = np.zeros_like(Xf)

    for band, gate in bands:
        Yf[band] = Xf[band] * gate

    return np.real(np.fft.ifft(Yf, axis=0))


# 1D:
# x_f = np.fft.rfft(x, axis=(-1)).real
# x_shift = np.fft.fftshift(x_f).real
# x_f_centered = np.fft.fftshift(x_shift, axes=-1).real
# x_f_centered = np.fft.fftshift(x_shift, axes=-2).real
# np.fft.fftshift(x_f[0].reshape(15, 65), axes=0).real.shape


# 2D:
# x_f = np.fft.rfft2(x, axes=(-2, -1)).real
# x_f_centered = np.fft.fftshift(x_f, axes=(-2,-1)).real


# make embedding layer
# take in variables, catgorical, etc
# Autoencoder -> layers to project to latent space
# reconstruction layer to rebuild the input
# train on reconstruct loss


class FourierAttention(Layer):
    #  https://ieeexplore.ieee.org/document/9616294
    def forward(self, x):
        fft_output = np.fft.fft2(x, axes=(-2, -1)).real
        # scale, constrain or norm?

    def backward(self, incoming_grad):
        df_fft = np.fft.ifft(incoming_grad).real




def wavelet_embedding(Layer):
    """
    _____________________
    |   LL    |   LH    |
    ----------+----------
    |   HL    |   HH    |
    _____________________
    Parameters
    ----------
    Layer :

    Returns
    -------

    """
    # from scipy.signal import cwt, morlet








# Wavelet packet transform (even closer to FFT bins)


if __name__ == "__main__":
    # (num_samples, x_axis (words in sentence, etc), hidden dimension)
    x = np.random.uniform(size=(12, 28, 512))

    x_hat = np.fft.fft2(x, axes=(-2, -1)).real.astype(np.float32)

    assert x.shape == x_hat.shape

    inverted = np.fft.ifft2(x_hat, axes=(-1, -2)).real.astype(np.float32)

    np.isclose(x, inverted)


