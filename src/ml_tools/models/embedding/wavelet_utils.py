from scipy.signal import upfirdn
from ml_tools.models.layers.layers import Layer
import numpy as np

'''
https://arxiv.org/abs/2504.08801
https://github.com/thqiu0419/MLWNet
'''


class WaveletEmbedding(Layer):
    """
    ai = α⊙x2i +β⊙x2i+1,
    di = γ⊙x2i +δ⊙x2i+1,
    """
    def __init__(self, ni: int, no: int):
        super().__init__()
        # haar wavelet
        # learnable params
        alpha = self.RNG.rand(size=(1))
        beta = self.RNG.rand(size=(1))
        gamma = self.RNG.rand(size=(1))
        delta = self.RNG.rand(size=(1))

    def forward(self, x_data):
        final_A, details = multiscale_learnable_wavelet(X, params_per_scale)




def haar_1d(x_data, alpha, beta, gamma, delta):
    """
    Single-scale learnable Haar transform
    Implements Eq. (5) from the paper.

    Parameters
    ----------
    X : ndarray, shape (T, d)
        Input sequence (T must be even)
    alpha, beta, gamma, delta : ndarray, shape (d,)
        Learnable parameters

    Returns
    -------
    A : ndarray, shape (T//2, d)
        Approximation coefficients
    D : ndarray, shape (T//2, d)
        Detail coefficients
    """
    num_samples, sequence, emebdding = x_data.shape
    assert sequence % 2 == 0, "Sequence length must be even"

    X_even = x_data[0::2]   # x_{2i}
    X_odd  = x_data[1::2]   # x_{2i+1}

    A = alpha * X_even + beta * X_odd
    D = gamma * X_even + delta * X_odd

    return A, D


def haar_multiscale(
    x_data,
    params_per_scale
):
    """
    Multi-scale learnable Haar decomposition
    Implements Eqs. (7)–(8)

    Parameters
    ----------
    X : ndarray, shape (T, d)
    params_per_scale : list of tuples
        [(alpha_l, beta_l, gamma_l, delta_l), ...]

    Returns
    -------
    approximations : ndarray
        Final coarse approximation a^{(L-1)}
    details : list of ndarray
        Detail coefficients at each scale
    """
    details = []
    current = x_data

    for (alpha, beta, gamma, delta) in params_per_scale:
        A, D = haar_1d(
            current, alpha, beta, gamma, delta
        )
        details.append(D)
        current = A

    return current, details


def dwt_1d_axis(x, h, g, axis):
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]

    pad = len(h) - 1
    x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="wrap")

    a = upfirdn(h, x, down=2, axis=-1)
    d = upfirdn(g, x, down=2, axis=-1)

    a = a[..., :n // 2]
    d = d[..., :n // 2]

    return (
        np.moveaxis(a, -1, axis),
        np.moveaxis(d, -1, axis),
    )


def idwt_1d_axis(a, d, hr, gr, axis):
    a = np.moveaxis(a, axis, -1)
    d = np.moveaxis(d, axis, -1)

    x = (
        upfirdn(hr, a, up=2, axis=-1) +
        upfirdn(gr, d, up=2, axis=-1)
    )

    pad = len(hr) - 1
    x = x[..., pad:-pad]

    return np.moveaxis(x, -1, axis)


def dwt_nd(x, h, g, axes=(-2, -1)):
    coeffs = x
    slices = []

    for axis in axes:
        a, d = dwt_1d_axis(coeffs, h, g, axis)
        slices.append((a.shape, d.shape))
        coeffs = np.concatenate([a, d], axis=axis)

    return coeffs


def idwt_nd(x, hr, gr, axes=(-2, -1)):
    coeffs = x

    for axis in reversed(axes):
        n = coeffs.shape[axis]
        a, d = np.split(coeffs, 2, axis=axis)
        coeffs = idwt_1d_axis(a, d, hr, gr, axis)

    return coeffs


# wave packet transforms ---

def wpt_1d_axis(x, h, g, axis, levels):
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]

    blocks = [x]

    for _ in range(levels):
        new_blocks = []
        for b in blocks:
            a, d = dwt_1d_axis(b, h, g, axis=-1)
            new_blocks.extend([a, d])
        blocks = new_blocks

    out = np.concatenate(blocks, axis=-1)
    return np.moveaxis(out, -1, axis)


def iwpt_1d_axis(x, hr, gr, axis, levels):
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    block_size = n >> levels

    blocks = np.split(x, 2 ** levels, axis=-1)

    for _ in range(levels):
        new_blocks = []
        for a, d in zip(blocks[::2], blocks[1::2]):
            new_blocks.append(idwt_1d_axis(a, d, hr, gr, axis=-1))
        blocks = new_blocks

    out = blocks[0]
    return np.moveaxis(out, -1, axis)


def wpt_nd(x, h, g, axes=(-2, -1), levels=1):
    out = x
    for axis in axes:
        out = wpt_1d_axis(out, h, g, axis, levels)
    return out


def iwpt_nd(x, hr, gr, axes=(-2, -1), levels=1):
    out = x
    for axis in reversed(axes):
        out = iwpt_1d_axis(out, hr, gr, axis, levels)
    return out

def wpt_bins(x, levels):
    n = x.shape[-1]
    return x.reshape(*x.shape[:-1], 2**levels, n // 2**levels)

# constraints:
# h = h / (np.linalg.norm(h) + EPSILON)
#