import jax.numpy as jnp
import numpy as np
import copy
# from scipy
from layers import Layer

# https://wandb.ai/sauravmaheshkar/FNet/reports/Fourier-Transform-in-Neural-Networks---Vmlldzo4MzMyNDc
# https://github.com/SauravMaheshkar/FNet-Flax/blob/main/fnet_flax/model.py

EPSILON = 1e-12
"""
x.shape will be num_data points, embeded dimension?
Replacing Self-attention mechanism with Fourier Transform embeddings
"""

def layer_norm(x, axis=(0, 1)):
    # zero mean and variance 1
    # TODO: axis - tuple? two dimensions enough?? what if we have > 1D values? (I can't think why it would be multiD)
    # persist as self. gamma / sigma or whatever.
    _var = np.var(x, axis=axis, keepdims=True)
    _mean = np.mean(x, axis=axis, keepdims=True)
    _x = (x - _mean) / (np.sqrt(_var + EPSILON))
    return _x


def fourier_embed(x):
    """ """
    fourier_x = np.fft.fft2(x, axes=(-1, -2))
    # fourier_x = jnp.fft.fft2(x, axes=(-1, -2))
    return layer_norm(x + fourier_x)


class FourierEncoder(Layer):
    """"""
    def __init__(self, ni, no):
        self.
        fourier_x = np.fft.fft2(x, axes=(-1, -2))
        # (x + fourier_x)
        # jnp.fft.fft2(x, axis=)

    def forward(self, x):
        # fourier transform of x
        fourier_x = np.fft.fft2(x, axes=(-1, -2))
        # add existing x to fourier transform
        normalized_fourier_x = norm (x + fourier_x)
        return normalized_fourier_x

        # nn_fourier_x - Feed forward with ff layer (norm_fourier_x)
        # embedded_x = Layer Normalize (x + nn_fourier_x)
        # return embedded_x

        pass

