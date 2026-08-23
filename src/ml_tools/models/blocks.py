from numpy.typing import NDArray
import numpy as np
from typing import Callable
from ml_tools.models.activations import mod_relu, mod_relu_derivative
from ml_tools.models.layers.layers import (
    GLOBAL_DTYPE,
    Layer,
    FullyConnectedLayer,
    FourierLayer,
    NormalizeLayer,
)

EPSILON = 1e-15


# -------------    adjoints of the real FFT pair    ----------------
def rfft_adjoint(grad_freq: NDArray, sequence_length: int, axis: int = 1) -> NDArray:
    """
    adjoint of np.fft.rfft. Only the non-negative frequencies are carried, so
    the half spectrum is zero-padded back to full length before inverting.
    """
    shape = list(grad_freq.shape)
    shape[axis] = sequence_length
    padded = np.zeros(shape, dtype=np.complex128)

    slicer = [slice(None)] * grad_freq.ndim
    slicer[axis] = slice(0, grad_freq.shape[axis])
    padded[tuple(slicer)] = grad_freq

    return np.fft.ifft(padded, axis=axis).real * sequence_length


def irfft_adjoint(grad_time: NDArray, sequence_length: int, axis: int = 1) -> NDArray:
    """
    adjoint of np.fft.irfft. numpy folds 1/n into the inverse transform, so the
    adjoint carries it too.

    Paired bins double, because each one stands for a conjugate pair under
    Hermitian symmetry. DC is never paired. Nyquist exists, and is likewise
    unpaired, only when the sequence length is even -- for an odd length the
    last bin is paired and must double as well.
    """
    out = np.fft.rfft(grad_time, axis=axis) / sequence_length

    nyquist_present = sequence_length % 2 == 0
    slicer = [slice(None)] * out.ndim
    slicer[axis] = slice(1, -1 if nyquist_present else None)
    out[tuple(slicer)] *= 2

    return out


# FNet, https://arxiv.org/abs/2105.03824
class FourierAttention(Layer):
    def __init__(self, ni: int, no: int, use_2d: bool = True):
        super().__init__()
        assert ni == no, (
            f"the feed forward residual needs matching widths, got ni={ni} no={no}"
        )
        self.fftlayer = FourierLayer(use_2d)
        self.norm_a = NormalizeLayer(ni=ni, shift_scale=False)
        self.fc = FullyConnectedLayer(ni=ni, no=no, activation_type="relu")
        self.norm_b = NormalizeLayer(ni=no, shift_scale=True)
        self.declare_shapes(inputs=((ni,),), outputs=((no,),))

    def forward(self, x_data: NDArray):
        if self.fftlayer.use_2d:
            assert x_data.ndim >= 3, (
                "use_2d mixes over the last two axes, which on a 2D "
                f"(batch, hidden) input means mixing across the batch and "
                f"leaking between samples. Got shape {x_data.shape}; pass "
                "(batch, sequence, hidden) or use use_2d=False."
            )

        fft_x = self.norm_a(self.fftlayer(x_data) + x_data)

        self.output = self.norm_b(self.fc(fft_x) + fft_x)

        return self.output

    def backward(self, incoming_gradient: NDArray):
        grad = self.norm_b.backward(incoming_gradient)
        # residual connections
        grad_fc_out = grad
        grad_skip_b = grad
        # ---- fully connected ----
        grad = self.fc.backward(grad_fc_out)
        # accumulate skip connection
        grad = grad + grad_skip_b

        grad = self.norm_a.backward(grad)
        # second residual connections
        grad_fft = grad
        grad_skip_a = grad
        # ---- FFT Layer ----
        grad = self.fftlayer.backward(grad_fft)

        # accumulate skip connection
        grad = grad + grad_skip_a

        self.gradient = grad
        return grad

    def purge(self):
        self.fftlayer.purge()
        self.norm_a.purge()
        self.fc.purge()
        self.norm_b.purge()

    def get_weights(self) -> tuple[NDArray]:
        return (
            self.norm_a.get_weights(),
            self.fc.get_weights(),
            self.norm_b.get_weights(),
        )

    def get_gradients(self) -> dict[str, NDArray] | None:
        return {
            "norm_a": self.norm_a.get_gradients(),
            "fc": self.fc.get_gradients(),
            "norm_b": self.norm_b.get_gradients(),
        }

    def zero_gradients(self):
        self.norm_a.zero_gradients()
        self.fc.zero_gradients()
        self.norm_b.zero_gradients()

    @property
    def num_parameters(self) -> int:
        return (
            self.norm_a.num_parameters
            + self.fc.num_parameters
            + self.norm_b.num_parameters
        )

    def update_weights(
        self,
        norm_a: dict[str, NDArray],
        fc: dict[str, NDArray],
        norm_b: dict[str, NDArray],
    ) -> None:
        self.norm_a.update_weights(**norm_a)
        self.fc.update_weights(**fc)
        self.norm_b.update_weights(**norm_b)


class SpectreAttention(Layer):
    """
    SPECTRE mixing layer, https://arxiv.org/abs/2502.18394

    Query and value projections only (the paper defines no key), a real FFT of
    the values along the sequence axis, a content-adaptive diagonal spectral
    gate driven by the sequence-pooled query, then an inverse real FFT. The
    gate is per frequency and broadcasts across every channel.

    Deviations from the paper, both deliberate:
      - single head. The paper is per-head with heads concatenated.
      - no positional phase. The paper only defines the phase rotation
        exp(j 2 pi k t / n) for its decode path, where t is the absolute
        decode step. It specifies nothing for the parallel training path,
        so nothing is applied here.

    Normalisation follows the paper: the forward rfft is unnormalised and the
    1/n sits on the inverse only, which is numpy's default convention.
    """

    def __init__(self, sequence_length: int, hidden_dim: int, band_radius: int = 0):
        """
        Parameters
        ----------
        sequence_length : tokens per sample, the axis the FFT runs over
        hidden_dim : channel width of the input
        band_radius : radius r of the optional Toeplitz band update on the
            gate. 0 disables it. r > 0 adds 2r+1 complex taps.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        # the gate holds one entry per frequency of a fixed length transform,
        # so the sequence axis is pinned here rather than left free
        self.declare_shapes(
            inputs=((sequence_length, hidden_dim),),
            outputs=((sequence_length, hidden_dim),),
        )

        self.num_frequencies = sequence_length // 2 + 1
        self.activation_bias = np.zeros(self.num_frequencies, dtype=GLOBAL_DTYPE) - 0.1

        self.fc_query = FullyConnectedLayer(ni=hidden_dim, no=hidden_dim, activation_type="linear")
        self.fc_values = FullyConnectedLayer(ni=hidden_dim, no=hidden_dim, activation_type="linear")

        # LN over the feature axis of the pooled query, per the paper
        self.norm_query = NormalizeLayer(ni=hidden_dim, shift_scale=False)

        self.fc_1 = FullyConnectedLayer(ni=hidden_dim, no=hidden_dim, activation_type="relu")
        self.fc_2 = FullyConnectedLayer(ni=hidden_dim, no=2 * self.num_frequencies, activation_type="linear")

        assert band_radius >= 0, (
            f"band_radius must be zero or positive, got {band_radius}. A "
            "negative radius produces no taps and silently disables the gate."
        )

        self.band_radius = band_radius
        self.band_offsets = tuple(range(-band_radius, band_radius + 1))
        if band_radius:
            self.band_taps = np.zeros(len(self.band_offsets), dtype=np.complex128)

        self.activation: Callable = mod_relu
        self.activation_derivative: Callable = mod_relu_derivative

    @staticmethod
    def _shift(array: NDArray, offset: int) -> NDArray:
        """shift along the frequency axis, zero filled rather than circular"""
        out = np.zeros_like(array)
        if offset > 0:
            out[..., offset:] = array[..., :-offset]
        elif offset < 0:
            out[..., :offset] = array[..., -offset:]
        else:
            out[...] = array
        return out

    def _band_update(self, gate: NDArray) -> NDArray:
        """
        Toeplitz band update from the paper, g <- g + (t * g), where * is a
        convolution along the frequency axis with 2r+1 complex taps.
        """
        banded = np.zeros_like(gate)
        for tap, offset in zip(self.band_taps, self.band_offsets):
            banded += tap * self._shift(gate, offset)
        return gate + banded

    def forward(self, input_data: NDArray):
        """
        input_data: (batch, sequence, hidden)
        """
        assert input_data.ndim == 3, (
            f"expected (batch, sequence, hidden), got shape {input_data.shape}"
        )
        assert input_data.shape[1] == self.sequence_length, (
            f"built for sequence_length {self.sequence_length}, got "
            f"{input_data.shape[1]}. The gate has one entry per frequency of a "
            "fixed length transform, so the sequence length cannot vary."
        )
        assert input_data.shape[-1] == self.hidden_dim, (
            f"built for hidden_dim {self.hidden_dim}, got {input_data.shape[-1]}"
        )

        self.input = input_data

        query_forward = self.fc_query(input_data)
        value_forward = self.fc_values(input_data)

        # rfft along the SEQUENCE axis, cached for the backward pass
        self.value_transform = np.fft.rfft(value_forward, axis=1)

        # pool the query over the sequence, then LN over the feature axis
        self.seq_mu = np.mean(query_forward, axis=1)
        descriptor = self.norm_query(self.seq_mu)

        # two layer MLP to the complex gate
        gate_projection = self.fc_2(self.fc_1(descriptor))
        g_real, g_imag = np.split(gate_projection, 2, axis=-1)
        self.gate_raw = g_real + 1j * g_imag

        self.gate_activated = self.activation(self.gate_raw, self.activation_bias)

        if self.band_radius:
            self.gate = self._band_update(self.gate_activated)
        else:
            self.gate = self.gate_activated

        # diagonal spectral gating, one scalar per frequency across all channels
        values_gated = self.value_transform * self.gate[..., None]

        self.output = np.fft.irfft(values_gated, n=self.sequence_length, axis=1)
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        sequence = incoming_gradient.shape[1]

        dvalues_gated = irfft_adjoint(
            incoming_gradient, self.sequence_length, axis=1
        )

        # gating is elementwise complex, so each side picks up the other's conjugate
        dV_hat = dvalues_gated * np.conj(self.gate)[..., None]
        dgate = np.sum(dvalues_gated * np.conj(self.value_transform), axis=2)

        if self.band_radius:
            self.gradient_band = np.array([
                np.sum(dgate * np.conj(self._shift(self.gate_activated, offset)))
                for offset in self.band_offsets
            ])
            dgate_activated = dgate + sum(
                np.conj(tap) * self._shift(dgate, -offset)
                for tap, offset in zip(self.band_taps, self.band_offsets)
            )
        else:
            dgate_activated = dgate

        self.gradient_bias, dgate_raw = self.activation_derivative(
            self.gate_raw, self.activation_bias, dgate_activated
        )

        dgate_projection = np.concatenate(
            [dgate_raw.real, dgate_raw.imag], axis=-1
        )

        dhidden = self.fc_2.backward(dgate_projection)
        ddescriptor = self.fc_1.backward(dhidden)
        dseq_mu = self.norm_query.backward(ddescriptor)

        # the pool was a mean, so the gradient spreads evenly back over the sequence
        dquery = np.repeat(dseq_mu[:, None, :], sequence, axis=1) / sequence

        dinput_from_q = self.fc_query.backward(dquery)

        dvalues = rfft_adjoint(dV_hat, self.sequence_length, axis=1)
        dinput_from_v = self.fc_values.backward(dvalues)

        return dinput_from_q + dinput_from_v

    def get_gradients(self) -> dict[str, NDArray | dict]:
        gradients = {
            "gradient_bias": self.gradient_bias,
            "fc_query": self.fc_query.get_gradients(),
            "fc_values": self.fc_values.get_gradients(),
            "fc_1": self.fc_1.get_gradients(),
            "fc_2": self.fc_2.get_gradients(),
        }
        if self.band_radius:
            gradients["gradient_band"] = self.gradient_band
        return gradients

    def purge(self) -> None:
        self.input = None
        self.value_transform = None
        self.seq_mu = None
        self.gate_raw = None
        self.gate_activated = None
        self.gate = None
        self.output = None
        self.norm_query.purge()
        self.fc_query.purge()
        self.fc_values.purge()
        self.fc_1.purge()
        self.fc_2.purge()

    def zero_gradients(self):
        self.gradient_bias = np.zeros_like(self.activation_bias)
        if self.band_radius:
            self.gradient_band = np.zeros_like(self.band_taps)
        for layer in (self.fc_query, self.fc_values, self.fc_1, self.fc_2):
            layer.zero_gradients()

    @property
    def num_parameters(self) -> int:
        total = (
            self.activation_bias.size
            + self.fc_query.num_parameters
            + self.fc_values.num_parameters
            + self.fc_1.num_parameters
            + self.fc_2.num_parameters
        )
        if self.band_radius:
            total += 2 * self.band_taps.size
        return total

    def update_weights(
            self,
            gradient_bias: NDArray,
            fc_query: dict[str, NDArray],
            fc_values: dict[str, NDArray],
            fc_1: dict[str, NDArray],
            fc_2: dict[str, NDArray],
            gradient_band: NDArray = None,
    ) -> None:
        self.activation_bias -= gradient_bias
        self.fc_query.update_weights(**fc_query)
        self.fc_values.update_weights(**fc_values)
        self.fc_1.update_weights(**fc_1)
        self.fc_2.update_weights(**fc_2)
        if gradient_band is not None:
            self.band_taps -= gradient_band

    def __str__(self):
        band = f", band radius {self.band_radius}" if self.band_radius else ""
        return (
            f"SPECTRE mixer, sequence {self.sequence_length}, "
            f"hidden {self.hidden_dim}{band}"
        )

    def __repr__(self):
        return self.__str__()



if __name__ == "__main__":
    from ml_tools.models.optimizers import SGD
    from ml_tools.models.model_loss import MSELoss
    import matplotlib.pyplot as plt
    from ml_tools.generators.periodic_signal_gen import (
        make_multifreq_dataset,
        make_phase_mix_dataset
    )


    # x, y = make_phase_mix_dataset(50, 128, 3, 11)
    # loss = MSELoss()
    # optimizer = SGD(2e-4)
    #
    # attn = FourierAttention(ni=128, no=128, use_2d=True)
    # # attn2 = FourierAttention(ni=512, no=512, use_2d=True)
    # # attn3 = FourierAttention(ni=512, no=512, use_2d=True)
    #
    # all_loss = []
    # for _ in range(5000):
    #     out = attn.forward(x)
    #     # out = attn3(attn2(attn(x)))
    #     _l = loss.forward(out, y)
    #     if _ % 10 == 0:
    #         print(_l.item())
    #     all_loss.append(_l.item())
    #     grad = loss.backward()
    #     # attn.backward(attn2.backward(attn3.backward(grad)))
    #     # optimizer.step([attn, attn2, attn3])
    #     attn.backward(grad)
    #     optimizer.step([attn])
    #
    # plt.plot(all_loss)
    # plt.show()

    import matplotlib.pyplot as plt

    np.random.seed(0)

    seq_len = 64
    hidden_dim = 128

    x, y = make_multifreq_dataset(batch_size=16, seq_len=seq_len, hidden_dim=hidden_dim)
    loss = MSELoss()
    optimizer = SGD(0.02)
    all_loss = []

    model = SpectreAttention(sequence_length=seq_len, hidden_dim=hidden_dim)

    for _ in range(500):
        out = model.forward(x)
        _l = loss.forward(out, y)

        if _ % 50 == 0:
            print(_l.item())
            plt.plot(y[0, :, 0], label="target")
            plt.plot(out[0, :, 0], label="spectre")
            plt.legend()
            plt.show()
        all_loss.append(_l.item())
        grad = loss.backward()

        _g = model.backward(grad)
        optimizer.step([model])

    plt.plot(all_loss)
    plt.show()
