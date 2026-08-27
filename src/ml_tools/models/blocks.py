from numpy.typing import NDArray
import numpy as np
from typing import Callable
from ml_tools.models.constants import GLOBAL_DTYPE, EPSILON
from ml_tools.models.activations import mod_relu, mod_relu_derivative
from ml_tools.models.layers.layers import (
    GLOBAL_DTYPE,
    Layer,
    FullyConnectedLayer,
    FourierLayer,
    NormalizeLayer,
)


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
    """

    def __init__(self,
                 sequence_length: int,
                 hidden_dim: int,
                 num_heads: int = 1,
                 band_radius: int = 0):
        """
        Parameters
        ----------
        sequence_length : tokens per sample, the axis the FFT runs over
        hidden_dim : channel width of the input
        num_heads : gates learned in parallel, each over its own slice of the
            channel axis. 1 recovers the single-head layer exactly.
        band_radius : radius r of the optional Toeplitz band update on the
            gate. 0 disables it. r > 0 adds 2r+1 complex taps per head.
        """
        super().__init__()
        assert num_heads >= 1, f"num_heads must be at least 1, got {num_heads}"
        assert hidden_dim % num_heads == 0, (
            f"num_heads {num_heads} must divide hidden_dim {hidden_dim}. Heads "
            "partition the channel axis, so a remainder would leave channels "
            "ungated."
        )

        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # declared on the channel axis only, matching every other layer, so the
        # graph's right-aligned edge check reads it. The sequence length is
        # still fixed -- the gate has one entry per frequency of a fixed length
        # transform -- but that is enforced in forward, where the data arrives.
        self.declare_shapes(inputs=((hidden_dim,),), outputs=((hidden_dim,),))

        self.num_frequencies = sequence_length // 2 + 1
        self.activation_bias = np.zeros(
            (num_heads, self.num_frequencies), dtype=GLOBAL_DTYPE
        ) - 0.1

        self.fc_query = FullyConnectedLayer(ni=hidden_dim, no=hidden_dim, activation_type="linear")
        self.fc_values = FullyConnectedLayer(ni=hidden_dim, no=hidden_dim, activation_type="linear")

        # LN over the feature axis of the pooled query, per the paper
        self.norm_query = NormalizeLayer(ni=hidden_dim, shift_scale=False)

        self.fc_1 = FullyConnectedLayer(ni=hidden_dim, no=hidden_dim, activation_type="relu")
        self.fc_2 = FullyConnectedLayer(
            ni=hidden_dim,
            no=2 * num_heads * self.num_frequencies,
            activation_type="linear",
        )

        assert band_radius >= 0, (
            f"band_radius must be zero or positive, got {band_radius}. A "
            "negative radius produces no taps and silently disables the gate."
        )

        self.band_radius = band_radius
        self.band_offsets = tuple(range(-band_radius, band_radius + 1))
        if band_radius:
            # depth-wise, so the taps are per head and never mix across heads
            self.band_taps = np.zeros(
                (num_heads, len(self.band_offsets)), dtype=np.complex128
            )

        self.activation: Callable = mod_relu
        self.activation_derivative: Callable = mod_relu_derivative

        # seeded here so a gradient read before the first backward finds zeros
        # rather than raising AttributeError
        self.zero_gradients()

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

    def _split_heads(self, spectrum: NDArray) -> NDArray:
        """(batch, frequency, hidden) -> (batch, frequency, head, head_dim)"""
        return spectrum.reshape(
            *spectrum.shape[:2], self.num_heads, self.head_dim
        )

    def _merge_heads(self, spectrum: NDArray) -> NDArray:
        """(batch, frequency, head, head_dim) -> (batch, frequency, hidden)"""
        return spectrum.reshape(*spectrum.shape[:2], self.hidden_dim)

    @staticmethod
    def _align_gate(gate: NDArray) -> NDArray:
        """
        (batch, head, frequency) -> (batch, frequency, head, 1)

        The gate is built with frequency last, because the band convolution and
        the modReLU bias both run along that axis, but the values carry
        frequency second. This is the transpose between the two.
        """
        return np.transpose(gate, (0, 2, 1))[..., None]

    def _band_update(self, gate: NDArray) -> NDArray:
        """
        Toeplitz band update from the paper, g <- g + (t * g), where * is a
        convolution along the frequency axis with 2r+1 complex taps per head.
        """
        banded = np.zeros_like(gate)
        for index, offset in enumerate(self.band_offsets):
            banded += self.band_taps[:, index, None] * self._shift(gate, offset)
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

        # two layer MLP to the complex gate, one gate per head
        gate_projection = self.fc_2(self.fc_1(descriptor))
        g_real, g_imag = np.split(gate_projection, 2, axis=-1)
        self.gate_raw = (g_real + 1j * g_imag).reshape(
            -1, self.num_heads, self.num_frequencies
        )

        self.gate_activated = self.activation(self.gate_raw, self.activation_bias)

        if self.band_radius:
            self.gate = self._band_update(self.gate_activated)
        else:
            self.gate = self.gate_activated

        # diagonal gating, one scalar per frequency across each head's channels
        values_gated = self._split_heads(self.value_transform) * self._align_gate(
            self.gate
        )

        self.output = np.fft.irfft(
            self._merge_heads(values_gated), n=self.sequence_length, axis=1
        )
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        sequence = incoming_gradient.shape[1]

        dvalues_gated = self._split_heads(
            irfft_adjoint(incoming_gradient, self.sequence_length, axis=1)
        )
        value_heads = self._split_heads(self.value_transform)

        # gating is elementwise complex, so each side picks up the other's conjugate
        dV_hat = self._merge_heads(
            dvalues_gated * np.conj(self._align_gate(self.gate))
        )
        # the gate is shared across a head's channels, so its gradient sums over them
        dgate = np.transpose(
            np.sum(dvalues_gated * np.conj(value_heads), axis=-1), (0, 2, 1)
        )

        if self.band_radius:
            self.gradient_band = np.stack([
                np.sum(
                    dgate * np.conj(self._shift(self.gate_activated, offset)),
                    axis=(0, 2),
                )
                for offset in self.band_offsets
            ], axis=-1)
            dgate_activated = dgate + sum(
                np.conj(self.band_taps[:, index, None])
                * self._shift(dgate, -offset)
                for index, offset in enumerate(self.band_offsets)
            )
        else:
            dgate_activated = dgate

        self.gradient_bias, dgate_raw = self.activation_derivative(
            self.gate_raw, self.activation_bias, dgate_activated
        )

        batch = dgate_raw.shape[0]
        dgate_projection = np.concatenate(
            [dgate_raw.real.reshape(batch, -1), dgate_raw.imag.reshape(batch, -1)],
            axis=-1,
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

    def get_weights(self) -> tuple:
        weights = [
            self.activation_bias,
            self.fc_query.get_weights(),
            self.fc_values.get_weights(),
            self.fc_1.get_weights(),
            self.fc_2.get_weights(),
        ]
        if self.band_radius:
            weights.append(self.band_taps)
        return tuple(weights)

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
            f"hidden {self.hidden_dim}, {self.num_heads} heads{band}"
        )

    def __repr__(self):
        return self.__str__()


class CausalSpectreAttention(SpectreAttention):
    """
    causal SPECTRE with the Prefix-FFT cache, https://arxiv.org/abs/2502.18394

    Forward is the parallel teacher-forced path, and is exact rather than an
    approximation of the incremental one -- position t is reconstructed from
    the prefix ending at t, which is the diagonal of the inverse transform.
    Unlike the encoder it accepts any length up to the transform length, since
    a shorter prefix is just the zero padded one. prefill and decode_step run
    the same arithmetic a token at a time for generation.
    """

    def __init__(self,
                 sequence_length: int,
                 hidden_dim: int,
                 num_heads: int = 1,
                 band_radius: int = 0):
        super().__init__(sequence_length, hidden_dim, num_heads, band_radius)

        positions = np.arange(sequence_length)[:, None]
        frequencies = np.arange(self.num_frequencies)[None, :]
        self.twiddle = np.exp(
            -2j * np.pi * frequencies * positions / sequence_length
        )

        # irfft folds the conjugate pairs back in
        hermitian = np.full(self.num_frequencies, 2.0)
        hermitian[0] = 1.0
        if sequence_length % 2 == 0:  # evens == 1
            hermitian[-1] = 1.0
        self.hermitian = hermitian
        self.inverse_basis = hermitian * np.conj(self.twiddle) / sequence_length

        self.prefix_spectrum = None
        self.prefix_counts = None
        self.reset_cache()

    # -------------    the gate, shared by both paths    ---------------
    def _spectral_gate(self, descriptor: NDArray) -> NDArray:
        """
        descriptor (..., hidden)
        gate (..., head, frequency)
        """
        leading = descriptor.shape[:-1]
        projection = self.fc_2(self.fc_1(descriptor))
        real_part, imaginary_part = np.split(projection, 2, axis=-1)

        self.gate_raw = (real_part + 1j * imaginary_part).reshape(
            -1, self.num_heads, self.num_frequencies
        )
        self.gate_activated = self.activation(self.gate_raw, self.activation_bias)
        self.gate = (
            self._band_update(self.gate_activated)
            if self.band_radius
            else self.gate_activated
        )
        return self.gate.reshape(*leading, self.num_heads, self.num_frequencies)

    def _spectral_gate_backward(self, dgate: NDArray) -> NDArray:
        """
        d_gate (..., head, frequency)
        d_descriptor (..., hidden)
        """
        leading = dgate.shape[:-2]
        dgate = dgate.reshape(-1, self.num_heads, self.num_frequencies)

        if self.band_radius:
            self.gradient_band = np.stack([
                np.sum(
                    dgate * np.conj(self._shift(self.gate_activated, offset)),
                    axis=(0, 2),
                )
                for offset in self.band_offsets
            ], axis=-1)
            dgate = dgate + sum(
                np.conj(self.band_taps[:, index, None])
                * self._shift(dgate, -offset)
                for index, offset in enumerate(self.band_offsets)
            )

        self.gradient_bias, dgate_raw = self.activation_derivative(
            self.gate_raw, self.activation_bias, dgate
        )

        rows = dgate_raw.shape[0]
        width = self.num_heads * self.num_frequencies
        dprojection = np.concatenate(
            [dgate_raw.real.reshape(rows, width),
             dgate_raw.imag.reshape(rows, width)],
            axis=-1,
        ).reshape(*leading, 2 * width)

        return self.fc_1.backward(self.fc_2.backward(dprojection))

    # -------------    prefix axis helpers    -------------------------
    def _split_prefix(self, spectrum: NDArray) -> NDArray:
        """
        (batch, position, frequency, hidden)
        returns
        (..., head, head_dim)
        """
        return spectrum.reshape(
            *spectrum.shape[:3], self.num_heads, self.head_dim
        )

    def _merge_prefix(self, spectrum: NDArray) -> NDArray:
        """
        (batch, position, frequency, head, head_dim)
        returns
        (..., hidden)
        """
        return spectrum.reshape(*spectrum.shape[:3], self.hidden_dim)

    @staticmethod
    def _align_prefix_gate(gate: NDArray) -> NDArray:
        """
        (batch, position, head, frequency)
        Returns
        (batch, position, frequency, head, 1)
        """
        return np.transpose(gate, (0, 1, 3, 2))[..., None]

    # -------------    the parallel path
    def _run_prefix(self, input_data: NDArray) -> tuple:
        """
        Every position's causal output with the running state the
        cache needs
        forward and prefill differ only in what they retain/pass along
        """
        length = input_data.shape[1]
        query_forward = self.fc_query(input_data)
        value_forward = self.fc_values(input_data)

        spectrum = np.cumsum(
            value_forward[:, :, None, :] * self.twiddle[:length][None, :, :, None],
            axis=1,
        )
        counts = np.arange(1, length + 1, dtype=np.float64)
        query_cumulative = np.cumsum(query_forward, axis=1)

        gate = self._spectral_gate(
            self.norm_query(query_cumulative / counts[None, :, None])
        )
        gated = self._merge_prefix(
            self._split_prefix(spectrum) * self._align_prefix_gate(gate)
        )

        # only the diagonal is wanted: position t rebuilt from the prefix that
        # ends at t (causal contract)
        outputs = np.real(
            np.sum(gated * self.inverse_basis[:length][None, :, :, None], axis=2)
        )
        return outputs, spectrum, query_cumulative, counts

    def forward(self, input_data: NDArray) -> NDArray:
        """
        (batch, sequence, hidden)
        teacher forced
        """
        assert input_data.ndim == 3, (
            f"expected (batch, sequence, hidden), got shape {input_data.shape}"
        )
        assert 0 < input_data.shape[1] <= self.sequence_length, (
            f"built for sequence_length {self.sequence_length}, got "
            f"{input_data.shape[1]}. A shorter prefix is fine, since it is the "
            "zero padded one, but the transform length is the ceiling."
        )
        assert input_data.shape[-1] == self.hidden_dim, (
            f"built for hidden_dim {self.hidden_dim}, got {input_data.shape[-1]}"
        )

        self.input = input_data
        (
            self.output,
            self.prefix_spectrum,
            _,
            self.prefix_counts,
        ) = self._run_prefix(input_data)
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        assert self.prefix_spectrum is not None, (
            "backward needs the state forward cached, and decode_step has "
            "since overwritten it. decode_step shares the sub-layers and the "
            "gate with the parallel path and carries no gradient of its own, "
            "so it is inference only. Re-run forward before backward."
        )
        length = incoming_gradient.shape[1]

        # adjoint of reading the diagonal of the inverse transform
        dgated = self._split_prefix(
            incoming_gradient[:, :, None, :]
            * np.conj(self.inverse_basis[:length])[None, :, :, None]
        )
        gate = self.gate.reshape(
            -1, length, self.num_heads, self.num_frequencies
        )
        spectrum_heads = self._split_prefix(self.prefix_spectrum)

        dspectrum = self._merge_prefix(
            dgated * np.conj(self._align_prefix_gate(gate))
        )
        dgate = np.transpose(
            np.sum(dgated * np.conj(spectrum_heads), axis=-1), (0, 1, 3, 2)
        )

        dprefix_mean = self.norm_query.backward(self._spectral_gate_backward(dgate))

        # a prefix sum run backwards : position m feeds every prefix from m on,
        # so its gradient collects everything at or after it.
        dquery = np.cumsum(
            (dprefix_mean / self.prefix_counts[None, :, None])[:, ::-1], axis=1
        )[:, ::-1]
        dtwiddled = np.cumsum(dspectrum[:, ::-1], axis=1)[:, ::-1]
        dvalues = np.real(
            np.sum(
                dtwiddled * np.conj(self.twiddle[:length])[None, :, :, None],
                axis=2,
            )
        )

        return self.fc_query.backward(dquery) + self.fc_values.backward(dvalues)

    # -------------    the Prefix-FFT cache    ------------------------
    def reset_cache(self) -> None:
        """drop the cache, so the next decode starts a fresh sequence"""
        self.cache_spectrum = None
        self.cache_query_sum = None
        self.cache_position = 0

    def prefill(self, input_data: NDArray) -> NDArray:
        outputs, spectrum, query_cumulative, _ = self._run_prefix(input_data)

        self.cache_spectrum = spectrum[:, -1].copy()
        self.cache_query_sum = query_cumulative[:, -1].copy()
        self.cache_position = input_data.shape[1]
        return outputs

    def decode_step(self, token: NDArray) -> NDArray:
        """
        (batch, hidden), or (batch, 1, hidden), the newest / next position only

        Returns that position's output at (batch, hidden)

        inference only: this reuses the sub-layers' forward caches, so it
        overwrites whatever the last training forward left there.
        """
        if token.ndim == 3:
            assert token.shape[1] == 1, (
                f"decode_step takes one position, got {token.shape[1]}. Use "
                "prefill for a context."
            )
            token = token[:, 0]
        assert token.shape[-1] == self.hidden_dim, (
            f"built for hidden_dim {self.hidden_dim}, got {token.shape[-1]}"
        )

        position = self.cache_position
        assert position < self.sequence_length, (
            f"the cache is full at {self.sequence_length} positions. The gate "
            "is tied to a fixed transform length, so generation cannot run "
            "past it."
        )

        if self.cache_spectrum is None:
            self.cache_spectrum = np.zeros(
                (token.shape[0], self.num_frequencies, self.hidden_dim),
                dtype=np.complex128,
            )
            self.cache_query_sum = np.zeros(
                (token.shape[0], self.hidden_dim), dtype=np.float64
            )

        # the sub-layers and the gate are shared with the parallel path, so
        # stepping here retires whatever the last forward left for backward
        self.prefix_spectrum = None

        self.cache_query_sum = self.cache_query_sum + self.fc_query(token)
        self.cache_spectrum = self.cache_spectrum + (
            self.fc_values(token)[:, None, :]
            * self.twiddle[position][None, :, None]
        )
        self.cache_position = position + 1

        gate = self._spectral_gate(
            self.norm_query(self.cache_query_sum / self.cache_position)
        )
        gated = self._merge_heads(
            self._split_heads(self.cache_spectrum) * self._align_gate(gate)
        )
        return np.real(
            np.sum(gated * self.inverse_basis[position][None, :, None], axis=1)
        )

    def purge(self) -> None:
        super().purge()
        self.prefix_spectrum = None
        self.prefix_counts = None
        self.reset_cache()

    def __str__(self):
        band = f", band radius {self.band_radius}" if self.band_radius else ""
        return (
            f"causal SPECTRE mixer with prefix-FFT cache, sequence "
            f"{self.sequence_length}, hidden {self.hidden_dim}, "
            f"{self.num_heads} heads{band}"
        )


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
