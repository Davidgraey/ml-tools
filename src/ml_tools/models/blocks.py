from numpy.typing import NDArray
import numpy as np
from typing import Callable
from ml_tools.models.constants import GLOBAL_DTYPE, EPSILON
from ml_tools.models.activations import mod_relu, mod_relu_derivative
from ml_tools.models.draft_heads import CategoricalHead, DraftHead
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

    Residual: the layer returns its input plus the mixing, so it is a block
    rather than a bare transform. CausalSpectreAttention overrides forward and
    backward to run the prefix path and carries the same residual there, so both
    classes behave the same way around their mixing.
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

        # residual around the mixing. The gate is multiplicative in the
        # frequency domain, so without it a head that closes drops its channels
        # entirely rather than passing them through
        self.output = input_data + np.fft.irfft(
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

        # the residual's own path: the identity carries the gradient straight
        # through alongside the two projections
        return dinput_from_q + dinput_from_v + incoming_gradient

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

    Residual, as in the parent: the layer returns its input plus the mixing.
    It sits inside _run_prefix and decode_step rather than in forward, so all
    three entry points agree on what the layer's output is.

    The drafting path in DFlashSpectreAttention has its own, around the block
    MLP. Its mixer slots are learned constants rather than positions of an input
    sequence, so there is no input there to carry forward.
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
        # ends at t (causal contract). The residual rides here rather than in
        # forward so that prefill returns the same thing forward would
        outputs = input_data + np.real(
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

        # the residual's own path, matching the parallel layer's
        return (
            self.fc_query.backward(dquery)
            + self.fc_values.backward(dvalues)
            + incoming_gradient
        )

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
        # the same residual the parallel path applies, or stepping the cache
        # would disagree with forward at the position it just consumed
        return token + np.real(
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


class DynamicCausalConv:
    """
    Grouped causal convolution with a content-dependent kernel, the local
    mixing DFlash2 wraps around a draft sublayer.

    One projection of the sublayer's input produces two kernels, one for the
    input and one for the output. Each is a learned base kernel shared across
    positions plus a dynamic part that all channels in a group share, so the
    kernel adapts to content without paying a projection per channel.

    Causal and block local. The taps reach backwards only, and zero fill at the
    start rather than wrapping, so a draft block never convolves in the block
    before it.

    Not a Layer: it is used in two phases around a sublayer rather than in one
    shot, so it does not honour the single forward, single backward contract.
    """

    def __init__(self, hidden_dim: int, kernel_size: int = 2, group_size: int = 16):
        assert kernel_size >= 1, f"need at least one tap, got {kernel_size}"
        assert hidden_dim % group_size == 0, (
            f"conv_group_size {group_size} must divide hidden_dim {hidden_dim}. "
            "Groups partition the channel axis, so a remainder would leave "
            "channels without a dynamic kernel."
        )
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.num_groups = hidden_dim // group_size

        # identity at initialisation, so the wrapper starts as a no-op and the
        # drafter is not handed a scrambled block before it has learned anything
        self.base_input = np.zeros((kernel_size, hidden_dim), dtype=GLOBAL_DTYPE)
        self.base_output = np.zeros((kernel_size, hidden_dim), dtype=GLOBAL_DTYPE)
        self.base_input[0] = 1.0
        self.base_output[0] = 1.0

        self.fc_kernel = FullyConnectedLayer(
            ni=hidden_dim,
            no=2 * kernel_size * self.num_groups,
            activation_type="linear",
        )
        self.zero_gradients()

    @staticmethod
    def _lag(array: NDArray, tap: int) -> NDArray:
        """x[t - tap] along the position axis, zero filled at the block start"""
        out = np.zeros_like(array)
        if tap == 0:
            out[...] = array
        else:
            out[:, tap:] = array[:, :-tap]
        return out

    @staticmethod
    def _lead(array: NDArray, tap: int) -> NDArray:
        """the adjoint of _lag, x[t + tap], zero filled at the block end"""
        out = np.zeros_like(array)
        if tap == 0:
            out[...] = array
        else:
            out[:, :-tap] = array[:, tap:]
        return out

    def _expand(self, dynamic: NDArray) -> NDArray:
        """(batch, position, tap, group) -> (batch, position, tap, hidden)"""
        return np.repeat(dynamic, self.group_size, axis=-1)

    def _convolve(self, data: NDArray, base: NDArray, dynamic: NDArray) -> NDArray:
        kernel = base[None, None] + self._expand(dynamic)
        return sum(
            kernel[:, :, tap] * self._lag(data, tap)
            for tap in range(self.kernel_size)
        )

    def _convolve_backward(
        self, incoming: NDArray, data: NDArray, base: NDArray, dynamic: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        kernel = base[None, None] + self._expand(dynamic)
        taps = range(self.kernel_size)

        dkernel = np.stack(
            [incoming * self._lag(data, tap) for tap in taps], axis=2
        )
        dbase = dkernel.sum(axis=(0, 1))
        ddynamic = dkernel.reshape(
            *dkernel.shape[:3], self.num_groups, self.group_size
        ).sum(axis=-1)
        ddata = sum(
            self._lead(kernel[:, :, tap] * incoming, tap) for tap in taps
        )
        return ddata, dbase, ddynamic

    def forward_input(self, data: NDArray) -> NDArray:
        """convolve the sublayer's input, and derive both kernels from it"""
        self.input = data
        width = self.kernel_size * self.num_groups
        projection = self.fc_kernel(data)
        self.dynamic_input = projection[..., :width].reshape(
            *data.shape[:-1], self.kernel_size, self.num_groups
        )
        self.dynamic_output = projection[..., width:].reshape(
            *data.shape[:-1], self.kernel_size, self.num_groups
        )
        return self._convolve(data, self.base_input, self.dynamic_input)

    def forward_output(self, data: NDArray) -> NDArray:
        """convolve the sublayer's output, reusing the kernels already derived"""
        self.sublayer_output = data
        return self._convolve(data, self.base_output, self.dynamic_output)

    def backward_output(self, incoming: NDArray) -> NDArray:
        """called first, being the later of the two in the forward direction"""
        ddata, self.gradient_base_output, self._ddynamic_output = (
            self._convolve_backward(
                incoming,
                self.sublayer_output,
                self.base_output,
                self.dynamic_output,
            )
        )
        return ddata

    def backward_input(self, incoming: NDArray) -> NDArray:
        """
        Both kernels were projected from this conv's input, so the gradient
        arriving here is joined by the one the output kernel sent back before
        it reaches the projection.
        """
        ddata, self.gradient_base_input, ddynamic_input = self._convolve_backward(
            incoming, self.input, self.base_input, self.dynamic_input
        )

        leading = self.input.shape[:-1]
        dprojection = np.concatenate(
            [
                ddynamic_input.reshape(*leading, -1),
                self._ddynamic_output.reshape(*leading, -1),
            ],
            axis=-1,
        )
        return ddata + self.fc_kernel.backward(dprojection)

    def get_gradients(self) -> dict:
        return {
            "gradient_base_input": self.gradient_base_input,
            "gradient_base_output": self.gradient_base_output,
            "fc_kernel": self.fc_kernel.get_gradients(),
        }

    def update_weights(
        self,
        gradient_base_input: NDArray,
        gradient_base_output: NDArray,
        fc_kernel: dict,
    ) -> None:
        self.base_input -= gradient_base_input
        self.base_output -= gradient_base_output
        self.fc_kernel.update_weights(**fc_kernel)

    def zero_gradients(self) -> None:
        self.gradient_base_input = np.zeros_like(self.base_input)
        self.gradient_base_output = np.zeros_like(self.base_output)
        self._ddynamic_output = None
        self.fc_kernel.zero_gradients()

    def purge(self) -> None:
        self.input = None
        self.sublayer_output = None
        self.dynamic_input = None
        self.dynamic_output = None
        self._ddynamic_output = None
        self.fc_kernel.purge()

    @property
    def num_parameters(self) -> int:
        return (
            self.base_input.size
            + self.base_output.size
            + self.fc_kernel.num_parameters
        )

    def __str__(self):
        return (
            f"dynamic causal conv, {self.kernel_size} taps over "
            f"{self.hidden_dim} channels in groups of {self.group_size}"
        )

    def __repr__(self):
        return self.__str__()


class DFlashSpectreAttention(CausalSpectreAttention):
    """
    DFlash2 style block drafting on a SPECTRE mixer,
    https://arxiv.org/abs/2602.06036

    DFlash drafts a whole block of tokens in one pass instead of stepping a
    small autoregressive model, and conditions that block on features taken
    from the target rather than re-reading the context. Both parts land
    naturally here.

    Where DFlash injects target hidden states into the draft's KV cache, this
    injects the context into the Prefix-FFT cache. The cache already is the
    context, summarised in K complex numbers per channel rather than one
    key-value pair per token, so the draft is conditioned on the full prefix at
    a cost that does not grow with it.

    Where DFlash gives the block a non-causal mask so every slot sees every
    other, all block slots here share one spectrum -- the cached context plus
    every mask slot's contribution -- and differ only in the position they read
    out at. One spectrum, one gate, one pass, and the block is bidirectional
    within itself while the context stays strictly causal.

    From DFlash2: a dynamic grouped causal convolution wraps the block MLP so
    neighbouring slots exchange local information, and a low-rank
    predecessor-conditioned selector reranks the top-k unary candidates into a
    coherent path.

    Two departures worth knowing. The mask slots are parameterised by their
    already-projected values, since a learned vector followed by a fixed linear
    map is just a learned vector. And the convolution wraps only the MLP, not
    the mixer -- the mixer's per-slot input is a learned constant, where a
    convolution would be redundant.
    """

    # the base constructor seeds gradients through zero_gradients, which is
    # overridden here, so it runs once before the draft parts exist. This says
    # whether they do.
    _draft_built: bool = False

    def __init__(self,
                 sequence_length: int,
                 hidden_dim: int,
                 vocab_size: int | None = None,
                 num_heads: int = 1,
                 band_radius: int = 0,
                 block_size: int = 8,
                 conv_kernel_size: int = 2,
                 conv_group_size: int = 16,
                 selector_rank: int = 256,
                 selector_top_k: int = 16,
                 sample_from_anchor: bool = False,
                 head: DraftHead | None = None):
        super().__init__(sequence_length, hidden_dim, num_heads, band_radius)

        assert 1 <= block_size <= sequence_length, (
            f"block_size {block_size} must fit inside the transform length "
            f"{sequence_length}"
        )
        assert (head is None) != (vocab_size is None), (
            "pass either a head or a vocab_size. vocab_size is shorthand for a "
            "CategoricalHead built with the selector arguments here; a head of "
            "your own carries its own."
        )
        self.block_size = block_size

        # the mask slots, held already projected into value and query space
        scale = 1.0 / np.sqrt(hidden_dim)
        self.mask_values = (
            self.RNG.normal(scale=scale, size=(block_size, hidden_dim))
        ).astype(GLOBAL_DTYPE)
        self.mask_query = (
            self.RNG.normal(scale=scale, size=(block_size, hidden_dim))
        ).astype(GLOBAL_DTYPE)

        # built from this block's own stream, so the shorthand initialises
        # exactly as it did when these parameters were declared inline
        if head is None:
            head = CategoricalHead(
                hidden_dim, block_size, vocab_size,
                selector_rank=selector_rank, selector_top_k=selector_top_k,
                sample_from_anchor=sample_from_anchor, rng=self.RNG,
            )
        assert (head.hidden_dim, head.block_size) == (hidden_dim, block_size), (
            f"head is built for hidden {head.hidden_dim} and block "
            f"{head.block_size}, this drafter for {hidden_dim} and {block_size}"
        )
        self.head = head

        self.conv_mlp = DynamicCausalConv(
            hidden_dim, kernel_size=conv_kernel_size, group_size=conv_group_size
        )
        self.fc_block_a = FullyConnectedLayer(hidden_dim, 2 * hidden_dim, "relu")
        self.fc_block_b = FullyConnectedLayer(2 * hidden_dim, hidden_dim, "linear")
        self.fc_unary = FullyConnectedLayer(hidden_dim, head.width, "linear")

        self._draft_built = True
        self.zero_gradients()

    @property
    def vocab_size(self) -> int | None:
        """the head's vocabulary, where it has one"""
        return getattr(self.head, "vocab_size", None)

    # -------------    the block draft    -----------------------------
    def _block_positions(self, anchors: NDArray) -> NDArray:
        """anchor t drafts the block that starts at t + 1"""
        return anchors[:, None] + 1 + np.arange(self.block_size)[None, :]

    def _draft_head(self, block_hidden: NDArray) -> NDArray:
        """
        The per-slot head: a conv wrapped MLP with a residual, then the unary
        projection. Flattened over batch and anchor, so the convolution runs
        along the block axis and stops at its edges.
        """
        self._head_shape = block_hidden.shape
        flat = block_hidden.reshape(-1, self.block_size, self.hidden_dim)

        self._head_input = flat
        pre = self.conv_mlp.forward_input(flat)
        post = self.conv_mlp.forward_output(self.fc_block_b(self.fc_block_a(pre)))
        self._head_output = flat + post

        return self.fc_unary(self._head_output).reshape(
            *self._head_shape[:-1], self.head.width
        )

    def _draft_head_backward(self, dlogits: NDArray) -> NDArray:
        dhidden = self.fc_unary.backward(
            dlogits.reshape(-1, self.block_size, self.head.width)
        )
        dpre = self.fc_block_a.backward(
            self.fc_block_b.backward(self.conv_mlp.backward_output(dhidden))
        )
        dflat = dhidden + self.conv_mlp.backward_input(dpre)
        return dflat.reshape(self._head_shape)

    def draft_forward(self, input_data: NDArray) -> NDArray:
        """
        Teacher forced drafting for training: at every anchor whose block fits,
        draft the whole block in one pass.

        Returns unary logits at (batch, anchor, block_size, vocab). Anchor a
        predicts the tokens at positions a + 1 .. a + block_size.
        """
        assert input_data.ndim == 3, (
            f"expected (batch, sequence, hidden), got {input_data.shape}"
        )
        length = input_data.shape[1]
        anchors = length - self.block_size
        assert anchors >= 1, (
            f"a sequence of {length} leaves no room for a block of "
            f"{self.block_size} after an anchor"
        )

        self.input = input_data
        self.num_anchors = anchors
        query_forward = self.fc_query(input_data)
        value_forward = self.fc_values(input_data)

        self.prefix_spectrum = np.cumsum(
            value_forward[:, :, None, :] * self.twiddle[:length][None, :, :, None],
            axis=1,
        )[:, :anchors]
        query_cumulative = np.cumsum(query_forward, axis=1)[:, :anchors]

        self.block_pos = self._block_positions(np.arange(anchors))
        block_twiddle = self.twiddle[self.block_pos]
        self.block_basis = self.inverse_basis[self.block_pos]

        # every slot contributes to one shared spectrum, which is what makes the
        # block bidirectional within itself
        spectrum = self.prefix_spectrum + np.einsum(
            "jh,ajk->akh", self.mask_values, block_twiddle
        )[None]

        self.prefix_counts = (
            np.arange(1, anchors + 1, dtype=np.float64) + self.block_size
        )
        gate = self._spectral_gate(
            self.norm_query(
                (query_cumulative + self.mask_query.sum(axis=0))
                / self.prefix_counts[None, :, None]
            )
        )

        self.block_spectrum = spectrum
        gated = self._merge_prefix(
            self._split_prefix(spectrum) * self._align_prefix_gate(gate)
        )
        block_hidden = np.real(
            np.einsum("bakh,ajk->bajh", gated, self.block_basis)
        )
        return self._draft_head(block_hidden)

    def draft_backward(self, dlogits: NDArray) -> NDArray:
        assert self.prefix_spectrum is not None, (
            "draft_backward needs the state draft_forward cached, and it has "
            "since been overwritten. Re-run draft_forward."
        )
        dblock = self._draft_head_backward(dlogits)

        dgated = self._split_prefix(
            np.einsum("bajh,ajk->bakh", dblock, np.conj(self.block_basis))
        )
        gate = self.gate.reshape(
            -1, self.num_anchors, self.num_heads, self.num_frequencies
        )
        dspectrum = self._merge_prefix(
            dgated * np.conj(self._align_prefix_gate(gate))
        )
        dgate = np.transpose(
            np.sum(
                dgated * np.conj(self._split_prefix(self.block_spectrum)), axis=-1
            ),
            (0, 1, 3, 2),
        )

        self.gradient_mask_values = np.real(
            np.einsum(
                "bakh,ajk->jh", dspectrum, np.conj(self.twiddle[self.block_pos])
            )
        )

        dmean = self.norm_query.backward(self._spectral_gate_backward(dgate))
        dmean = dmean / self.prefix_counts[None, :, None]
        self.gradient_mask_query = np.repeat(
            dmean.sum(axis=(0, 1))[None], self.block_size, axis=0
        )

        length = self.input.shape[1]
        dquery = np.zeros(self.input.shape, dtype=np.float64)
        dquery[:, : self.num_anchors] = np.cumsum(dmean[:, ::-1], axis=1)[:, ::-1]

        dtwiddled = np.zeros(
            (*self.input.shape[:2], self.num_frequencies, self.hidden_dim),
            dtype=np.complex128,
        )
        dtwiddled[:, : self.num_anchors] = np.cumsum(dspectrum[:, ::-1], axis=1)[
            :, ::-1
        ]
        dvalues = np.real(
            np.sum(
                dtwiddled * np.conj(self.twiddle[:length])[None, :, :, None],
                axis=2,
            )
        )

        return self.fc_query.backward(dquery) + self.fc_values.backward(dvalues)

    def draft_block(self) -> tuple[NDArray, NDArray]:
        """
        One-pass block draft from the Prefix-FFT cache, for generation.

        Returns (unary logits, per-slot hidden) at (batch, block_size, ...).
        The cache is read, never advanced -- verification decides how many of
        these tokens are real, and only then are they absorbed.
        """
        assert self.cache_spectrum is not None, (
            "the cache is empty. Run prefill or decode_step before drafting."
        )
        position = self.cache_position
        assert position + self.block_size <= self.sequence_length, (
            f"a block of {self.block_size} from position {position} runs past "
            f"the transform length {self.sequence_length}"
        )

        block_pos = self._block_positions(np.array([position - 1]))
        spectrum = self.cache_spectrum + np.einsum(
            "jh,ajk->kh", self.mask_values, self.twiddle[block_pos]
        )[None]

        gate = self._spectral_gate(
            self.norm_query(
                (self.cache_query_sum + self.mask_query.sum(axis=0))
                / (position + self.block_size)
            )
        )
        gated = self._merge_heads(
            self._split_heads(spectrum) * self._align_gate(gate)
        )
        block_hidden = np.real(
            np.einsum("bkh,ajk->bjh", gated, self.inverse_basis[block_pos])
        )
        logits = self._draft_head(block_hidden)
        return logits, self._head_output.reshape(block_hidden.shape)

    # -------------    delegated to the head    -----------------------
    def select_path(self,
                    prediction: NDArray,
                    hidden: NDArray,
                    anchor: NDArray) -> NDArray:
        """
        Walk one drafted block into a coherent path.

        What the walk is depends on the head: reranking the unary top-k for a
        vocabulary, refining each slot against its predecessor for real values.
        Either way the draft model ran once and only the walk over block_size
        slots is sequential.
        """
        return self.head.propose(prediction, hidden, anchor)

    def select_forward(self,
                       prediction: NDArray,
                       hidden: NDArray,
                       predecessors: NDArray,
                       targets: NDArray) -> tuple[float, NDArray, NDArray]:
        """
        Teacher forced training of the walk, which is where its sequential
        nature goes away: the predecessor at each slot is taken from the target
        sequence rather than from the walk's own output, so every slot trains in
        parallel.

        Returns the cost and the two upstream gradients. Adding those to the
        unary head's own trains the two jointly; discarding them trains the
        walk alongside a unary head that stays exactly a decoder's.
        """
        return self.head.refine_forward(prediction, hidden, predecessors, targets)

    def propose(self, anchor: NDArray) -> NDArray:
        """draft a block off the cache and walk it into a path"""
        prediction, hidden = self.draft_block()
        return self.select_path(prediction, hidden, anchor)

    def accept_length(self, draft: NDArray, verified: NDArray) -> NDArray:
        """
        The longest valid prefix, per row. DFlash verifies the whole block at
        once and keeps tokens up to the first disagreement.

        Only what counts as agreeing belongs to the head -- exact equality for a
        vocabulary, a tolerance for real values. Taking the longest agreeing
        prefix does not, so it lives here and is shared.
        """
        agree = self.head.accepted(draft, verified)
        return np.argmin(
            np.concatenate(
                [agree, np.zeros((agree.shape[0], 1), dtype=bool)], axis=1
            ),
            axis=1,
        )

    # -------------    the training objective    ----------------------
    @property
    def draft_hidden(self) -> NDArray:
        """the per-slot hidden from the last draft, what the selector reads"""
        return self._head_output

    @staticmethod
    def block_targets(sequence: NDArray,
                      block_size: int,
                      shift: int = 1) -> NDArray:
        """
        The sliding window of labels that lines up with draft_forward.

        Anchor a predicts positions a + 1 .. a + block_size, so shift 1 gives
        what each slot should emit, and shift 0 gives the value before it, the
        predecessor the walk conditions on.

        Head agnostic: this indexes axis 1 and touches nothing else, so a
        (batch, length) stream of token ids and a (batch, length, channels)
        stream of real values both come back with a block axis inserted.
        """
        anchors = sequence.shape[1] - block_size
        offsets = np.arange(shift, block_size + shift)
        return sequence[:, np.arange(anchors)[:, None] + offsets[None, :]]

    def block_loss(self, prediction: NDArray, targets: NDArray) -> tuple[float, NDArray]:
        """
        The head's mean cost over every slot of every block, and its gradient.

        Whatever the head, block_size 1 reduces this to the ordinary one step
        objective -- a decoder's cross entropy, or a plain heteroscedastic
        regression loss -- which is the cheapest check that the block machinery
        changed the shape of the objective and not the objective.
        """
        return self.head.loss(prediction, targets)

    # -------------    bookkeeping    ---------------------------------
    def get_gradients(self) -> dict:
        gradients = super().get_gradients()
        gradients.update({
            "gradient_mask_values": self.gradient_mask_values,
            "gradient_mask_query": self.gradient_mask_query,
            "conv_mlp": self.conv_mlp.get_gradients(),
            "fc_block_a": self.fc_block_a.get_gradients(),
            "fc_block_b": self.fc_block_b.get_gradients(),
            "fc_unary": self.fc_unary.get_gradients(),
            "head": self.head.get_gradients(),
        })
        return gradients

    def update_weights(self,
                       gradient_mask_values: NDArray,
                       gradient_mask_query: NDArray,
                       conv_mlp: dict,
                       fc_block_a: dict,
                       fc_block_b: dict,
                       fc_unary: dict,
                       head: dict,
                       **inherited) -> None:
        super().update_weights(**inherited)
        self.mask_values -= gradient_mask_values
        self.mask_query -= gradient_mask_query
        self.conv_mlp.update_weights(**conv_mlp)
        self.fc_block_a.update_weights(**fc_block_a)
        self.fc_block_b.update_weights(**fc_block_b)
        self.fc_unary.update_weights(**fc_unary)
        self.head.update_weights(**head)

    def zero_gradients(self) -> None:
        super().zero_gradients()
        if not self._draft_built:
            return
        self.gradient_mask_values = np.zeros_like(self.mask_values)
        self.gradient_mask_query = np.zeros_like(self.mask_query)
        self.conv_mlp.zero_gradients()
        self.head.zero_gradients()
        for layer in (self.fc_block_a, self.fc_block_b, self.fc_unary):
            layer.zero_gradients()

    def purge(self) -> None:
        super().purge()
        self.block_spectrum = None
        self.block_basis = None
        self.block_pos = None
        self.num_anchors = None
        self._head_input = None
        self._head_output = None
        self.conv_mlp.purge()
        self.head.purge()
        for layer in (self.fc_block_a, self.fc_block_b, self.fc_unary):
            layer.purge()

    @property
    def num_parameters(self) -> int:
        return (
            super().num_parameters
            + self.mask_values.size
            + self.mask_query.size
            + self.conv_mlp.num_parameters
            + self.fc_block_a.num_parameters
            + self.fc_block_b.num_parameters
            + self.fc_unary.num_parameters
            + self.head.num_parameters
        )

    def __str__(self):
        band = f", band radius {self.band_radius}" if self.band_radius else ""
        return (
            f"DFlash2 SPECTRE drafter, block {self.block_size}, sequence "
            f"{self.sequence_length}, hidden {self.hidden_dim}, "
            f"{self.num_heads} heads, {self.head}{band}"
        )


if __name__ == "__main__":
    """
    A very shallow causal LM over spectral decoders, then DFlash speculative
    decoding on top of it.

    Three things are worth watching. The LM carries no positional encoding --
    the mixer's twiddle factors already are the position. The drafter is trained
    against the target's own greedy output rather than against the data, because
    acceptance measures agreement with the target and where the target is
    confidently wrong the drafter should be wrong in the same way. And the
    decoded text is asserted identical to plain greedy decoding on every trial:
    categorical speculation is exact, so the only thing it buys is fewer target
    calls, and that is the number reported.
    """
    from ml_tools.models.optimizers import SGD

    VOCAB, HIDDEN, LENGTH = 8, 32, 32
    DEPTH, MIXER_HEADS = 2, 4
    BLOCK = 5
    NOISE = 0.05
    PROMPT, HORIZON = 3, 24
    DRAFT_FEATURE_RMS = 0.3

    RNG = np.random.default_rng(0)
    TABLE = RNG.integers(0, VOCAB, size=(VOCAB, VOCAB))


    def sample_language(count, rng):
        """
        A second order chain: the next symbol is fixed by the previous two, apart
        from a noise floor. Second order on purpose -- a block drafter has to commit
        to several symbols at once, so the slots have to agree with each other.
        """
        ids = np.zeros((count, LENGTH), dtype=np.int64)
        ids[:, :2] = rng.integers(0, VOCAB, size=(count, 2))
        for step in range(2, LENGTH):
            follow = TABLE[ids[:, step - 2], ids[:, step - 1]]
            noisy = rng.random(count) < NOISE
            ids[:, step] = np.where(noisy, rng.integers(0, VOCAB, size=count), follow)
        return ids


    class TokenEmbedding:
        """a lookup table, and the scatter that is its gradient"""

        def __init__(self, vocab_size, hidden_dim, rng):
            self.table = rng.normal(scale=0.1, size=(vocab_size, hidden_dim))
            self.ids = None
            self.zero_gradients()

        def forward(self, ids):
            self.ids = ids
            return self.table[ids]

        def backward(self, incoming):
            np.add.at(
                self.gradient_table,
                self.ids.reshape(-1),
                incoming.reshape(-1, incoming.shape[-1]),
            )

        def get_gradients(self):
            return {"gradient_table": self.gradient_table}

        def update_weights(self, gradient_table):
            self.table -= gradient_table

        def zero_gradients(self):
            self.gradient_table = np.zeros_like(self.table)

        def purge(self):
            self.ids = None

        @property
        def num_parameters(self):
            return self.table.size


    class SpectralDecoder:
        """
        One causal decoder layer: spectral token mixing, then a position wise MLP,
        each around a residual.

        Only the MLP's residual is written here. The mixer carries its own, so
        adding one around it would count the input twice.

        CausalSpectreAttention is the mixer alone -- its own fc_1 and fc_2 produce
        the complex gate, not a feed forward over tokens -- so the MLP belongs here.
        """

        def __init__(self, sequence_length, hidden_dim, num_heads):
            self.mixer = CausalSpectreAttention(
                sequence_length, hidden_dim, num_heads
            )
            self.up = FullyConnectedLayer(hidden_dim, 2 * hidden_dim, "relu")
            self.down = FullyConnectedLayer(2 * hidden_dim, hidden_dim, "linear")

        def forward(self, hidden):
            self.mixed = self.mixer.forward(hidden)
            return self.mixed + self.down(self.up(self.mixed))

        def backward(self, incoming):
            dmixed = incoming + self.up.backward(self.down.backward(incoming))
            return self.mixer.backward(dmixed)

        @property
        def parts(self):
            return [self.mixer, self.up, self.down]


    class ShallowCausalLM:
        """
        Embedding, a couple of spectral decoders, and an unembedding.

        No positional encoding anywhere: the mixer's twiddle factors are the
        position, so a token's place in the sequence is already in the transform.
        """

        def __init__(self, sequence_length, hidden_dim, vocab_size, depth,
                     num_heads, rng):
            self.embedding = TokenEmbedding(vocab_size, hidden_dim, rng)
            self.decoders = [
                SpectralDecoder(sequence_length, hidden_dim, num_heads)
                for _ in range(depth)
            ]
            self.unembedding = FullyConnectedLayer(hidden_dim, vocab_size, "linear")
            # a zeroed output projection starts the model at the uniform
            # distribution, so the first loss is ln(vocab) rather than whatever the
            # residual stream's scale happens to make it
            self.unembedding.weights[...] = 0.0

        def forward(self, ids):
            hidden = self.embedding.forward(ids)
            for decoder in self.decoders:
                hidden = decoder.forward(hidden)
            return self.unembedding(hidden), hidden

        def backward(self, dlogits):
            gradient = self.unembedding.backward(dlogits)
            for decoder in reversed(self.decoders):
                gradient = decoder.backward(gradient)
            self.embedding.backward(gradient)

        @property
        def parts(self):
            pieces = [self.embedding, self.unembedding]
            for decoder in self.decoders:
                pieces.extend(decoder.parts)
            return pieces

        @property
        def num_parameters(self):
            return sum(part.num_parameters for part in self.parts)


    def features(hidden):
        """
        Rescale the target's hidden states to a fixed RMS before the drafter reads
        them.

        Two reasons, and the second is the sharp one. The residual stream carries
        whatever scale training leaves it at, and the drafter would inherit it. And
        the drafter's block readout grows with the prefix: its prefix spectrum is a
        plain cumulative sum, and only the gate's query is divided by the count, so
        a unit RMS input here gives a block hidden with a standard deviation in the
        tens and gradients to match. Fixing the input scale keeps that in range.

        Applied identically in training and in decoding, so the drafter never sees
        two different input scales.
        """
        scale = np.sqrt((hidden ** 2).mean(axis=-1, keepdims=True) + 1e-6)
        return hidden * (DRAFT_FEATURE_RMS / scale)


    def next_token_loss(logits, ids):
        """teacher forced cross entropy, and the gradient padded back out"""
        cost, trimmed = CategoricalHead.loss(logits[:, :-1], ids[:, 1:])
        dlogits = np.zeros_like(logits)
        dlogits[:, :-1] = trimmed
        return cost, dlogits


    def train_target(model, steps, batch, learning_rate):
        optimizer = SGD(learning_rate=learning_rate)
        rng = np.random.default_rng(1)
        first = last = None
        for step in range(steps):
            ids = sample_language(batch, rng)
            logits, _ = model.forward(ids)
            last, dlogits = next_token_loss(logits, ids)
            for part in model.parts:
                part.zero_gradients()
            model.backward(dlogits)
            optimizer.step(model.parts)
            first = first if first is not None else last
            if step % 40 == 0 or step == steps - 1:
                print(f"       step {step:3d}  next token CE {last:.4f}")
        return first, last


    def target_stream(model, ids):
        """
        What the target would emit at every position, and the hidden states the
        drafter reads.

        The drafter is trained against these, not against the data: acceptance
        measures agreement with the target, and where the target is confidently
        wrong the drafter should be wrong the same way.
        """
        logits, hidden = model.forward(ids)
        greedy = np.argmax(logits, axis=-1)
        stream = np.concatenate([ids[:, :1], greedy[:, :-1]], axis=1)
        return stream, features(hidden)


    def train_drafter(model, drafter, steps, batch, learning_rate):
        optimizer = SGD(learning_rate=learning_rate)
        rng = np.random.default_rng(2)
        first = last = None
        for step in range(steps):
            ids = sample_language(batch, rng)
            stream, hidden = target_stream(model, ids)
            labels = drafter.block_targets(stream, BLOCK, 1)
            predecessors = drafter.block_targets(stream, BLOCK, 0)

            prediction = drafter.draft_forward(hidden)
            last, dlogits = drafter.block_loss(prediction, labels)
            drafter.zero_gradients()
            _, drefine, _ = drafter.select_forward(
                prediction, drafter.draft_hidden, predecessors, labels
            )
            drafter.draft_backward(dlogits + drefine)
            optimizer.step([drafter])
            first = first if first is not None else last
            if step % 40 == 0 or step == steps - 1:
                print(f"       step {step:3d}  block CE {last:.4f}")
        return first, last


    def greedy_decode(model, prompt, horizon):
        """one target call per token, the thing speculation has to beat"""
        ids = prompt.copy()
        calls = 0
        while ids.shape[1] < horizon:
            logits, _ = model.forward(ids)
            calls += 1
            ids = np.concatenate(
                [ids, np.argmax(logits[:, -1:], axis=-1)], axis=1
            )
        return ids, calls


    def speculative_decode(model, drafter, prompt, horizon):
        """
        Draft a block, verify it in one target pass, keep the agreeing prefix plus
        one bonus token.

        The bonus is what makes this worth doing: the target's token at the first
        disagreement is conditioned only on tokens that were accepted, so it is
        correct and free. A fully accepted block of BLOCK - 1 drafted tokens
        therefore commits BLOCK of them.

        The drafter's cache stops one position short of the committed sequence,
        because the newest committed token is the bonus and the verifying pass never
        saw it. That is exactly what sample_from_anchor False is for: slot 0
        re-predicts the token already known at the cache's edge and is dropped,
        anchoring the walk without spending a slot.
        """
        ids = prompt.copy()
        _, hidden = model.forward(ids)
        calls = 1
        runs = []

        while ids.shape[1] < horizon:
            length = ids.shape[1]
            drafter.reset_cache()
            drafter.prefill(features(hidden[:, :length - 1]))
            draft = drafter.propose(ids[:, length - 2])
            width = draft.shape[1]

            logits, hidden = model.forward(
                np.concatenate([ids, draft], axis=1)
            )
            calls += 1
            verified = np.argmax(
                logits[:, length - 1:length - 1 + width], axis=-1
            )
            taken = int(drafter.accept_length(draft, verified)[0])

            ids = np.concatenate(
                [ids, draft[:, :taken], verified[:, taken:taken + 1]], axis=1
            )
            hidden = hidden[:, :ids.shape[1] - 1]
            runs.append(taken + 1)

        return ids, calls, runs


    print("=" * 68)
    print("a shallow causal LM over spectral decoders, then DFlash speculation")
    print("=" * 68)
    target = ShallowCausalLM(LENGTH, HIDDEN, VOCAB, DEPTH, MIXER_HEADS,
                             np.random.default_rng(3))
    print(f"   target: {DEPTH} spectral decoders, {target.num_parameters:,} "
          f"parameters, vocab {VOCAB}")
    print(f"   uniform CE would be ln(vocab) = {np.log(VOCAB):.4f}")
    print("   training the target:")
    train_target(target, steps=400, batch=32, learning_rate=0.02)

    holdout = sample_language(64, np.random.default_rng(99))
    logits, _ = target.forward(holdout)
    agreement = (np.argmax(logits[:, :-1], -1) == holdout[:, 1:]).mean()
    print(f"   target next token accuracy on held out data {agreement:.1%}")
    print()

    drafter = DFlashSpectreAttention(
        sequence_length=LENGTH, hidden_dim=HIDDEN, vocab_size=VOCAB,
        num_heads=MIXER_HEADS, block_size=BLOCK, selector_rank=32,
        selector_top_k=8,
    )
    drafter.fc_unary.weights[...] = 0.0
    drafter.zero_gradients()
    print(f"   drafter: {drafter}")
    print(f"   {drafter.num_parameters:,} parameters, "
          f"{drafter.num_parameters / target.num_parameters:.2f}x the target")
    print("   training the drafter on the target's own output:")
    train_drafter(target, drafter, steps=400, batch=32, learning_rate=0.05)
    print()

    print("   decoding:")
    greedy_calls = speculative_calls = 0
    greedy_tokens = speculative_tokens = 0
    runs = []
    for trial in range(16):
        prompt = sample_language(1, np.random.default_rng(500 + trial))[:, :PROMPT]
        plain, plain_calls = greedy_decode(target, prompt, HORIZON)
        fast, fast_calls, fast_runs = speculative_decode(
            target, drafter, prompt, HORIZON
        )

        assert np.array_equal(plain[:, :HORIZON], fast[:, :HORIZON]), (
            f"trial {trial}: speculation changed the output"
        )
        greedy_calls += plain_calls
        speculative_calls += fast_calls
        greedy_tokens += plain.shape[1] - PROMPT
        speculative_tokens += fast.shape[1] - PROMPT
        runs.extend(fast_runs)

    runs = np.array(runs)
    print(f"   16 prompts of {PROMPT}, decoded to at least {HORIZON} tokens")
    print(f"   outputs identical over the horizon, asserted on every trial")
    print()
    print(f"   greedy       {greedy_calls:4d} target calls for {greedy_tokens} "
          f"tokens, {greedy_tokens / greedy_calls:.2f} per call")
    print(f"   speculative  {speculative_calls:4d} target calls for "
          f"{speculative_tokens} tokens, "
          f"{speculative_tokens / speculative_calls:.2f} per call")
    print(f"                {len(runs)} verifications plus 16 prefills")
    print()
    print(f"   committed per verification: mean {runs.mean():.2f} of a possible "
          f"{BLOCK}")
    for length in range(1, BLOCK + 1):
        share = (runs == length).mean()
        print(f"       {length} token{'s' if length > 1 else ' '}  "
              f"{'#' * round(share * 40):40s} {share:5.1%}")
    print()
    print(f"   {greedy_calls / speculative_calls:.2f}x fewer target calls for "
          f"the same output")
