from numpy.typing import NDArray
import numpy as np
from typing import Callable, Optional
from ml_tools.models.constants import EPSILON, GLOBAL_DTYPE, GLOBAL_COMPLEX_DTYPE
from ml_tools.models.activations import mod_relu, mod_relu_derivative
from ml_tools.models.layers.wavelet_layers import WaveletRefinementModule
from ml_tools.models.layers.layers import (
    xavier,
    kaiming,
    Layer,
    FullyConnectedLayer,
    NormalizeLayer
)



# -------------    adjoints of the real FFT pair    ----------------
def rfft_adjoint(grad_freq: NDArray,
                 sequence_length: int,
                 axis: int = 1
                 ) -> NDArray:
    out = np.fft.irfft(grad_freq, n=sequence_length, axis=axis)
    return (out * sequence_length).astype(GLOBAL_DTYPE)


def irfft_adjoint(grad_time: NDArray,
                  sequence_length: int,
                  axis: int = 1,
                  ) -> NDArray:
    out = np.fft.rfft(grad_time, n=sequence_length, axis=axis) / sequence_length
    interior = [slice(None)] * out.ndim
    interior[axis] = slice(1, -1 if sequence_length % 2 == 0 else None)
    out[tuple(interior)] *= 2

    return out.astype(GLOBAL_COMPLEX_DTYPE)


class PersistentMemory(Layer):
    """
    Learned, fixed-width context that is persisted
    per SPECTRE's persistent-memory extension

    holds M, shape (memory_tokens, hidden_dim), trained jointly with the model.
    """

    preserves_shape = False

    def __init__(self, memory_tokens: int, hidden_dim: int):
        super().__init__()
        assert memory_tokens >= 0, "memory_tokens must be zero or positive"
        self.memory_tokens = memory_tokens
        self.hidden_dim = hidden_dim
        # self.sequence_length = sequence_length

        # num_total = memory_tokens + sequence_length
        # self.num_frequencies = num_total // 2 + 1

        self.declare_shapes(inputs=(), outputs=((self.hidden_dim,),))

        self.memory = xavier(self.RNG, ni=memory_tokens, no=hidden_dim).astype(GLOBAL_DTYPE)
        self.zero_gradients()

    def get_memory(self) -> NDArray:
        return self.memory

    def forward(self) -> NDArray:
        return self.get_memory()

    def backward(self, incoming_gradient: NDArray) -> None:
        self.gradient_memory += incoming_gradient

    def update_weights(self, gradient_memory: NDArray) -> None:
        self.memory -= gradient_memory

    def purge(self) -> None:
        self._stale = True

    def zero_gradients(self) -> None:
        self.gradient_memory = np.zeros_like(self.memory)

    def get_weights(self, for_serialize: bool = False):
        if for_serialize:
            return {"memory": self.memory}
        return self.memory

    def set_weights(self, weights: dict) -> None:
        if weights is not None:
            self.memory = np.asarray(
                weights["memory"],
                dtype=GLOBAL_DTYPE,
            )
            self._stale = True

    def get_gradients(self) -> dict[str, NDArray]:
        if self.memory_tokens == 0:
            return {}
        return {"gradient_memory": self.gradient_memory}

    @property
    def num_parameters(self) -> int:
        return self.memory.size


class PrefixFFTCache:
    """
    Batched, hidden_dim-wide Prefix-FFT cache shared by all heads of a
    SpectreDecoderAttention layer.

    Ring layout (per batch element), length `max_sequence = memory_tokens +
    window'
    """

    def __init__(self,
                 sequence_length: int,
                 hidden_dim: int,
                 batch_size: int,
                 memory_tokens: int = 0):
        self.sequence_length = int(sequence_length)
        self.memory_tokens = int(memory_tokens)
        self.max_sequence = self.memory_tokens + self.sequence_length
        self.hidden_dim = int(hidden_dim)
        self.batch_size = int(batch_size)
        self.n_freq = self.max_sequence // 2 + 1

        self.prefix_fft = np.zeros(
            shape=(self.batch_size, self.n_freq, self.hidden_dim),
            dtype=GLOBAL_COMPLEX_DTYPE
        )
        self.value_buffer = np.zeros(
            shape=(self.batch_size, self.max_sequence, self.hidden_dim),
            dtype=GLOBAL_DTYPE
        )
        self.query_buffer = np.zeros(
            shape=(self.batch_size, self.max_sequence, self.hidden_dim),
            dtype=GLOBAL_DTYPE
        )
        self.mask_buffer = np.zeros((self.batch_size, self.max_sequence), dtype=bool)
        self.sum_query = np.zeros((self.batch_size, self.hidden_dim), dtype=GLOBAL_DTYPE)

        # absolute step counter for the *sliding* part only; memory slots
        # are written once (in prefill / set_memory) and are never touched
        # by this counter.
        self.position = 0
        self.length = np.zeros(self.batch_size)

        k = np.arange(self.n_freq, dtype=GLOBAL_DTYPE)
        t = np.arange(self.max_sequence, dtype=GLOBAL_DTYPE)
        self._twiddle = np.exp(
            -2j * np.pi * np.outer(t, k) / self.max_sequence
        ).astype(GLOBAL_COMPLEX_DTYPE)

    # set up and reset funcs ---------
    def reset(self):
        self.prefix_fft.fill(0)
        self.value_buffer.fill(0)
        self.query_buffer.fill(0)
        self.mask_buffer.fill(False)
        self.sum_query.fill(0)
        self.position = 0
        self.length.fill(0)

    def set_memory(self, memory: np.ndarray):
        """
        Seed the persistent memory "slot positions". `memory` is
        (memory_tokens, hidden_dim), broadcast across the batch -- matches
        PersistentMemory.get_memory(), which is shared, not per-sample.
        Call this once per session (or whenever the memory weights change);
        it does not touch the sliding-window part of the cache.
        """
        if self.memory_tokens == 0:
            return
        if memory.shape != (self.memory_tokens, self.hidden_dim):
            raise ValueError(
                f"expected memory shape ({self.memory_tokens}, {self.hidden_dim}), "
                f"got {memory.shape}"
            )
        self.value_buffer[:, : self.memory_tokens] = memory[None].astype(GLOBAL_DTYPE)
        self.mask_buffer[:, : self.memory_tokens] = True

        self.prefix_fft[...] = np.fft.rfft(
            self.value_buffer, n=self.max_sequence, axis=1).astype(GLOBAL_COMPLEX_DTYPE)

    def prefill(self,
                query: np.ndarray,
                value: np.ndarray,
                mask: Optional[np.ndarray] = None):
        """
        One-shot cache initialisation

        query, value : (batch, seq, hidden_dim), already re-merged after heads.
        mask : (batch, seq) optional validity mask to identify.

        single RFFT seeds the full cache.
        """
        batch, length, hidden_dim = value.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"expected hidden_dim={self.hidden_dim}, got {hidden_dim}")
        if length > self.sequence_length:
            raise ValueError(f"prompt length {length} exceeds window={self.sequence_length}")
        if batch != self.batch_size:
            raise ValueError(f"cache batch_size={self.batch_size}, got {batch}")

        if mask is None:
            mask = np.ones((batch, length), dtype=GLOBAL_DTYPE)
        mask = mask.astype(GLOBAL_DTYPE)

        memory_rows = self.value_buffer[:, : self.memory_tokens].copy()
        memory_mask = self.mask_buffer[:, : self.memory_tokens].copy()
        self.reset()
        self.value_buffer[:, : self.memory_tokens] = memory_rows
        self.mask_buffer[:, : self.memory_tokens] = memory_mask

        query_valid = (query * mask[..., None]).astype(GLOBAL_DTYPE)
        value_valid = (value * mask[..., None]).astype(GLOBAL_DTYPE)

        start = self.memory_tokens
        self.value_buffer[:, start:start + length] = value_valid
        self.query_buffer[:, start:start + length] = query_valid
        self.mask_buffer[:, start:start + length] = mask.astype(bool)

        self.prefix_fft[...] = np.fft.rfft(
            self.value_buffer, n=self.max_sequence, axis=1
        ).astype(GLOBAL_COMPLEX_DTYPE)

        self.sum_query[...] = query_valid.sum(axis=1)
        self.length[...] = mask.sum(axis=1).astype(np.int32)
        self.position = length

    # DECODE STEPS ------------------
    def decode_step(self,
                    query_t: np.ndarray,
                    value_t: np.ndarray,
                     valid=True) -> int:
        """
        append one token to the sliding window

        query_t, value_t : (batch, hidden_dim), already re-merged after heads.
        valid : bool or (batch,) bool array, for padded/finished sequences

        returns the ring's positional `slot` the new token was written to, so callers can
        read the reconstructed row for the newest token
        """
        batch = value_t.shape[0]
        if batch != self.batch_size:
            raise ValueError(f"cache batch_size={self.batch_size}, got {batch}")

        valid = np.asarray(valid, dtype=bool)
        if valid.ndim == 0:
            valid = np.full(batch, bool(valid))

        t = self.position
        slot = self.memory_tokens + (t % self.sequence_length)

        query_t = np.where(valid[:, None], query_t, 0.0).astype(GLOBAL_DTYPE)
        value_t = np.where(valid[:, None], value_t, 0.0).astype(GLOBAL_DTYPE)

        if t >= self.sequence_length:
            old_slot = self.memory_tokens + ((t - self.sequence_length) % self.sequence_length)
            old_value = self.value_buffer[:, old_slot].copy()
            old_query = self.query_buffer[:, old_slot].copy()
            was_valid = self.mask_buffer[:, old_slot].copy()

            # evict using the same twiddle index
            self.prefix_fft -= (
                self._twiddle[old_slot, :][None, :, None] * old_value[:, None, :]
            )
            self.sum_query -= np.where(was_valid[:, None], old_query, 0.0)

        self.prefix_fft += (
            self._twiddle[slot][None, :, None] * value_t[:, None, :]
        )

        self.value_buffer[:, slot] = value_t
        self.query_buffer[:, slot] = query_t
        self.mask_buffer[:, slot] = valid
        self.sum_query += query_t

        self.position += 1
        self.length = np.minimum(self.length + valid.astype(np.int64), self.sequence_length)

        return slot

    # ------------------------------------------------------------------
    @property
    def live_length(self) -> int:
        return self.memory_tokens + int(min(self.position, self.sequence_length))

    def reconstruct(self,
                    gate: np.ndarray) -> np.ndarray:
        """
        gate : (batch, n_freq, hidden_dim) complex spectral gate, already
            broadcast/merged across heads (needs to be aligned before reconstruct)

        Returns the full ring-ordered reconstruction, shape (batch, max_sequence, hidden_dim).
        Slot ordering, not chronological ordering.
        see `read_slot` / `chronological_order` to extract a specific token or the whole window in seqence
        """
        spectrum = self.prefix_fft * gate
        return np.fft.irfft(spectrum, n=self.max_sequence, axis=1).astype(GLOBAL_DTYPE)

    def get_chronological_order(self) -> np.ndarray:
        """
        Index array that reorders the ring buffer's movible or sliding portion into
        chronological (oldest -> newest) order, given the current pointer position.
        Memory slots are already in a fixed order.
        """
        if self.position == 0:
            window_order = np.arange(self.sequence_length)
        else:
            newest_slot = (self.position - 1) % self.sequence_length
            window_order = (np.arange(self.sequence_length) + newest_slot + 1) % self.sequence_length
        return np.concatenate([np.arange(self.memory_tokens),
                                self.memory_tokens + window_order])

class SpectreAttention(Layer):
    """
    SPECTRE mixing layer, https://arxiv.org/abs/2502.18394

    Query/value projections are per-head (num_heads independent head_dim x
    head_dim maps), matching the paper's W(q), W(v) per head, rather than one
    shared hidden_dim x hidden_dim projection split afterward.

    """
    registry_name = "SPECTREAttention"
    preserves_shape = True
    
    def __init__(self,
                 sequence_length: int,
                 hidden_dim: int,
                 num_heads: int = 1,
                 band_radius: int = 0,
                 memory_tokens: int = 0,
                 causal_decode: bool = False,
                 modrelu_bias: float = 0.0,
                 use_wrm: bool = False,
                 use_positional_phase: bool = True,
                 ):
        """
        Parameters
        ----------
        sequence_length : tokens per sample, the axis the FFT runs over
        hidden_dim : channel width of the input
        num_heads : gates learned in parallel, each over its own slice of the
            channel axis. 1 recovers the single-head layer exactly.
        band_radius : radius r of the optional Toeplitz band update on the
            gate. 0 disables it. r > 0 adds 2r+1 complex taps per head.
        memory_tokens : size of an optional learned, persistent, never-evicted
            context bank (paper Sec 3.4). 0 disables it.
        """
        assert memory_tokens >= 0, "memory_tokens must be zero or positive"
        assert num_heads >= 1, f"num_heads must be at least 1, got {num_heads}"
        assert hidden_dim % num_heads == 0, (
            f"num_heads {num_heads} must divide hidden_dim {hidden_dim}. Heads "
            "partition the channel axis, so a remainder would leave channels "
            "ungated."
        )
        super().__init__()

        self.sequence_length = sequence_length
        # head dimensions ---
        self.hidden_dim: int = hidden_dim
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_dim // num_heads
        self.memory_tokens: int = memory_tokens

        self.use_positional_phase: bool = use_positional_phase
        self.causal_decode: bool = causal_decode
        self.modrelu_bias: float = modrelu_bias

        self.training_now: bool = True

        self.declare_shapes(inputs=((hidden_dim,),), outputs=((hidden_dim,),))

        if self.memory_tokens > 0:
            self.memory = PersistentMemory(memory_tokens=self.memory_tokens,
                                           hidden_dim=self.hidden_dim)

        self.fft_length = self.sequence_length + self.memory_tokens

        self._cache = None
        self._last_forward = None

        self.declare_shapes(
            inputs=((None, None, self.hidden_dim),),
            outputs=((None, None, self.hidden_dim),),
        )

        self.num_frequencies = self.fft_length // 2 + 1
        self.activation_bias = np.zeros((num_heads, self.num_frequencies), dtype=GLOBAL_DTYPE) - 0.1


        # per-head independent projections
        self.query_weights = self.init_head_projection()
        self.query_bias = np.zeros((num_heads, self.head_dim), dtype=GLOBAL_DTYPE)
        self.values_weights = self.init_head_projection()
        self.values_bias = np.zeros((num_heads, self.head_dim), dtype=GLOBAL_DTYPE)

        # LN over the feature axis of the pooled query, per the paper
        self.norm_query = NormalizeLayer(ni=hidden_dim, shift_scale=True)

        self.fc_1 = FullyConnectedLayer(ni=hidden_dim,
                                        no=hidden_dim,
                                        activation_type="relu")
        self.fc_2 = FullyConnectedLayer(ni=hidden_dim,
                                        no=2 * num_heads * self.num_frequencies,
                                        activation_type="linear",
        )

        assert band_radius >= 0, (
            f"band_radius must be zero or positive, got {band_radius}. A "
            "negative radius produces no taps and silently disables the gate."
        )

        # band radius is a dimensionality reduction mechanism
        self.band_radius = band_radius
        self.band_offsets = tuple(range(-band_radius, band_radius + 1))
        if band_radius:
            self.band_taps = np.zeros(
                (num_heads, len(self.band_offsets)), dtype=np.complex64
            )

        self.use_wrm = use_wrm
        if use_wrm:
            self.wrm = WaveletRefinementModule(
                hidden_dim=hidden_dim,
                sequence_length=self.sequence_length,
                on_rate=0.1,
                skip_threshold=0.5,
            )

        self.activation: Callable = mod_relu
        self.activation_derivative: Callable = mod_relu_derivative

        # seeded here so a gradient read before the first backward finds zeros
        # rather than raising AttributeError
        self.zero_gradients()

    def init_head_projection(self):
        # (num_heads, head_dim, head_dim)
        return np.stack(
            [kaiming(self.RNG, ni=self.head_dim, no=self.head_dim,)
             for _ in range(self.num_heads)
             ]
        ).astype(GLOBAL_DTYPE)


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
        """(..., hidden) -> (..., head, head_dim). Any number of leading axes."""
        return spectrum.reshape(*spectrum.shape[:-1], self.num_heads, self.head_dim)

    def _merge_heads(self, spectrum: NDArray) -> NDArray:
        """(..., head, head_dim) -> (..., hidden). Any number of leading axes."""
        return spectrum.reshape(*spectrum.shape[:-2], self.hidden_dim)

    def _project_heads(self, input_data: NDArray, weights: NDArray, bias: NDArray
                       ) -> NDArray:
        """
        Independent (head_dim, head_dim) map per head.

        input_data : (..., hidden)
        weights : (num_heads, head_dim, head_dim)
        bias : (num_heads, head_dim)
        """
        heads = self._split_heads(input_data)
        projected = np.einsum('...hd,hde->...he', heads, weights) + bias
        return self._merge_heads(projected)

    # def _project_heads_backward(self, input_data: NDArray, doutput: NDArray, weights: NDArray
    #                             ) -> tuple[NDArray, NDArray, NDArray]:
    #     input_heads = self._split_heads(input_data)
    #     doutput_heads = self._split_heads(doutput)
    #
    #     leading_shape = np.broadcast_shapes(input_heads.shape[:-2], doutput_heads.shape[:-2])
    #     input_heads = np.broadcast_to(input_heads, (*leading_shape, *input_heads.shape[-2:]))
    #     doutput_heads = np.broadcast_to(doutput_heads, (*leading_shape, *doutput_heads.shape[-2:]))
    #
    #     leading_axes = tuple(range(len(leading_shape)))
    #     dweights = np.einsum('...hd,...he->...hde', input_heads, doutput_heads)
    #     dbias = doutput_heads.sum(axis=leading_axes)
    #     dinput = self._merge_heads(np.einsum('...he,hde->...hd', doutput_heads, weights))
    #     return dweights, dbias, dinput
    def _project_heads_backward(
            self,
            input_data: NDArray,
            doutput: NDArray,
            weights: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:

        input_heads = self._split_heads(input_data)
        doutput_heads = self._split_heads(doutput)

        leading_shape = np.broadcast_shapes(
            input_heads.shape[:-2],
            doutput_heads.shape[:-2],
        )

        input_heads = np.broadcast_to(
            input_heads,
            (*leading_shape, self.num_heads, self.head_dim),
        )

        doutput_heads = np.broadcast_to(
            doutput_heads,
            (*leading_shape, self.num_heads, self.head_dim),
        )

        # Collapse every leading dimension into one sample/token axis.
        #
        # (B, L, H, D) -> (B*L, H, D)
        #
        # This avoids materialising a huge (B, L, H, D, D) tensor.
        flat_input = input_heads.reshape(
            -1,
            self.num_heads,
            self.head_dim,
        )

        flat_doutput = doutput_heads.reshape(
            -1,
            self.num_heads,
            self.head_dim,
        )

        # dW[h, d_in, d_out]
        #
        # sum over all batch/token positions.
        dweights = np.einsum(
            'nhd,nhe->hde',
            flat_input,
            flat_doutput,
            optimize=True,
        )

        # db[h, d_out]
        dbias = flat_doutput.sum(axis=0)

        # dX[..., h, d]
        dinput_heads = np.einsum(
            '...he,hde->...hd',
            doutput_heads,
            weights,
            optimize=True,
        )

        dinput = self._merge_heads(dinput_heads)

        return dweights, dbias, dinput

    @staticmethod
    def _align_gate(gate: NDArray) -> NDArray:
        """(batch, head, frequency) -> (batch, frequency, head, 1)"""
        return np.transpose(gate, (0, 2, 1))[..., None]

    def _band_update(self, gate: NDArray) -> NDArray:
        banded = np.zeros_like(gate)
        for index, offset in enumerate(self.band_offsets):
            banded += self.band_taps[:, index, None] * self._shift(gate, offset)
        return gate + banded

    def forward(
            self,
            input_data: NDArray,
            mask: Optional[NDArray] = None,
            training_now: bool = True,
    ):
        assert input_data.ndim == 3
        assert input_data.shape[1] == self.sequence_length
        assert input_data.shape[2] == self.hidden_dim

        self.input = input_data
        self.training_now = training_now

        batch = input_data.shape[0]

        self.mask = (
            mask.astype(GLOBAL_DTYPE)
            if mask is not None
            else np.ones(input_data.shape[:2], dtype=GLOBAL_DTYPE)
        )

        self.counts = np.maximum(self.mask.sum(axis=1, keepdims=True),1.0)
        mask_column = self.mask[..., None]

        if self.memory_tokens:
            memory = self.memory.get_memory()
            memory_batch = np.broadcast_to(memory[None, :, :],(batch, self.memory_tokens, self.hidden_dim))
            # this is a tricky point - we need to concat the trainable memory tokens with the input data, then in backpass
            # we have to split out the gradients to their distinct sources.
            combined = np.concatenate([memory_batch, input_data], axis=1)

        else:
            combined = input_data

        self.combined_input = combined
        total_length = combined.shape[1]

        query_all = self._project_heads(combined, self.query_weights, self.query_bias)
        value_all = self._project_heads(combined, self.values_weights, self.values_bias)

        if self.memory_tokens:
            memory_query = query_all[:, :self.memory_tokens]
            query_forward = query_all[:, self.memory_tokens:]

            memory_value = value_all[:, :self.memory_tokens]
            value_forward = value_all[:, self.memory_tokens:]

            value_forward = value_forward * mask_column

            self.total_counts = (self.counts + self.memory_tokens)

            seq_sum = (query_forward * mask_column).sum(axis=1)
            seq_sum += memory_query.sum(axis=1)

        else:
            query_forward = query_all
            value_forward = value_all * mask_column

            self.total_counts = self.counts

            seq_sum = (query_forward * mask_column).sum(axis=1)

        self.seq_mu = seq_sum / self.total_counts
        self.descriptor = self.norm_query(self.seq_mu)

        # -- project through the FC layers
        gate_projection = self.fc_2(self.fc_1(self.descriptor))

        g_real, g_imag = np.split(gate_projection,2, axis=-1)

        self.gate_raw = (g_real + 1j * g_imag).astype(GLOBAL_COMPLEX_DTYPE)

        self.gate_raw = self.gate_raw.reshape(batch, self.num_heads, self.num_frequencies)

        if self.band_radius:
            self.gate_pre_activation = self._band_update(self.gate_raw)
        else:
            self.gate_pre_activation = self.gate_raw

        self.gate = self.activation(self.gate_pre_activation, self.activation_bias)

        # Full combined sequence gets a single transform into frequency space:
        self.combined_length = total_length
        self.num_combined_frequencies = total_length // 2 + 1

        self.value_transform = np.fft.rfft(
            value_all,
            n=total_length,
            axis=1,
        )

        values_gated = (self._split_heads(self.value_transform) * self._align_gate(self.gate))

        output_all = np.fft.irfft(
            self._merge_heads(values_gated),
            n=total_length,
            axis=1,
        )

        if self.memory_tokens:
            self.output = output_all[:, self.memory_tokens:]
        else:
            self.output = output_all

        if self.use_wrm:
            self.output = self.wrm.forward(self.output, self.descriptor, training_now=self.training_now)

        return self.output


    def backward(self, incoming_gradient: NDArray) -> NDArray:
        mask_column = self.mask[..., None]
        if self.use_wrm:
            incoming_gradient, d_descriptor_wrm = self.wrm.backward(incoming_gradient)
        else:
            d_descriptor_wrm = np.zeros_like(self.descriptor)

        # if memory, we have to be careful with the backprop, as part of the sequence is from our data, and part
        # is from our memory
        if self.memory_tokens:
            full_gradient = np.zeros(
                (incoming_gradient.shape[0], self.combined_length, self.hidden_dim),
                dtype=incoming_gradient.dtype,
            )
            full_gradient[:, self.memory_tokens:] = incoming_gradient
        else:
            full_gradient = incoming_gradient

        dvalues_spec_gated = self._split_heads(
            irfft_adjoint(full_gradient,
                          self.combined_length,
                          axis=1)
        )

        # split the gradients back across our heads -----
        value_heads = self._split_heads(self.value_transform)

        dV_hat = self._merge_heads(dvalues_spec_gated * np.conj(self._align_gate(self.gate)))
        dgate = np.transpose(
            np.sum(dvalues_spec_gated * np.conj(value_heads), axis=-1),
            (0, 2, 1)
        )

        if self.band_radius:
            self.gradient_band = np.stack([
                np.sum(
                    dgate * np.conj(self._shift(self.gate, offset)),
                    axis=(0, 2),
                )
                for offset in self.band_offsets
            ], axis=-1)

            dgate_activated = dgate.copy()
            for index, offset in enumerate(self.band_offsets):
                dgate_activated += (
                        np.conj(self.band_taps[:, index, None])
                        * self._shift(dgate, -offset)
                )

        else:
            dgate_activated = dgate

        self.gradient_bias, dgate_pre = self.activation_derivative(
            z=self.gate_pre_activation,
            beta=self.activation_bias,
            dout=dgate_activated,
        )

        batch = dgate_pre.shape[0]
        dgate_projection = np.concatenate(
            [dgate_pre.real.reshape(batch, -1), dgate_pre.imag.reshape(batch, -1)],
            axis=-1,
        )

        dhidden = self.fc_2.backward(dgate_projection)
        ddescriptor_spectral = self.fc_1.backward(dhidden)
        ddescriptor = ddescriptor_spectral + d_descriptor_wrm

        dseq_mu = self.norm_query.backward(ddescriptor)

        dquery = mask_column * (dseq_mu[:, None, :] / self.total_counts[..., None])
        dvalues = rfft_adjoint(dV_hat, self.combined_length, axis=1)

        if self.memory_tokens:
            d_memory_query = np.broadcast_to(
                (dseq_mu / self.total_counts)[:, None, :],
                (dseq_mu.shape[0], self.memory_tokens, self.hidden_dim),
            )
            d_memory_value = dvalues[:, :self.memory_tokens]
            dvalues = dvalues[:, self.memory_tokens:] * mask_column

            dquery_all = np.concatenate([d_memory_query, dquery], axis=1).astype(GLOBAL_DTYPE)
            dvalue_all = np.concatenate([d_memory_value, dvalues], axis=1).astype(GLOBAL_DTYPE)

            # d_memory_value = np.real(
            #     np.einsum('mf,bfh->bmh', np.conj(self.memory_twiddle), dV_hat)
            # )


        else:
            dvalues = dvalues * mask_column
            dquery_all = dquery.astype(GLOBAL_DTYPE)
            dvalue_all = dvalues.astype(GLOBAL_DTYPE)

        self.gradient_query_weights, self.gradient_query_bias, dinput_from_q_all = (
            self._project_heads_backward(
                self.combined_input,
                dquery_all,
                self.query_weights)
        )
        self.gradient_values_weights, self.gradient_values_bias, dinput_from_v_all = (
            self._project_heads_backward(
                self.combined_input,
                dvalue_all,
                self.values_weights)
        )

        if self.memory_tokens:
            self.memory.backward(
                dinput_from_q_all[:, :self.memory_tokens].sum(axis=0)
                + dinput_from_v_all[:, :self.memory_tokens].sum(axis=0)
            )
            dinput_from_q = dinput_from_q_all[:, self.memory_tokens:]
            dinput_from_v = dinput_from_v_all[:, self.memory_tokens:]
        else:
            dinput_from_q = dinput_from_q_all
            dinput_from_v = dinput_from_v_all

        grad_real = (dinput_from_q + dinput_from_v).real.astype(GLOBAL_DTYPE)
        return grad_real

    def get_weights(self, for_serialize: bool = False) -> tuple|dict:
        if for_serialize:
            weights = {
                "activation_bias": self.activation_bias,
                "query_weights": self.query_weights,
                "query_bias": self.query_bias,
                "values_weights": self.values_weights,
                "values_bias": self.values_bias,
                "norm_query": self.norm_query.get_weights(for_serialize=True),
                "fc_1": self.fc_1.get_weights(for_serialize=True),
                "fc_2": self.fc_2.get_weights(for_serialize=True),
            }
            if self.band_radius:
                weights["band_taps"] = self.band_taps

            if self.use_wrm:
                weights["wrm"] = self.wrm.get_weights(for_serialize=True)

            if self.memory_tokens:
                weights["persistent_memory"] = self.memory.get_weights(for_serialize=True)

            return weights

        weights = [
            self.activation_bias,
            self.query_weights,
            self.query_bias,
            self.values_weights,
            self.values_bias,
            self.norm_query.get_weights(for_serialize=False),
            self.fc_1.get_weights(for_serialize=False),
            self.fc_2.get_weights(for_serialize=False),
        ]
        if self.memory_tokens:
            weights.append(self.memory.get_weights(for_serialize=False))

        if self.band_radius:
            weights.append(self.band_taps)

        if self.use_wrm:
            weights.append(self.wrm.get_weights(for_serialize=False))
        return tuple(weights)

    def set_weights(self, weights: dict) -> None:
        if weights is None:
            return
        self.activation_bias = np.asarray(weights["activation_bias"], dtype=GLOBAL_DTYPE)
        self.query_weights = np.asarray(weights["query_weights"], dtype=GLOBAL_DTYPE)
        self.query_bias = np.asarray(weights["query_bias"], dtype=GLOBAL_DTYPE)
        self.values_weights = np.asarray(weights["values_weights"], dtype=GLOBAL_DTYPE)
        self.values_bias = np.asarray(weights["values_bias"], dtype=GLOBAL_DTYPE)
        self.norm_query.set_weights(weights["norm_query"])
        self.fc_1.set_weights(weights["fc_1"])
        self.fc_2.set_weights(weights["fc_2"])

        if self.band_radius and "band_taps" in weights:
            self.band_taps = np.asarray(weights["band_taps"], dtype=GLOBAL_COMPLEX_DTYPE)

        if self.use_wrm and "wrm" in weights:
            self.wrm.set_weights(weights["wrm"])

        if self.memory_tokens and "persistent_memory" in weights:
            self.memory.set_weights(weights["persistent_memory"])

    def get_gradients(self) -> dict[str, NDArray | dict]:
        gradients = {
            "gradient_bias": self.gradient_bias,
            "gradient_query_weights": self.gradient_query_weights,
            "gradient_query_bias": self.gradient_query_bias,
            "gradient_values_weights": self.gradient_values_weights,
            "gradient_values_bias": self.gradient_values_bias,
            "norm_query": self.norm_query.get_gradients(),
            "fc_1": self.fc_1.get_gradients(),
            "fc_2": self.fc_2.get_gradients(),
        }
        if self.memory_tokens:
            gradients["persistent_memory"] = self.memory.get_gradients()

        if self.band_radius:
            gradients["gradient_band"] = self.gradient_band

        if self.use_wrm:
            gradients["wrm"] = self.wrm.get_gradients()

        return gradients

    def purge(self) -> None:
        self.input = None
        self.combined_input = None
        self.value_transform = None
        self.seq_mu = None
        self.total_counts = None
        self.gate_raw = None
        self.gate_activated = None
        self.gate = None
        self.output = None
        self.mask = None
        self.counts = None

        self.norm_query.purge()
        self.fc_1.purge()
        self.fc_2.purge()

        if self.memory_tokens:
            self.memory.purge()

        if self.use_wrm:
            self.wrm.purge()

    def zero_gradients(self):
        self.gradient_bias = np.zeros_like(self.activation_bias)
        self.gradient_query_weights = np.zeros_like(self.query_weights)
        self.gradient_query_bias = np.zeros_like(self.query_bias)
        self.gradient_values_weights = np.zeros_like(self.values_weights)
        self.gradient_values_bias = np.zeros_like(self.values_bias)
        self.fc_1.zero_gradients()
        self.fc_2.zero_gradients()

        if self.band_radius:
            self.gradient_band = np.zeros_like(self.band_taps)

        if self.memory_tokens:
            self.memory.zero_gradients()

        if self.use_wrm:
            self.wrm.zero_gradients()

    @property
    def num_parameters(self) -> int:
        total = (self.activation_bias.size
                 + self.query_weights.size
                 + self.query_bias.size
                 + self.values_weights.size
                 + self.values_bias.size
                 + self.norm_query.num_parameters
                 + self.fc_1.num_parameters
                 + self.fc_2.num_parameters)

        if self.memory_tokens:
            total += self.memory.num_parameters

        if self.band_radius:
            total += 2 * self.band_taps.size

        if self.use_wrm:
            total += self.wrm.num_parameters
        return total

    def update_weights(
            self,
            gradient_bias: NDArray,
            gradient_query_weights: NDArray,
            gradient_query_bias: NDArray,
            gradient_values_weights: NDArray,
            gradient_values_bias: NDArray,
            norm_query: dict[str, NDArray],
            fc_1: dict[str, NDArray],
            fc_2: dict[str, NDArray],
            persistent_memory: dict[str, NDArray] = None,
            gradient_band: NDArray = None,
            wrm: dict = None,
    ) -> None:

        self.activation_bias -= gradient_bias
        self.query_weights -= gradient_query_weights
        self.query_bias -= gradient_query_bias
        self.values_weights -= gradient_values_weights
        self.values_bias -= gradient_values_bias

        self.norm_query.update_weights(**norm_query)
        self.fc_1.update_weights(**fc_1)
        self.fc_2.update_weights(**fc_2)

        if self.memory_tokens and persistent_memory is not None:
            self.memory.update_weights(**persistent_memory)

        if self.band_radius and gradient_band is not None:
            self.band_taps -= gradient_band

        if self.use_wrm and wrm is not None:
            self.wrm.update_weights(**wrm)



    def __str__(self):
        band = f", band radius {self.band_radius}" if self.band_radius else ""
        memory = f", {self.memory_tokens} memory slots" if self.memory_tokens else ""
        return (
            f"SPECTRE mixer, sequence {self.sequence_length}, "
            f"hidden {self.hidden_dim}, {self.num_heads} heads{band}{memory}"
        )

    def __repr__(self):
        return self.__str__()

# =============================== the causal Decoder version =============================
class SpectreDecoderAttention(SpectreAttention):
    """
    Causal, autoregressive-decoding companion to `SpectreAttention`.
    Usage
    -----
        layer = SpectreDecoderAttention(sequence_length=..., hidden_dim=..., num_heads=..., memory_tokens=...)

        # train as normal — self.cache stays None the whole time
        out = layer.forward(batch_x, mask=batch_mask, training_now=True)
        grad = layer.backward(deltaout)
        layer.update_weights(**layer.get_gradients())

        # it's important to seperate the training from generation -- since the self.forward tracks the most recent.
        # So after training, generation:
        last_hidden = layer.prefill(prompt_embeddings, mask=prompt_mask)
        for _ in range(n_new_tokens):
            last_hidden = layer.decode_step(next_token_embedding)

    use_wrm is not supported here. WaveletRefinementModule's Haar transform
    needs the whole sequence_length window at once, but prefill/decode_step
    only ever reconstruct one live position at a time, so it has no
    single-token forward it could call.
    """
    registry_name = "SPECTREDecoderAttention"

    def __init__(self,
                 sequence_length: int,
                 hidden_dim: int,
                 num_heads: int = 1,
                 band_radius: int = 0,
                 memory_tokens: int = 0,
                 causal_decode: bool = False,
                 modrelu_bias: float = 0.0,
                 use_wrm: bool = False,
                 use_positional_phase: bool = True,
                 ):
        assert not use_wrm, (
            "SpectreDecoderAttention does not support use_wrm: the Wavelet "
            "Refinement Module needs the full sequence_length window, but "
            "prefill/decode_step only ever produce one live token at a time"
        )
        super().__init__(
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            band_radius=band_radius,
            memory_tokens=memory_tokens,
            causal_decode=causal_decode,
            modrelu_bias=modrelu_bias,
            use_wrm=use_wrm,
            use_positional_phase=use_positional_phase,
        )
        self.cache: Optional[PrefixFFTCache] = None

    def reset_cache(self, batch_size: int):
        self.cache = PrefixFFTCache(
            sequence_length=self.sequence_length,
            hidden_dim=self.hidden_dim,
            batch_size=batch_size,
            memory_tokens=self.memory_tokens,
        )
        if self.memory_tokens:
            self.cache.set_memory(self.memory.get_memory())

    # ------------------------------------------------------------------
    def _gate_from_sum_query(self,
                             sum_query: NDArray,
                             total_counts: NDArray) -> NDArray:
        """
        Shared gate computation facto
        sum_query, total_counts : (batch, hidden_dim), (batch, 1)
        returns gate : (batch, num_heads, num_frequencies) (complex dtype)
        """
        batch = sum_query.shape[0]
        seq_mu = sum_query / total_counts
        descriptor = self.norm_query(seq_mu)

        gate_projection = self.fc_2(self.fc_1(descriptor))
        g_real, g_imag = np.split(gate_projection, 2, axis=-1)
        gate_raw = (g_real + 1j * g_imag).astype(GLOBAL_COMPLEX_DTYPE)
        gate_raw = gate_raw.reshape(batch, self.num_heads, self.num_frequencies)

        gate_pre = self._band_update(gate_raw) if self.band_radius else gate_raw
        gate = self.activation(gate_pre, self.activation_bias)
        return gate, descriptor

    # ------------------------------------------------------------------
    def prefill(self, input_data: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Process a prompt and return the live representation for the *last*
        prompt token (what you'd feed to the LM head to sample the first
        new token). Populates the Prefix-FFT cache as a side effect.

        input_data : (batch, L, hidden_dim), L <= sequence_length
        mask : (batch, L) optional
        """
        assert input_data.ndim == 3
        batch, length, hidden_dim = input_data.shape
        assert hidden_dim == self.hidden_dim

        self.reset_cache(batch_size=batch)

        mask = (mask.astype(GLOBAL_DTYPE) if mask is not None
                else np.ones((batch, length), dtype=GLOBAL_DTYPE))

        query_all = self._project_heads(input_data, self.query_weights, self.query_bias)
        value_all = self._project_heads(input_data, self.values_weights, self.values_bias)

        self.cache.prefill(query_all, value_all, mask=mask)

        total_counts = np.maximum(mask.sum(axis=1, keepdims=True), 1.0) + self.memory_tokens
        gate, _descriptor = self._gate_from_sum_query(self.cache.sum_query, total_counts)

        gate_aligned = self._align_gate(gate)  # (batch, n_freq, heads, 1)
        gate_full = self._merge_heads(
            np.broadcast_to(
                gate_aligned,
                (batch, self.cache.n_freq, self.num_heads, self.head_dim),
            )
        )
        window = self.cache.reconstruct(gate_full)  # (batch, max_sequence, hidden)

        last_slot = self.memory_tokens + ((length - 1) % self.sequence_length)
        output = window[:, last_slot, :]

        if self.use_wrm:
            output = self.wrm.forward(
                output[:, None, :], _descriptor, training_now=False
            )[:, 0, :]

        return output

    # ------------------------------------------------------------------
    def decode_step(self, input_t: NDArray, valid=True) -> NDArray:
        """
        Append one new token and return its live representation.

        input_t : (batch, hidden_dim) -- raw embedding for the position
        valid : bool or (batch,) bool -- False marks a padding step for
            finished sequences in a batch; it still advances the cache
            (consistent with the ring-buffer accounting) but writes a zero
            token and does not affect sum_query.

        positional phase -- Multiplying the gate by exp(j2*pi*k*t/N). It is decode-only.
        """
        assert self.cache is not None, "call reset_cache()/prefill() first"
        assert input_t.ndim == 2 and input_t.shape[1] == self.hidden_dim

        t = self.cache.position

        query_t = self._project_heads(input_t, self.query_weights, self.query_bias)
        value_t = self._project_heads(input_t, self.values_weights, self.values_bias)

        slot = self.cache.decode_step(query_t, value_t, valid=valid)

        total_counts = self.cache.length[:, None].astype(GLOBAL_DTYPE) + self.memory_tokens
        total_counts = np.maximum(total_counts, 1.0)
        gate, descriptor = self._gate_from_sum_query(self.cache.sum_query, total_counts)

        if self.use_positional_phase:
            phase = np.conj(self.cache._twiddle[t % self.cache.max_sequence])
            gate = gate * phase[None, None, :]

        gate_aligned = self._align_gate(gate)
        gate_full = self._merge_heads(
            np.broadcast_to(
                gate_aligned,
                shape=(input_t.shape[0], self.cache.n_freq, self.num_heads, self.head_dim),
            )
        )
        window = self.cache.reconstruct(gate_full)
        output = window[:, slot, :]

        if self.use_wrm:
            output = self.wrm.forward(
                output[:, None, :], descriptor, training_now=False
            )[:, 0, :]

        return output


if __name__ == "__main__":

    def check_rfft_adjoint(N=17):
        x = np.random.randn(3, N, 5).astype(np.float64)
        g = (
                np.random.randn(3, N // 2 + 1, 5)
                + 1j * np.random.randn(3, N // 2 + 1, 5)
        )

        Ax = np.fft.rfft(x, axis=1)
        ATg = rfft_adjoint(g, N, axis=1)

        lhs = np.real(np.sum(np.conj(Ax) * g))
        rhs = np.sum(x * ATg)

        print(lhs, rhs, abs(lhs - rhs))


    def check_irfft_adjoint(N=17):
        g = (
                np.random.randn(3, N // 2 + 1, 5)
                + 1j * np.random.randn(3, N // 2 + 1, 5)
        )
        y = np.random.randn(3, N, 5)

        Ay = np.fft.irfft(g, n=N, axis=1)
        ATy = irfft_adjoint(y, N, axis=1)

        lhs = np.sum(Ay * y)
        rhs = np.real(np.sum(np.conj(g) * ATy))

        print(lhs, rhs, abs(lhs - rhs))


if __name__ == "__main__":
    from ml_tools.models.optimizers import SGD
    def _mse(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((pred - target) ** 2))


    def _mse_grad(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (2.0 / pred.size) * (pred - target)


    def _train_to_fit(layer, x, y, mask=None, steps=300, lr=5e-3, verbose=False):
        """
        Train `layer` with Adam to reproduce `y` from `x`.
        Returns (first_loss, last_loss).
        """
        optimizer = SGD(learning_rate=lr)
        losses = []

        for step in range(steps):
            layer.zero_gradients()  # belt-and-braces: some sub-layers accumulate via +=

            pred = layer.forward(x, mask=mask, training_now=True)
            loss = _mse(pred, y)
            losses.append(loss)

            grad_out = _mse_grad(pred, y)
            layer.backward(grad_out)
            optimizer.step([layer])

            if verbose and step % 50 == 0:
                print(f"step {step:4d}  loss {loss:.6f}")

        return losses[0], losses[-1]


    def test_spectre_attention_overfits_small_batch():
        """
        Plain (non-causal, no memory, no band, no WRM) SpectreAttention should
        be able to drive down the MSE on a small, fixed synthetic batch.
        """
        rng = np.random.default_rng(0)

        batch, seq_len, hidden, heads = 4, 8, 8, 2

        layer = SpectreAttention(
            sequence_length=seq_len,
            hidden_dim=hidden,
            num_heads=heads,
            band_radius=0,
            memory_tokens=0,
            use_wrm=False,
        )

        x = rng.normal(size=(batch, seq_len, hidden)).astype(np.float32)
        y = (rng.normal(size=(batch, seq_len, hidden)) * 0.5).astype(np.float32)

        first_loss, last_loss = _train_to_fit(layer, x, y, steps=300, lr=5e-3)

        print(f"[SpectreAttention] first_loss={first_loss:.6f} last_loss={last_loss:.6f}")

        assert np.isfinite(last_loss), "loss went non-finite -- likely a NaN/inf leak in fwd/bwd"
        assert last_loss < first_loss * 0.2, (
            f"expected loss to drop by at least 5x over training, "
            f"got {first_loss:.6f} -> {last_loss:.6f}"
        )


    def test_spectre_decoder_attention_overfits_with_memory_and_band():
        """
        SpectreDecoderAttention, trained the normal (non-cached) way per its own
        docstring, with memory tokens *and* a banded gate enabled -- the two
        extra mechanisms most likely to silently break gradients. Also exercises
        the masked-token path.
        """
        rng = np.random.default_rng(1)

        batch, seq_len, hidden, heads = 3, 6, 8, 2

        layer = SpectreDecoderAttention(
            sequence_length=seq_len,
            hidden_dim=hidden,
            num_heads=heads,
            band_radius=1,
            memory_tokens=2,
            use_wrm=False,
        )

        x = rng.normal(size=(batch, seq_len, hidden)).astype(np.float32)
        y = (rng.normal(size=(batch, seq_len, hidden)) * 0.5).astype(np.float32)

        # a chunk of forward/backward branches on mask/total_counts -- worth
        # covering here rather than in a separate test.
        mask = np.ones((batch, seq_len), dtype=np.float32)
        mask[:, -1] = 0.0  # last token padded out for every sample

        first_loss, last_loss = _train_to_fit(layer, x, y, mask=mask, steps=400, lr=5e-3)

        print(f"[SpectreDecoderAttention] first_loss={first_loss:.6f} last_loss={last_loss:.6f}")

        assert np.isfinite(last_loss)
        assert last_loss < first_loss * 0.2, (
            f"expected loss to drop by at least 5x over training, "
            f"got {first_loss:.6f} -> {last_loss:.6f}"
        )

    test_spectre_attention_overfits_small_batch()
    test_spectre_decoder_attention_overfits_with_memory_and_band()
    print("all good")