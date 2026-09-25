from typing import Callable, Optional

import numpy as np
from polyergalio.models.activations import mod_relu, mod_relu_derivative
from polyergalio.models.constants import EPSILON, GLOBAL_COMPLEX_DTYPE, GLOBAL_DTYPE
from polyergalio.models.layers.basal_layers import Layer, kaiming, xavier
from polyergalio.models.layers.wavelet_layers import WaveletRefinementModule
from numpy.typing import NDArray


# -------------    adjoints of the real FFT pair    ----------------
def rfft_adjoint(grad_freq: NDArray, sequence_length: int, axis: int = 1) -> NDArray:
    frequencies = grad_freq.shape[axis]
    pad_width = [(0, 0)] * grad_freq.ndim
    pad_width[axis] = (0, sequence_length - frequencies)
    padded = np.pad(grad_freq, pad_width)
    out = np.fft.ifft(padded, n=sequence_length, axis=axis) * sequence_length
    return out.real.astype(GLOBAL_DTYPE)


def irfft_adjoint(
    grad_time: NDArray,
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

        self.memory = xavier(self.RNG, ni=memory_tokens, no=hidden_dim).astype(
            GLOBAL_DTYPE
        )
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

    def __init__(
        self,
        sequence_length: int,
        hidden_dim: int,
        batch_size: int,
        memory_tokens: int = 0,
    ):
        self.sequence_length = int(sequence_length)
        self.memory_tokens = int(memory_tokens)
        self.max_sequence = self.memory_tokens + self.sequence_length
        self.hidden_dim = int(hidden_dim)
        self.batch_size = int(batch_size)
        self.n_freq = self.max_sequence // 2 + 1

        self.prefix_fft = np.zeros(
            shape=(self.batch_size, self.n_freq, self.hidden_dim),
            dtype=GLOBAL_COMPLEX_DTYPE,
        )
        self.value_buffer = np.zeros(
            shape=(self.batch_size, self.max_sequence, self.hidden_dim),
            dtype=GLOBAL_DTYPE,
        )
        self.query_buffer = np.zeros(
            shape=(self.batch_size, self.max_sequence, self.hidden_dim),
            dtype=GLOBAL_DTYPE,
        )
        self.mask_buffer = np.zeros((self.batch_size, self.max_sequence), dtype=bool)
        self.sum_query = np.zeros(
            (self.batch_size, self.hidden_dim), dtype=GLOBAL_DTYPE
        )

        # absolute step counter for the *sliding* part only; memory slots
        # are written once (in prefill / set_memory) and are never touched
        # by this counter.
        self.position = 0
        self.length = np.zeros(self.batch_size)
        self.memory_values = np.zeros((self.memory_tokens, self.hidden_dim), dtype=GLOBAL_DTYPE)
        self.memory_query_sum = np.zeros(self.hidden_dim, dtype=GLOBAL_DTYPE)

        k = np.arange(self.n_freq, dtype=GLOBAL_DTYPE)
        t = np.arange(self.max_sequence, dtype=GLOBAL_DTYPE)
        self._twiddle = np.exp(-2j * np.pi * np.outer(t, k) / self.max_sequence).astype(
            GLOBAL_COMPLEX_DTYPE
        )

    # set up and reset funcs ---------
    def reset(self):
        """clear the sliding window, keeping the persistent memory set by set_memory"""
        self.prefix_fft.fill(0)
        self.value_buffer.fill(0)
        self.query_buffer.fill(0)
        self.mask_buffer.fill(False)
        self.sum_query.fill(0)
        self.position = 0
        self.length.fill(0)
        if self.memory_tokens:
            self.value_buffer[:, : self.memory_tokens] = self.memory_values[None]
            self.mask_buffer[:, : self.memory_tokens] = True
            self.sum_query[...] = self.memory_query_sum[None]
            self.prefix_fft[...] = np.fft.rfft(self.value_buffer, n=self.max_sequence, axis=1)

    def set_memory(self, memory_values: np.ndarray, memory_queries: np.ndarray):
        """
        Seed the persistent memory slots, (memory_tokens, hidden_dim) each, already passed through the
        layer's value and query projections -- the training forward projects memory the same way, so
        memory contributes projected values to the mix and its queries to the pooled descriptor.
        Shared across the batch; clears the sliding window.
        """
        if self.memory_tokens == 0:
            return
        expected = (self.memory_tokens, self.hidden_dim)
        if memory_values.shape != expected or memory_queries.shape != expected:
            raise ValueError(
                f"expected memory shape {expected}, got {memory_values.shape} and {memory_queries.shape}"
            )
        self.memory_values = memory_values.astype(GLOBAL_DTYPE)
        self.memory_query_sum = memory_queries.sum(axis=0).astype(GLOBAL_DTYPE)
        self.reset()

    def prefill(
        self, query: np.ndarray, value: np.ndarray, mask: Optional[np.ndarray] = None
    ):
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
            raise ValueError(
                f"prompt length {length} exceeds window={self.sequence_length}"
            )
        if batch != self.batch_size:
            raise ValueError(f"cache batch_size={self.batch_size}, got {batch}")

        if mask is None:
            mask = np.ones((batch, length), dtype=GLOBAL_DTYPE)
        mask = mask.astype(GLOBAL_DTYPE)

        self.reset()

        query_valid = (query * mask[..., None]).astype(GLOBAL_DTYPE)
        value_valid = (value * mask[..., None]).astype(GLOBAL_DTYPE)

        start = self.memory_tokens
        self.value_buffer[:, start : start + length] = value_valid
        self.query_buffer[:, start : start + length] = query_valid
        self.mask_buffer[:, start : start + length] = mask.astype(bool)

        self.prefix_fft[...] = np.fft.rfft(
            self.value_buffer, n=self.max_sequence, axis=1
        ).astype(GLOBAL_COMPLEX_DTYPE)

        self.sum_query[...] = self.memory_query_sum[None] + query_valid.sum(axis=1)
        self.length[...] = mask.sum(axis=1).astype(np.int32)
        self.position = length

    # DECODE STEPS ------------------
    def decode_step(self, query_t: np.ndarray, value_t: np.ndarray, valid=True) -> int:
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
            old_slot = self.memory_tokens + (
                (t - self.sequence_length) % self.sequence_length
            )
            old_value = self.value_buffer[:, old_slot].copy()
            old_query = self.query_buffer[:, old_slot].copy()
            was_valid = self.mask_buffer[:, old_slot].copy()

            # evict using the same twiddle index
            self.prefix_fft -= (
                self._twiddle[old_slot, :][None, :, None] * old_value[:, None, :]
            )
            self.sum_query -= np.where(was_valid[:, None], old_query, 0.0)

        self.prefix_fft += self._twiddle[slot][None, :, None] * value_t[:, None, :]

        self.value_buffer[:, slot] = value_t
        self.query_buffer[:, slot] = query_t
        self.mask_buffer[:, slot] = valid
        self.sum_query += query_t

        self.position += 1
        self.length = np.minimum(
            self.length + valid.astype(np.int64), self.sequence_length
        )

        return slot

    # ------------------------------------------------------------------
    @property
    def live_length(self) -> int:
        return self.memory_tokens + int(min(self.position, self.sequence_length))

    def reconstruct(self, gate: np.ndarray, spectrum: Optional[np.ndarray] = None) -> np.ndarray:
        """
        gate : (batch, n_freq, hidden_dim) complex spectral gate, already
            broadcast/merged across heads (needs to be aligned before reconstruct)

        Returns the full ring-ordered reconstruction, shape (batch, max_sequence, hidden_dim).
        Slot ordering, not chronological ordering.
        see `read_slot` / `chronological_order` to extract a specific token or the whole window in seqence
        """
        spectrum = self.prefix_fft if spectrum is None else spectrum
        return np.fft.irfft(spectrum * gate, n=self.max_sequence, axis=1).astype(GLOBAL_DTYPE)

    def read_position(self, gate: np.ndarray, slot: int, spectrum: Optional[np.ndarray] = None) -> np.ndarray:
        """
        One position of the reconstruction, (batch, hidden_dim), without a full irfft: the gated
        spectrum phase-rotated to `slot` and summed over frequencies -- SPECTRE's positional phase,
        O(n_freq * hidden) per step. Equals reconstruct(gate)[:, slot].
        """
        weights = np.conj(self._twiddle[slot]) / self.max_sequence
        weights[1:] *= 2
        if self.max_sequence % 2 == 0:
            weights[-1] /= 2
        spectrum = self.prefix_fft if spectrum is None else spectrum
        return np.real(np.einsum("bkd,k->bd", spectrum * gate, weights)).astype(GLOBAL_DTYPE)

    def chronological_spectrum(self) -> np.ndarray:
        """
        rfft of memory followed by the window in oldest-to-newest order. Once the ring has wrapped,
        a window rotated behind fixed memory slots is no longer a circular shift of the training
        layout, so reads switch to this -- O(max_sequence log max_sequence) per step, memory only.
        """
        ordered = self.value_buffer[:, self.get_chronological_order()]
        return np.fft.rfft(ordered, n=self.max_sequence, axis=1)

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
            window_order = (
                np.arange(self.sequence_length) + newest_slot + 1
            ) % self.sequence_length
        return np.concatenate(
            [np.arange(self.memory_tokens), self.memory_tokens + window_order]
        )


DESCRIPTOR_EPS = 0.1
"""
LayerNorm epsilon for the pooled query. The mean of zero-mean queries over a sequence shrinks like
1/sqrt(length), so a standard 1e-6 epsilon divides sampling noise by its own tiny spread and the gate
swings on every weight update. 0.1 caps that amplification; the paper does not specify a value.
"""


class HeadGate:
    """
    SPECTRE's per-head spectral gate. Each head pools its own queries, layer-normalises them over
    head_dim, and maps them through its own two-layer MLP to one complex gate per frequency:

        gate_h = W2_h relu(W1_h LN_h(mean_q_h) + b1_h) + b2_h,  split into real | imaginary halves

    Heads share nothing, so a head's filter depends only on its own queries.

    Starts at the identity: b2 puts every gate at 1 + 0j after modReLU, and W2 is shrunk so content
    moves the gate gradually. A random gate starts each frequency at an arbitrary magnitude and,
    being bilinear with the values, diverges under plain SGD within a few steps.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_frequencies: int,
        gate_hidden: int,
        activation_bias: NDArray,
        rng: np.random.RandomState,
        eps: float = DESCRIPTOR_EPS,
        weight_scale: float = 0.1,
    ):
        """
        Parameters
        ----------
        num_heads, head_dim : head layout of the pooled query
        num_frequencies : gate entries per head
        gate_hidden : width of each head's hidden layer
        activation_bias : (num_heads, num_frequencies) modReLU bias, read once to centre the gate at 1
        rng : source for the MLP weights
        eps : LayerNorm epsilon, see DESCRIPTOR_EPS
        weight_scale : shrink on W2 at initialisation
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_frequencies = num_frequencies
        self.gate_hidden = gate_hidden
        self.eps = eps

        self.gamma = np.ones((num_heads, head_dim), dtype=GLOBAL_DTYPE)
        self.beta = np.zeros((num_heads, head_dim), dtype=GLOBAL_DTYPE)
        self.weights_1 = np.stack([kaiming(rng, ni=head_dim, no=gate_hidden) for _ in range(num_heads)])
        self.bias_1 = np.zeros((num_heads, gate_hidden), dtype=GLOBAL_DTYPE)
        self.weights_2 = weight_scale * np.stack(
            [xavier(rng, ni=gate_hidden, no=2 * num_frequencies) for _ in range(num_heads)]
        )
        self.bias_2 = np.zeros((num_heads, 2 * num_frequencies), dtype=GLOBAL_DTYPE)
        self.bias_2[:, :num_frequencies] = 1.0 - activation_bias

        self.purge()
        self.zero_gradients()

    def forward(self, pooled_query: NDArray) -> NDArray:
        """
        Parameters
        ----------
        pooled_query : (batch, hidden) mean query, heads laid out along the hidden axis

        Returns
        -------
        (batch, num_heads, num_frequencies) complex gate, before the band update and modReLU
        """
        heads = pooled_query.reshape(pooled_query.shape[0], self.num_heads, self.head_dim)
        centred = heads - heads.mean(axis=-1, keepdims=True)
        self.std = np.sqrt(np.mean(centred ** 2, axis=-1, keepdims=True) + self.eps)
        self.x_norm = centred / self.std
        self.normed = self.gamma * self.x_norm + self.beta

        self.hidden_pre = np.einsum("bhd,hdm->bhm", self.normed, self.weights_1) + self.bias_1
        self.hidden = np.maximum(self.hidden_pre, 0.0)
        projection = np.einsum("bhm,hmf->bhf", self.hidden, self.weights_2) + self.bias_2

        real, imaginary = np.split(projection, 2, axis=-1)
        return (real + 1j * imaginary).astype(GLOBAL_COMPLEX_DTYPE)

    @property
    def descriptor(self) -> NDArray:
        """(batch, hidden) per-head normalised pooled query, the conditioning the WRM reads"""
        return self.normed.reshape(self.normed.shape[0], -1)

    def backward(self, gate_gradient: NDArray, descriptor_gradient: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        gate_gradient : (batch, num_heads, num_frequencies) complex, dL/dRe + i dL/dIm
        descriptor_gradient : (batch, hidden) gradient reaching the descriptor from elsewhere (the WRM)

        Returns
        -------
        (batch, hidden) gradient on the pooled query
        """
        batch = gate_gradient.shape[0]
        projection_gradient = np.concatenate([gate_gradient.real, gate_gradient.imag], axis=-1)

        self.gradient_weights_2 = np.einsum("bhm,bhf->hmf", self.hidden, projection_gradient)
        self.gradient_bias_2 = projection_gradient.sum(axis=0)

        hidden_gradient = np.einsum("bhf,hmf->bhm", projection_gradient, self.weights_2) * (self.hidden_pre > 0)
        self.gradient_weights_1 = np.einsum("bhd,bhm->hdm", self.normed, hidden_gradient)
        self.gradient_bias_1 = hidden_gradient.sum(axis=0)

        normed_gradient = np.einsum("bhm,hdm->bhd", hidden_gradient, self.weights_1)
        if descriptor_gradient is not None:
            normed_gradient = normed_gradient + descriptor_gradient.reshape(normed_gradient.shape)
        self.gradient_gamma = np.sum(normed_gradient * self.x_norm, axis=0)
        self.gradient_beta = normed_gradient.sum(axis=0)

        x_gradient = normed_gradient * self.gamma
        heads_gradient = (
            x_gradient
            - x_gradient.mean(axis=-1, keepdims=True)
            - self.x_norm * np.mean(x_gradient * self.x_norm, axis=-1, keepdims=True)
        ) / self.std
        return heads_gradient.reshape(batch, -1)

    def get_weights(self, for_serialize: bool = False):
        weights = {
            "gamma": self.gamma,
            "beta": self.beta,
            "weights_1": self.weights_1,
            "bias_1": self.bias_1,
            "weights_2": self.weights_2,
            "bias_2": self.bias_2,
        }
        return weights if for_serialize else tuple(weights.values())

    def set_weights(self, weights: dict) -> None:
        for name in ("gamma", "beta", "weights_1", "bias_1", "weights_2", "bias_2"):
            setattr(self, name, np.asarray(weights[name], dtype=GLOBAL_DTYPE))

    def get_gradients(self) -> dict[str, NDArray]:
        return {
            "gradient_gamma": self.gradient_gamma,
            "gradient_beta": self.gradient_beta,
            "gradient_weights_1": self.gradient_weights_1,
            "gradient_bias_1": self.gradient_bias_1,
            "gradient_weights_2": self.gradient_weights_2,
            "gradient_bias_2": self.gradient_bias_2,
        }

    def update_weights(
        self,
        gradient_gamma: NDArray,
        gradient_beta: NDArray,
        gradient_weights_1: NDArray,
        gradient_bias_1: NDArray,
        gradient_weights_2: NDArray,
        gradient_bias_2: NDArray,
    ) -> None:
        self.gamma -= gradient_gamma
        self.beta -= gradient_beta
        self.weights_1 -= gradient_weights_1
        self.bias_1 -= gradient_bias_1
        self.weights_2 -= gradient_weights_2
        self.bias_2 -= gradient_bias_2

    def zero_gradients(self) -> None:
        self.gradient_gamma = np.zeros_like(self.gamma)
        self.gradient_beta = np.zeros_like(self.beta)
        self.gradient_weights_1 = np.zeros_like(self.weights_1)
        self.gradient_bias_1 = np.zeros_like(self.bias_1)
        self.gradient_weights_2 = np.zeros_like(self.weights_2)
        self.gradient_bias_2 = np.zeros_like(self.bias_2)

    def purge(self) -> None:
        self.std = None
        self.x_norm = None
        self.normed = None
        self.hidden_pre = None
        self.hidden = None

    @property
    def num_parameters(self) -> int:
        return sum(weight.size for weight in self.get_weights(for_serialize=True).values())


class SpectreAttention(Layer):
    """
    SPECTRE mixing layer, https://arxiv.org/abs/2502.18394

    Query/value projections are per-head (num_heads independent head_dim x
    head_dim maps), matching the paper's W(q), W(v) per head, rather than one
    shared hidden_dim x hidden_dim projection split afterward.

    """

    registry_name = "SPECTREAttention"
    preserves_shape = True

    def __init__(
        self,
        sequence_length: int,
        hidden_dim: int,
        num_heads: int = 1,
        band_radius: int = 0,
        memory_tokens: int = 0,
        causal_decode: bool = False,
        modrelu_bias: float = -0.1,
        use_wrm: bool = False,
        use_positional_phase: bool = True,
        gate_hidden: Optional[int] = None,
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
        gate_hidden : width of each head's gate MLP hidden layer, head_dim if None
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

        if self.memory_tokens > 0:
            self.memory = PersistentMemory(
                memory_tokens=self.memory_tokens, hidden_dim=self.hidden_dim
            )

        self.fft_length = self.sequence_length + self.memory_tokens

        self._cache = None
        self._last_forward = None

        # sequence axis is pinned, not wildcarded: fft_length / num_frequencies
        # are sized off sequence_length at construction time, so a mismatched
        # sequence length here is a real, catchable error, not a free axis.
        self.declare_shapes(
            inputs=((self.sequence_length, self.hidden_dim),),
            outputs=((self.sequence_length, self.hidden_dim),),
        )

        self.num_frequencies = self.fft_length // 2 + 1
        self.activation_bias = np.full((num_heads, self.num_frequencies), modrelu_bias, dtype=GLOBAL_DTYPE)

        # per-head independent projections
        self.query_weights = self.init_head_projection()
        self.query_bias = np.zeros((num_heads, self.head_dim), dtype=GLOBAL_DTYPE)
        self.values_weights = self.init_head_projection()
        self.values_bias = np.zeros((num_heads, self.head_dim), dtype=GLOBAL_DTYPE)

        self.gate_hidden = gate_hidden
        self.head_gate = HeadGate(
            num_heads=num_heads,
            head_dim=self.head_dim,
            num_frequencies=self.num_frequencies,
            gate_hidden=gate_hidden or self.head_dim,
            activation_bias=self.activation_bias,
            rng=self.RNG,
        )

        assert band_radius >= 0, (
            f"band_radius must be zero or positive, got {band_radius}. A "
            "negative radius produces no taps and silently disables the gate."
        )

        # band radius is a dimensionality reduction mechanism
        self.band_radius = band_radius
        self.band_offsets = tuple(range(-band_radius, band_radius + 1))
        # every tap acts on all num_frequencies bins at once, so its gradient grows with sequence
        # length; scaling taps by 1/sqrt(num_frequencies) keeps their step size length-independent
        self.band_scale = 1.0 / np.sqrt(self.num_frequencies)
        if band_radius:
            self.band_taps = np.zeros(
                (num_heads, len(self.band_offsets)), dtype=GLOBAL_COMPLEX_DTYPE
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
            [
                xavier(
                    self.RNG,
                    ni=self.head_dim,
                    no=self.head_dim,
                )
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

    def _project_heads(
        self, input_data: NDArray, weights: NDArray, bias: NDArray
    ) -> NDArray:
        """
        Independent (head_dim, head_dim) map per head.

        input_data : (..., hidden)
        weights : (num_heads, head_dim, head_dim)
        bias : (num_heads, head_dim)
        """
        heads = self._split_heads(input_data)
        projected = np.einsum("...hd,hde->...he", heads, weights) + bias
        return self._merge_heads(projected)

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

        dweights = np.einsum(
            "nhd,nhe->hde",
            flat_input,
            flat_doutput,
            optimize=True,
        )

        dbias = flat_doutput.sum(axis=0)
        dinput_heads = np.einsum(
            "...he,hde->...hd",
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
            banded += self.band_scale * self.band_taps[:, index, None] * self._shift(gate, offset)
        return gate + banded

    def forward(
        self,
        input_data: NDArray,
        mask: Optional[NDArray] = None,
        training_now: Optional[bool] = None,
    ):
        training_now = self.training if training_now is None else training_now
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

        self.counts = np.maximum(self.mask.sum(axis=1, keepdims=True), 1.0)
        mask_column = self.mask[..., None]

        if self.memory_tokens:
            memory = self.memory.get_memory()
            memory_batch = np.broadcast_to(
                memory[None, :, :], (batch, self.memory_tokens, self.hidden_dim)
            )
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
            memory_query = query_all[:, : self.memory_tokens]
            query_forward = query_all[:, self.memory_tokens :]

            memory_value = value_all[:, : self.memory_tokens]
            value_forward = value_all[:, self.memory_tokens :]

            value_forward = value_forward * mask_column

            self.total_counts = self.counts + self.memory_tokens

            seq_sum = (query_forward * mask_column).sum(axis=1)
            seq_sum += memory_query.sum(axis=1)

            value_masked = np.concatenate([memory_value, value_forward], axis=1)

        else:
            query_forward = query_all
            value_forward = value_all * mask_column

            self.total_counts = self.counts

            seq_sum = (query_forward * mask_column).sum(axis=1)

            value_masked = value_forward

        self.seq_mu = seq_sum / self.total_counts
        self.gate_raw = self.head_gate.forward(self.seq_mu)
        self.descriptor = self.head_gate.descriptor

        if self.band_radius:
            self.gate_pre_activation = self._band_update(self.gate_raw)
        else:
            self.gate_pre_activation = self.gate_raw

        self.gate = self.activation(self.gate_pre_activation, self.activation_bias)

        # Full combined sequence gets a single transform into frequency space:
        self.combined_length = total_length
        self.num_combined_frequencies = total_length // 2 + 1

        self.value_transform = np.fft.rfft(
            value_masked,
            n=total_length,
            axis=1,
        )

        values_gated = self._split_heads(self.value_transform) * self._align_gate(
            self.gate
        )

        output_all = np.fft.irfft(
            self._merge_heads(values_gated),
            n=total_length,
            axis=1,
        )

        if self.memory_tokens:
            self.output = output_all[:, self.memory_tokens :]
        else:
            self.output = output_all

        if self.use_wrm:
            self.output = self.wrm.forward(
                self.output, self.descriptor, training_now=self.training_now
            )

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
            full_gradient[:, self.memory_tokens :] = incoming_gradient
        else:
            full_gradient = incoming_gradient

        dvalues_spec_gated = self._split_heads(
            irfft_adjoint(full_gradient, self.combined_length, axis=1)
        )

        # split the gradients back across our heads -----
        value_heads = self._split_heads(self.value_transform)

        dV_hat = self._merge_heads(
            dvalues_spec_gated * np.conj(self._align_gate(self.gate))
        )
        dgate = np.transpose(
            np.sum(dvalues_spec_gated * np.conj(value_heads), axis=-1), (0, 2, 1)
        )

        self.gradient_bias, dgate_pre = self.activation_derivative(
            z=self.gate_pre_activation,
            beta=self.activation_bias,
            dout=dgate,
        )

        if self.band_radius:
            self.gradient_band = np.stack(
                [
                    self.band_scale * np.sum(dgate_pre * np.conj(self._shift(self.gate_raw, offset)), axis=(0, 2))
                    for offset in self.band_offsets
                ],
                axis=-1,
            )
            dgate_raw = dgate_pre.copy()
            for index, offset in enumerate(self.band_offsets):
                dgate_raw += self.band_scale * np.conj(self.band_taps[:, index, None]) * self._shift(dgate_pre, -offset)
            dgate_pre = dgate_raw

        dseq_mu = self.head_gate.backward(dgate_pre, d_descriptor_wrm)

        dquery = mask_column * (dseq_mu[:, None, :] / self.total_counts[..., None])
        dvalues = rfft_adjoint(dV_hat, self.combined_length, axis=1)

        if self.memory_tokens:
            d_memory_query = np.broadcast_to(
                (dseq_mu / self.total_counts)[:, None, :],
                (dseq_mu.shape[0], self.memory_tokens, self.hidden_dim),
            )
            d_memory_value = dvalues[:, : self.memory_tokens]
            dvalues = dvalues[:, self.memory_tokens :] * mask_column

            dquery_all = np.concatenate([d_memory_query, dquery], axis=1).astype(
                GLOBAL_DTYPE
            )
            dvalue_all = np.concatenate([d_memory_value, dvalues], axis=1).astype(
                GLOBAL_DTYPE
            )

        else:
            dvalues = dvalues * mask_column
            dquery_all = dquery.astype(GLOBAL_DTYPE)
            dvalue_all = dvalues.astype(GLOBAL_DTYPE)

        self.gradient_query_weights, self.gradient_query_bias, dinput_from_q_all = (
            self._project_heads_backward(
                self.combined_input, dquery_all, self.query_weights
            )
        )
        self.gradient_values_weights, self.gradient_values_bias, dinput_from_v_all = (
            self._project_heads_backward(
                self.combined_input, dvalue_all, self.values_weights
            )
        )

        if self.memory_tokens:
            self.memory.backward(
                dinput_from_q_all[:, : self.memory_tokens].sum(axis=0)
                + dinput_from_v_all[:, : self.memory_tokens].sum(axis=0)
            )
            dinput_from_q = dinput_from_q_all[:, self.memory_tokens :]
            dinput_from_v = dinput_from_v_all[:, self.memory_tokens :]
        else:
            dinput_from_q = dinput_from_q_all
            dinput_from_v = dinput_from_v_all

        grad_real = (dinput_from_q + dinput_from_v).real.astype(GLOBAL_DTYPE)
        return grad_real

    def get_weights(self, for_serialize: bool = False) -> tuple | dict:
        if for_serialize:
            weights = {
                "activation_bias": self.activation_bias,
                "query_weights": self.query_weights,
                "query_bias": self.query_bias,
                "values_weights": self.values_weights,
                "values_bias": self.values_bias,
                "head_gate": self.head_gate.get_weights(for_serialize=True),
            }
            if self.band_radius:
                weights["band_taps"] = self.band_taps

            if self.use_wrm:
                weights["wrm"] = self.wrm.get_weights(for_serialize=True)

            if self.memory_tokens:
                weights["persistent_memory"] = self.memory.get_weights(
                    for_serialize=True
                )

            return weights

        weights = [
            self.activation_bias,
            self.query_weights,
            self.query_bias,
            self.values_weights,
            self.values_bias,
            self.head_gate.get_weights(for_serialize=False),
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
        self.activation_bias = np.asarray(
            weights["activation_bias"], dtype=GLOBAL_DTYPE
        )
        self.query_weights = np.asarray(weights["query_weights"], dtype=GLOBAL_DTYPE)
        self.query_bias = np.asarray(weights["query_bias"], dtype=GLOBAL_DTYPE)
        self.values_weights = np.asarray(weights["values_weights"], dtype=GLOBAL_DTYPE)
        self.values_bias = np.asarray(weights["values_bias"], dtype=GLOBAL_DTYPE)
        self.head_gate.set_weights(weights["head_gate"])

        if self.band_radius and "band_taps" in weights:
            self.band_taps = np.asarray(
                weights["band_taps"], dtype=GLOBAL_COMPLEX_DTYPE
            )

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
            "head_gate": self.head_gate.get_gradients(),
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

        self.head_gate.purge()

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
        self.head_gate.zero_gradients()

        if self.band_radius:
            self.gradient_band = np.zeros_like(self.band_taps)

        if self.memory_tokens:
            self.memory.zero_gradients()

        if self.use_wrm:
            self.wrm.zero_gradients()

    @property
    def num_parameters(self) -> int:
        total = (
            self.activation_bias.size
            + self.query_weights.size
            + self.query_bias.size
            + self.values_weights.size
            + self.values_bias.size
            + self.head_gate.num_parameters
        )

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
        head_gate: dict[str, NDArray],
        persistent_memory: dict[str, NDArray] = None,
        gradient_band: NDArray = None,
        wrm: dict = None,
    ) -> None:

        self.activation_bias -= gradient_bias
        self.query_weights -= gradient_query_weights
        self.query_bias -= gradient_query_bias
        self.values_weights -= gradient_values_weights
        self.values_bias -= gradient_values_bias

        self.head_gate.update_weights(**head_gate)

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
    Causal, autoregressive companion to `SpectreAttention`.

    Training and decoding compute the same function. Position p reads only values at positions <= p
    (a causal linear convolution, not the encoder's circular one), filtered by a gate built from the
    pooled query of its chunk's first token and everything before it

        anchor(p) = (p // chunk_size) * chunk_size
        gate(p) = gate from mean query over memory + tokens <= anchor(p)
        y_p = sum_{j <= p} h_anchor(p)[p - j] v_j,  h = irfft(gate)

    chunk_size=1 refreshes the gate at every token -- exact per-prefix gating at attention-sized
    O(T * N log N) training cost. Larger chunks cost about T / chunk_size FFT convolutions, and a
    chunk's gate lags its tokens by up to chunk_size - 1. Decoding refreshes the gate on the same
    anchors.

    Usage
    -----
        layer = SpectreDecoderAttention(sequence_length=..., hidden_dim=..., num_heads=..., chunk_size=...)

        out = layer.forward(batch_x, mask=batch_mask, training_now=True)
        layer.backward(delta_out)

        last_hidden = layer.prefill(prompt_embeddings, mask=prompt_mask)
        for _ in range(n_new_tokens):
            last_hidden = layer.decode_step(next_token_embedding)

    Past sequence_length tokens decoding slides its window; with chunk_size=1 each step still equals a
    training forward over the latest window, with larger chunks the anchors no longer line up with one.

    use_wrm is not supported: the Wavelet Refinement Module needs the whole window at once, but
    decoding produces one token at a time.
    """

    registry_name = "SPECTREDecoderAttention"

    def __init__(
        self,
        sequence_length: int,
        hidden_dim: int,
        num_heads: int = 1,
        band_radius: int = 0,
        memory_tokens: int = 0,
        causal_decode: bool = False,
        modrelu_bias: float = -0.1,
        use_wrm: bool = False,
        use_positional_phase: bool = True,
        gate_hidden: Optional[int] = None,
        chunk_size: int = 1,
    ):
        """
        Parameters
        ----------
        chunk_size : tokens that share one gate, see the class docstring. 1 is exact per-token gating
        other parameters : see SpectreAttention
        """
        assert not use_wrm, (
            "SpectreDecoderAttention does not support use_wrm"
        )
        assert 1 <= chunk_size <= sequence_length, (
            f"chunk_size must fall in [1, {sequence_length}], got {chunk_size}"
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
            gate_hidden=gate_hidden,
        )
        self.chunk_size = chunk_size
        self.cache: Optional[PrefixFFTCache] = None
        self.chunk_gate: Optional[NDArray] = None

    # ------------- causal training forward / backward
    def forward(
        self,
        input_data: NDArray,
        mask: Optional[NDArray] = None,
        training_now: Optional[bool] = None,
    ) -> NDArray:
        """
        Parameters
        ----------
        input_data : (batch, sequence_length, hidden_dim)
        mask : (batch, sequence_length), 1 for a real token and 0 for padding
        training_now : unused, accepted for the graph

        Returns
        -------
        (batch, sequence_length, hidden_dim), each position mixing only itself and earlier positions
        """
        assert input_data.ndim == 3
        batch, length, _ = input_data.shape
        assert length == self.sequence_length and input_data.shape[2] == self.hidden_dim
        memory_tokens = self.memory_tokens

        self.input = input_data
        self.mask = (
            mask.astype(GLOBAL_DTYPE) if mask is not None else np.ones((batch, length), dtype=GLOBAL_DTYPE)
        )
        mask_column = self.mask[..., None]

        if memory_tokens:
            memory = np.broadcast_to(self.memory.get_memory()[None], (batch, memory_tokens, self.hidden_dim))
            combined = np.concatenate([memory, input_data], axis=1)
        else:
            combined = input_data
        self.combined_input = combined
        self.combined_length = combined.shape[1]

        query_all = self._project_heads(combined, self.query_weights, self.query_bias)
        value_all = self._project_heads(combined, self.values_weights, self.values_bias)
        values = value_all.copy()
        values[:, memory_tokens:] *= mask_column

        self.anchors = np.arange(0, length, self.chunk_size)
        prefix_queries = np.cumsum(query_all[:, memory_tokens:] * mask_column, axis=1)[:, self.anchors]
        prefix_counts = np.cumsum(self.mask, axis=1)[:, self.anchors]
        self.chunk_counts = (np.maximum(prefix_counts, 1.0) + memory_tokens)[..., None]
        memory_query_sum = query_all[:, :memory_tokens].sum(axis=1, keepdims=True)
        pooled = (prefix_queries + memory_query_sum) / self.chunk_counts

        num_chunks = self.anchors.size
        self.gate_raw = self.head_gate.forward(pooled.reshape(batch * num_chunks, self.hidden_dim))
        self.gate_pre_activation = self._band_update(self.gate_raw) if self.band_radius else self.gate_raw
        self.gate = self.activation(self.gate_pre_activation, self.activation_bias)

        padded_length = 2 * self.combined_length
        self.value_transform = np.fft.rfft(values, n=padded_length, axis=1)
        chunk_gates = self.gate.reshape(batch, num_chunks, self.num_heads, self.num_frequencies)
        self.filter_transforms = []
        self.output = np.zeros_like(input_data)
        for chunk, anchor in enumerate(self.anchors):
            rows = slice(anchor, min(anchor + self.chunk_size, length))
            filters = np.fft.irfft(chunk_gates[:, chunk], n=self.combined_length, axis=-1)
            filter_transform = np.fft.rfft(filters, n=padded_length, axis=-1)
            self.filter_transforms.append(filter_transform)
            mixed = self._split_heads(self.value_transform) * np.transpose(filter_transform, (0, 2, 1))[..., None]
            convolved = np.fft.irfft(self._merge_heads(mixed), n=padded_length, axis=1)
            self.output[:, rows] = convolved[:, memory_tokens + rows.start: memory_tokens + rows.stop]
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        batch, length, _ = incoming_gradient.shape
        memory_tokens = self.memory_tokens
        padded_length = 2 * self.combined_length
        mask_column = self.mask[..., None]
        value_heads = self._split_heads(self.value_transform)

        value_transform_gradient = np.zeros_like(self.value_transform)
        gate_gradient = np.zeros((batch, self.anchors.size, self.num_heads, self.num_frequencies), dtype=GLOBAL_COMPLEX_DTYPE)
        for chunk, anchor in enumerate(self.anchors):
            rows = slice(anchor, min(anchor + self.chunk_size, length))
            convolved_gradient = np.zeros((batch, padded_length, self.hidden_dim))
            convolved_gradient[:, memory_tokens + rows.start: memory_tokens + rows.stop] = incoming_gradient[:, rows]
            mixed_gradient = self._split_heads(irfft_adjoint(convolved_gradient, padded_length, axis=1))

            filter_transform = np.transpose(self.filter_transforms[chunk], (0, 2, 1))[..., None]
            value_transform_gradient += self._merge_heads(mixed_gradient * np.conj(filter_transform))
            filter_transform_gradient = np.sum(mixed_gradient * np.conj(value_heads), axis=-1)
            filter_gradient = rfft_adjoint(filter_transform_gradient, padded_length, axis=1)[:, : self.combined_length]
            gate_gradient[:, chunk] = np.transpose(irfft_adjoint(filter_gradient, self.combined_length, axis=1), (0, 2, 1))

        gate_gradient = gate_gradient.reshape(-1, self.num_heads, self.num_frequencies)
        self.gradient_bias, gate_pre_gradient = self.activation_derivative(
            z=self.gate_pre_activation, beta=self.activation_bias, dout=gate_gradient
        )
        if self.band_radius:
            self.gradient_band = np.stack(
                [
                    self.band_scale * np.sum(gate_pre_gradient * np.conj(self._shift(self.gate_raw, offset)), axis=(0, 2))
                    for offset in self.band_offsets
                ],
                axis=-1,
            )
            raw_gradient = gate_pre_gradient.copy()
            for index, offset in enumerate(self.band_offsets):
                raw_gradient += self.band_scale * np.conj(self.band_taps[:, index, None]) * self._shift(gate_pre_gradient, -offset)
            gate_pre_gradient = raw_gradient

        pooled_gradient = self.head_gate.backward(gate_pre_gradient).reshape(batch, self.anchors.size, self.hidden_dim)
        per_anchor = pooled_gradient / self.chunk_counts
        reaching = np.cumsum(per_anchor[:, ::-1], axis=1)[:, ::-1]
        first_chunk = -(-np.arange(length) // self.chunk_size)
        covered = first_chunk < self.anchors.size
        token_query_gradient = np.zeros((batch, length, self.hidden_dim))
        token_query_gradient[:, covered] = reaching[:, first_chunk[covered]]
        token_query_gradient *= mask_column

        value_gradient = rfft_adjoint(value_transform_gradient, padded_length, axis=1)[:, : self.combined_length]
        value_gradient[:, memory_tokens:] *= mask_column
        query_gradient = np.concatenate(
            [np.broadcast_to(reaching[:, :1], (batch, memory_tokens, self.hidden_dim)), token_query_gradient], axis=1
        )

        self.gradient_query_weights, self.gradient_query_bias, from_queries = self._project_heads_backward(
            self.combined_input, query_gradient, self.query_weights
        )
        self.gradient_values_weights, self.gradient_values_bias, from_values = self._project_heads_backward(
            self.combined_input, value_gradient, self.values_weights
        )
        input_gradient = from_queries + from_values
        if memory_tokens:
            self.memory.backward(input_gradient[:, :memory_tokens].sum(axis=0))
        return input_gradient[:, memory_tokens:].astype(GLOBAL_DTYPE)

    # ------------- decoding
    def reset_cache(self, batch_size: int):
        self.cache = PrefixFFTCache(
            sequence_length=self.sequence_length,
            hidden_dim=self.hidden_dim,
            batch_size=batch_size,
            memory_tokens=self.memory_tokens,
        )
        self.chunk_gate = None
        if self.memory_tokens:
            memory = self.memory.get_memory()
            self.cache.set_memory(
                self._project_heads(memory, self.values_weights, self.values_bias),
                self._project_heads(memory, self.query_weights, self.query_bias),
            )

    def gate_from_pooled_sum(self, query_sum: NDArray, total_counts: NDArray) -> NDArray:
        """
        query_sum, total_counts : (batch, hidden_dim), (batch, 1)

        Returns
        -------
        (batch, num_frequencies, hidden_dim) activated gate, each head's gate repeated over its channels
        """
        gate_raw = self.head_gate.forward(query_sum / total_counts)
        gate_pre = self._band_update(gate_raw) if self.band_radius else gate_raw
        gate = self.activation(gate_pre, self.activation_bias)
        batch = gate.shape[0]
        return self._merge_heads(
            np.broadcast_to(self._align_gate(gate), (batch, self.num_frequencies, self.num_heads, self.head_dim))
        )

    def read_slot(self, gate_full: NDArray, slot: int) -> NDArray:
        """
        the live token's output: the paper's phase-rotated frequency sum when use_positional_phase,
        otherwise a full-window irfft indexed at slot. Both give the same values.
        """
        spectrum = None
        if self.memory_tokens and self.cache.position > self.sequence_length:
            spectrum = self.cache.chronological_spectrum()
            slot = self.cache.max_sequence - 1
        if self.use_positional_phase:
            return self.cache.read_position(gate_full, slot, spectrum)
        return self.cache.reconstruct(gate_full, spectrum)[:, slot, :]

    def prefill(self, input_data: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Process a prompt, populate the Prefix-FFT cache, and return the last prompt token's output.

        input_data : (batch, L, hidden_dim), L <= sequence_length
        mask : (batch, L) optional
        """
        assert input_data.ndim == 3
        batch, length, hidden_dim = input_data.shape
        assert hidden_dim == self.hidden_dim

        self.reset_cache(batch_size=batch)
        mask = mask.astype(GLOBAL_DTYPE) if mask is not None else np.ones((batch, length), dtype=GLOBAL_DTYPE)

        query_all = self._project_heads(input_data, self.query_weights, self.query_bias)
        value_all = self._project_heads(input_data, self.values_weights, self.values_bias)
        self.cache.prefill(query_all, value_all, mask=mask)

        anchor = ((length - 1) // self.chunk_size) * self.chunk_size
        query_sum = self.cache.memory_query_sum[None] + (query_all[:, : anchor + 1] * mask[:, : anchor + 1, None]).sum(axis=1)
        total_counts = np.maximum(mask[:, : anchor + 1].sum(axis=1, keepdims=True), 1.0) + self.memory_tokens
        self.chunk_gate = self.gate_from_pooled_sum(query_sum, total_counts)

        last_slot = self.memory_tokens + ((length - 1) % self.sequence_length)
        return self.read_slot(self.chunk_gate, last_slot)

    def decode_step(self, input_t: NDArray, valid=True) -> NDArray:
        """
        Append one new token and return its output. The gate is rebuilt from the cache's pooled query
        when the token starts a chunk, and reused otherwise.

        input_t : (batch, hidden_dim) raw embedding for the position
        valid : bool or (batch,) bools: False marks a padding step for finished sequences; it still
            advances the cache but writes a zero token and does not affect the pooled query.
        """
        assert self.cache is not None, "call reset_cache()/prefill() first"
        assert input_t.ndim == 2 and input_t.shape[1] == self.hidden_dim

        position = self.cache.position
        query_t = self._project_heads(input_t, self.query_weights, self.query_bias)
        value_t = self._project_heads(input_t, self.values_weights, self.values_bias)
        slot = self.cache.decode_step(query_t, value_t, valid=valid)

        if self.chunk_gate is None or position % self.chunk_size == 0:
            total_counts = np.maximum(self.cache.length[:, None].astype(GLOBAL_DTYPE), 1.0) + self.memory_tokens
            self.chunk_gate = self.gate_from_pooled_sum(self.cache.sum_query, total_counts)
        return self.read_slot(self.chunk_gate, slot)
