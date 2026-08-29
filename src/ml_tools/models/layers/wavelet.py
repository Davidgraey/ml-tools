"""
Shared wavelet machinery: filter families and one level of learnable transform.

Everything here is about a single question -- how to split a sequence into a
coarse half and a fine half, reversibly, with taps that can be learned. What
gets built on top of that split is elsewhere: wavelet_layers.py cascades it down
and back up as a replacement for self-attention, signal_encoder.py grows it into
a full packet tree and reads features off the bands.

Filters follow the Daubechies half-band construction. Set family="haar" for the
two tap case, "db2" and up for smoother bands, "sym4" and up to spend the same
taps on near-linear phase instead of minimum phase.
"""

import numpy as np
from numpy.typing import NDArray
from ml_tools.models.constants import GLOBAL_DTYPE
from ml_tools.models.layers.layers import Layer


# Orthonormal scaling filters. Each satisfies sum(h**2) == 1 and
# sum(h[n] * h[n + 2m]) == 0 for m != 0, which is what makes the analysis
# operator orthogonal and its adjoint an exact inverse.
#
# Both families come from the same spectral factorisation of the Daubechies
# half-band condition, so a symlet has exactly the same support and the same
# number of vanishing moments as the Daubechies filter of its order. They
# differ only in which member of each reciprocal root pair is kept: Daubechies
# takes every root inside the unit circle, giving minimum phase, while a symlet
# swaps some out to bring the phase closer to linear. That makes the symlets
# nearly symmetric, so a feature lands where it occurred instead of being
# smeared to one side -- worth having when the position of an event in the
# sequence carries meaning.
#
# Because the construction is shared, sym1, sym2 and sym3 are the same filters
# as haar, db2 and db3: below order four there is nothing to trade. They are
# left out rather than aliased, so every key here is a distinct filter.
#
# Values are computed from the factorisation with Newton-polished roots rather
# than copied from a published table. They agree with the usual tables to about
# 1e-12, which is the accuracy of those tables, while satisfying the two
# orthonormality conditions to ~1e-16. That margin is what lets a
# reconstruction test tell a real bug from truncated literals.
WAVELET_FAMILIES: dict[str, tuple[float, ...]] = {
    "haar": (
        0.7071067811865475,
        0.7071067811865475,
    ),
    "db2": (
        0.4829629131445342,
        0.8365163037378079,
        0.22414386804201342,
        -0.1294095225512604,
    ),
    "db3": (
        0.33267055295008263,
        0.8068915093110925,
        0.45987750211849165,
        -0.1350110200102545,
        -0.08544127388202666,
        0.035226291885709526,
    ),
    "db4": (
        0.23037781330889648,
        0.7148465705529157,
        0.6308807679298589,
        -0.02798376941685985,
        -0.18703481171909306,
        0.030841381835560767,
        0.03288301166688518,
        -0.010597401785069037,
    ),
    "sym4": (
        -0.07576571478950221,
        -0.029635527646002472,
        0.497618667632775,
        0.8037387518051321,
        0.297857795605306,
        -0.09921954357663357,
        -0.012603967262031309,
        0.03222310060405147,
    ),
    "sym5": (
        0.027333068344998768,
        0.029519490925706257,
        -0.039134249302313844,
        0.1993975339768556,
        0.7234076904040408,
        0.6339789634567922,
        0.016602105764511002,
        -0.17532808990805615,
        -0.021101834024689042,
        0.019538882735249823,
    ),
    "sym6": (
        0.015404109327044845,
        0.0034907120842221848,
        -0.11799011114852015,
        -0.04831174258569822,
        0.4910559419279739,
        0.7876411410286511,
        0.33792942172816537,
        -0.0726375227863771,
        -0.021060292512370987,
        0.044724901770781415,
        0.0017677118642540008,
        -0.0078007083250323924,
    ),
    "sym7": (
        0.0026818145682601497,
        -0.001047384888679734,
        -0.012636303403240567,
        0.030515513165877903,
        0.06789269350122056,
        -0.049552834937043086,
        0.01744125508683519,
        0.5361019170905689,
        0.767764317004883,
        0.28862963175064815,
        -0.14004724044293357,
        -0.10780823770328973,
        0.0040102448715223895,
        0.010268176708464818,
    ),
    "sym8": (
        -0.003382415951005008,
        -0.0005421323318000154,
        0.03169508781152603,
        0.007607487324976536,
        -0.14329423835127322,
        -0.061273359067812005,
        0.48135965125905283,
        0.7771857516996281,
        0.3644418948361791,
        -0.05194583810788176,
        -0.027219029917103378,
        0.049137179673730394,
        0.0038087520138944957,
        -0.01495225833706222,
        -0.0003029205147241336,
        0.0018899503327676917,
    ),
}


def quadrature_mirror(low_taps: NDArray) -> NDArray:
    """
    The high pass partner of a scaling filter, g[n] = (-1)**n * h[K-1-n].

    Reversing and alternating signs is what puts the two filters in different
    halves of the band, so the pair spans the input rather than measuring it
    twice. Leading axis is the tap axis, so this accepts both the 1D family
    table and the (taps, hidden) learnable array.
    """
    signs = (-1.0) ** np.arange(low_taps.shape[0])
    return low_taps[::-1] * signs.reshape(-1, *([1] * (low_taps.ndim - 1)))


def dyadic_indices(sequence_length: int, num_taps: int) -> NDArray:
    """
    Gather map for stride-two filtering with periodic boundaries.

    Row p holds the positions the p-th output coefficient reads, so filtering
    becomes one fancy-index gather and a contraction over the tap axis. The
    modulo is the wrap: a filter longer than two taps runs off the end of the
    sequence, and wrapping keeps the operator orthogonal where zero padding
    would not.
    """
    starts = 2 * np.arange(sequence_length // 2)
    return (starts[:, None] + np.arange(num_taps)[None, :]) % sequence_length


def family_taps(family: str, hidden_dim: int) -> tuple[NDArray, NDArray]:
    """(taps, hidden) low and high pass arrays, one independent copy per channel"""
    assert family in WAVELET_FAMILIES, (
        f"unknown family {family!r}, expected one of {sorted(WAVELET_FAMILIES)}"
    )
    low = np.asarray(WAVELET_FAMILIES[family], dtype=GLOBAL_DTYPE)
    return (
        np.repeat(low[:, None], hidden_dim, axis=1),
        np.repeat(quadrature_mirror(low)[:, None], hidden_dim, axis=1),
    )


def phase_nonlinearity(low_taps: NDArray) -> float:
    """
    How far a filter's phase departs from linear, in radians.

    A linear phase filter delays every frequency by the same amount, so a
    feature keeps its position. Anything else spreads it, and the spread is
    frequency dependent. Dividing out the bulk delay of (K-1)/2 samples leaves
    only the part that cannot be corrected by shifting, and the largest value
    of that residual is what this reports.

    Zero for a symmetric filter. This is the quantity the symlets minimise, and
    the only thing that separates them from the Daubechies filter of the same
    order, since support and vanishing moments are identical.

    A sign change in the amplitude response would register as a jump of pi that
    no delay accounts for, so the number only means what it says while the
    response keeps one sign. Every filter in WAVELET_FAMILIES does across the
    band, and any two of the same order share the response exactly, which is
    what makes them comparable.

    The band edge is dropped rather than trusted. These filters roll off as
    cos(w/2) to the power of the order, so near w = pi the response is smaller
    than the rounding of the polynomial that produced it and its angle is
    noise. Frequencies below a relative floor are excluded, which is also what
    makes the measure read exactly zero on a symmetric filter instead of
    reporting the noise as nonlinearity.
    """
    taps = np.asarray(low_taps, dtype=np.float64)
    frequencies = np.linspace(0, np.pi, 4096)[:-1]
    response = np.polyval(taps[::-1], np.exp(-1j * frequencies))

    magnitude = np.abs(response)
    trusted = magnitude > 1e-6 * magnitude.max()

    bulk_delay = np.exp(1j * (len(taps) - 1) / 2 * frequencies[trusted])
    residual = np.unwrap(np.angle(response[trusted] * bulk_delay))

    return float(np.abs(residual - residual[0]).max())


def packet_band_order(num_levels: int) -> NDArray:
    """
    Permutation putting packet bands in ascending frequency.

    A packet tree does not come out sorted. Splitting a band gives a low half
    and a high half, but the high half arrives frequency-reversed, so its own
    children come out backwards -- at two levels the natural order is
    LL, LH, HL, HH while the frequency order is LL, LH, HH, HL. Reversals
    compose the way Gray code does, which is why band f sits at natural index
    f XOR (f >> 1).

    Returns
    -------
    order such that bands[:, order] runs low frequency to high
    """
    frequencies = np.arange(2 ** num_levels)
    return frequencies ^ (frequencies >> 1)


class LearnableWaveletTransform(Layer):
    """
    One level of analysis: (batch, sequence, hidden) to two half length bands.

    The LMWT paper's single scale form is a[i] = alpha * x[2i] + beta * x[2i+1]
    and d[i] = gamma * x[2i] + delta * x[2i+1]. Those four scalars are the two
    tap case of the two filters held here, and the four are learned
    independently there, so the filters are learned independently here too.
    Only the initialization is tied, through the quadrature mirror.

    One forward call caches one set of patches, so a caller that wants the same
    taps applied to several bands must stack them onto the batch axis and make
    a single call rather than looping.
    """

    def __init__(self, hidden_dim: int, family: str = "haar"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.family = family
        self.low_taps, self.high_taps = family_taps(family, hidden_dim)
        self.num_taps = self.low_taps.shape[0]
        self.declare_shapes(
            inputs=((hidden_dim,),), outputs=((hidden_dim,), (hidden_dim,))
        )

        self._cached_length = None
        self._cached_indices = None
        self.patches = None
        self.zero_gradients()

    def _indices(self, sequence_length: int) -> NDArray:
        if self._cached_length != sequence_length:
            self._cached_indices = dyadic_indices(sequence_length, self.num_taps)
            self._cached_length = sequence_length
        return self._cached_indices

    def forward(self, incoming_x: NDArray) -> tuple[NDArray, NDArray]:
        """
        Returns
        -------
        (approximation, detail), each (batch, sequence // 2, hidden)
        """
        assert incoming_x.ndim == 3, (
            f"expected (batch, sequence, hidden), got shape {incoming_x.shape}"
        )
        assert incoming_x.shape[-1] == self.hidden_dim, (
            f"built for hidden_dim {self.hidden_dim}, got {incoming_x.shape[-1]}"
        )
        sequence = incoming_x.shape[1]
        assert sequence % 2 == 0 and sequence >= self.num_taps, (
            f"sequence must be even and at least {self.num_taps} taps long, "
            f"got {sequence}"
        )

        self.in_shape = incoming_x.shape
        self.indices = self._indices(sequence)
        self.patches = incoming_x[:, self.indices, :]

        approximation = np.einsum("bpkh,kh->bph", self.patches, self.low_taps)
        detail = np.einsum("bpkh,kh->bph", self.patches, self.high_taps)

        return approximation, detail

    def backward(self, grad_approximation: NDArray, grad_detail: NDArray) -> NDArray:
        """
        Scatter-add is the adjoint of the gather in forward. A position read by
        several coefficients collects a term from each, which is why this
        accumulates instead of assigning.
        """
        self.gradient_low = np.einsum(
            "bpkh,bph->kh", self.patches, grad_approximation
        )
        self.gradient_high = np.einsum("bpkh,bph->kh", self.patches, grad_detail)

        grad_patches = (
            grad_approximation[:, :, None, :] * self.low_taps
            + grad_detail[:, :, None, :] * self.high_taps
        )

        gradient = np.zeros(self.in_shape, dtype=grad_patches.dtype)
        np.add.at(gradient, (slice(None), self.indices), grad_patches)

        return gradient

    def update_weights(self, gradient_low: NDArray, gradient_high: NDArray) -> None:
        self.low_taps -= gradient_low
        self.high_taps -= gradient_high

    def purge(self) -> None:
        self.patches = None
        self.in_shape = None
        self.gradient_low = None
        self.gradient_high = None

    def get_weights(self) -> NDArray:
        return np.concatenate([self.low_taps.ravel(), self.high_taps.ravel()])

    def get_gradients(self) -> dict[str, NDArray]:
        return {
            "gradient_low": self.gradient_low,
            "gradient_high": self.gradient_high,
        }

    def zero_gradients(self) -> None:
        self.gradient_low = np.zeros_like(self.low_taps)
        self.gradient_high = np.zeros_like(self.high_taps)

    @property
    def num_parameters(self) -> int:
        return self.low_taps.size + self.high_taps.size

    def __str__(self):
        return (
            f"learnable wavelet analysis, {self.family} init, "
            f"{self.num_taps} taps over {self.hidden_dim} channels"
        )

    def __repr__(self):
        return self.__str__()


class LearnableWaveletSynthesis(Layer):
    """
    One level of synthesis: two half length bands back to full length.

    Structurally the transpose of the analysis layer, and initialized to the
    same taps, so a fresh analysis-synthesis pair reconstructs its input
    exactly. The taps are free to drift from that during training, which is
    what makes the inverse learnable rather than merely inverse.
    """

    def __init__(self, hidden_dim: int, family: str = "haar"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.family = family
        self.low_taps, self.high_taps = family_taps(family, hidden_dim)
        self.num_taps = self.low_taps.shape[0]
        self.declare_shapes(
            inputs=((hidden_dim,), (hidden_dim,)), outputs=((hidden_dim,),)
        )

        self._cached_length = None
        self._cached_indices = None
        self.approximation = None
        self.detail = None
        self.zero_gradients()

    def _indices(self, sequence_length: int) -> NDArray:
        if self._cached_length != sequence_length:
            self._cached_indices = dyadic_indices(sequence_length, self.num_taps)
            self._cached_length = sequence_length
        return self._cached_indices

    def forward(self, approximation: NDArray, detail: NDArray) -> NDArray:
        assert approximation.shape == detail.shape, (
            f"bands must match, got {approximation.shape} and {detail.shape}"
        )
        assert approximation.shape[-1] == self.hidden_dim, (
            f"built for hidden_dim {self.hidden_dim}, "
            f"got {approximation.shape[-1]}"
        )

        sequence = 2 * approximation.shape[1]
        self.out_shape = (approximation.shape[0], sequence, self.hidden_dim)
        self.indices = self._indices(sequence)
        self.approximation = approximation
        self.detail = detail

        patches = (
            approximation[:, :, None, :] * self.low_taps
            + detail[:, :, None, :] * self.high_taps
        )

        output = np.zeros(self.out_shape, dtype=patches.dtype)
        np.add.at(output, (slice(None), self.indices), patches)

        return output

    def backward(self, incoming_grad: NDArray) -> tuple[NDArray, NDArray]:
        """
        Returns
        -------
        (grad_approximation, grad_detail), matching the two forward inputs
        """
        grad_patches = incoming_grad[:, self.indices, :]

        self.gradient_low = np.einsum(
            "bpkh,bph->kh", grad_patches, self.approximation
        )
        self.gradient_high = np.einsum("bpkh,bph->kh", grad_patches, self.detail)

        grad_approximation = np.einsum("bpkh,kh->bph", grad_patches, self.low_taps)
        grad_detail = np.einsum("bpkh,kh->bph", grad_patches, self.high_taps)

        return grad_approximation, grad_detail

    def update_weights(self, gradient_low: NDArray, gradient_high: NDArray) -> None:
        self.low_taps -= gradient_low
        self.high_taps -= gradient_high

    def purge(self) -> None:
        self.approximation = None
        self.detail = None
        self.out_shape = None
        self.gradient_low = None
        self.gradient_high = None

    def get_weights(self) -> NDArray:
        return np.concatenate([self.low_taps.ravel(), self.high_taps.ravel()])

    def get_gradients(self) -> dict[str, NDArray]:
        return {
            "gradient_low": self.gradient_low,
            "gradient_high": self.gradient_high,
        }

    def zero_gradients(self) -> None:
        self.gradient_low = np.zeros_like(self.low_taps)
        self.gradient_high = np.zeros_like(self.high_taps)

    @property
    def num_parameters(self) -> int:
        return self.low_taps.size + self.high_taps.size

    def __str__(self):
        return (
            f"learnable wavelet synthesis, {self.family} init, "
            f"{self.num_taps} taps over {self.hidden_dim} channels"
        )

    def __repr__(self):
        return self.__str__()
