"""
trainable hierarchical wavelet feature encoding for one dimensional signals.

wavelet packet decomposition, with the filter taps learned rather than fixed.
The cascade in wavelet_layers.py splits only the approximation, so its bands
come out logarithmically spaced -- fine detail at the top, everything coarse
crammed into one band at the bottom. This splits both halves at every level, so
L levels give 2**L bands of equal width covering the spectrum uniformly. That
costs 2**L bands of memory instead of L+1, and buys resolution where a cascade
has none: a chirp sweeping through the upper half of the band is one blurred
detail coefficient to a cascade and a clear diagonal across packet bands.

Two things come out. The bands themselves, in ascending frequency order, for
anything downstream that wants the coefficients. And a fixed length feature
vector of two statistics per band per channel, pooled over time:

    log energy      how much power this band carries
    temporal entropy  how spread out in time that power is

The pair is chosen to separate waveform shape from the accidents of a
particular recording. Both are invariant to phase and to where in the window an
event lands, and log energy shifts by a constant under a change of amplitude,
so a downstream layer can subtract it away. Energy alone says a tone and a
click at the same scale are the same thing; entropy is what tells them apart,
being near one for power spread evenly across the band's timeline and near zero
for power concentrated in a few coefficients.

The learned object is the filter bank: one low pass and one high pass per
level, shared across every band at that level. Printing them says what the
model decided "coarse" and "fine" should mean for this data.
"""

import numpy as np
from numpy.typing import NDArray
from ml_tools.models.layers.layers import Layer
from ml_tools.models.layers.wavelet import (
    LearnableWaveletTransform,
    packet_band_order,
)


STATISTICS = ("log_energy", "temporal_entropy")


class WaveletPacketEncoder(Layer):
    """
    Signals to band coefficients and pooled per-band statistics.

    Accepts (batch, length) for a single channel or (batch, length, channels),
    and returns the shape it was given on the backward pass.

    One transform layer per level, applied once per level to every band at
    once by stacking bands onto the batch axis. That keeps the taps shared
    within a level -- which is what a packet transform means -- and keeps each
    layer's forward cache valid for the single backward call it will get.
    """

    def __init__(
        self,
        num_levels: int = 3,
        num_channels: int = 1,
        family: str = "db2",
        eps: float = 1e-8,
    ):
        """
        Parameters
        ----------
        num_levels : depth of the tree. Bands double and band length halves
            with each level, so this trades frequency resolution against time
            resolution and nothing else -- the total coefficient count is
            fixed.
        num_channels : channels per sample, 1 for a plain waveform
        family : which orthogonal filter initializes the taps, any key of
            WAVELET_FAMILIES. db2 by default rather than haar: haar's two taps
            leak badly between packet bands, which shows up directly as
            crosstalk in the statistics.
        eps : floor added to squared coefficients before the statistics, so an
            all-zero band gives a finite entropy instead of a nan
        """
        super().__init__()
        assert num_levels >= 1, f"num_levels must be at least 1, got {num_levels}"
        assert num_channels >= 1, f"num_channels must be at least 1, got {num_channels}"

        self.num_levels = num_levels
        self.num_channels = num_channels
        self.family = family
        self.eps = eps

        self.levels = [
            LearnableWaveletTransform(num_channels, family) for _ in range(num_levels)
        ]
        self.num_taps = self.levels[0].num_taps
        self.band_order = packet_band_order(num_levels)

        self.declare_shapes(
            inputs=((num_channels,),),
            outputs=((self.num_features,), (num_channels,)),
        )

        self.bands = None
        self.features = None
        self.squeezed = False

    # -------------    what the caller needs to read the output    -----
    @property
    def num_bands(self) -> int:
        return 2 ** self.num_levels

    @property
    def num_features(self) -> int:
        return self.num_bands * self.num_channels * len(STATISTICS)

    @property
    def band_centers(self) -> NDArray:
        """
        Centre frequency of each band, as a fraction of the Nyquist rate.

        Band f covers [f, f+1] / num_bands after the reordering, so multiply
        by sample_rate / 2 to read these in Hz.
        """
        return (np.arange(self.num_bands) + 0.5) / self.num_bands

    @property
    def feature_names(self) -> tuple[str, ...]:
        """
        Labels in the order the feature vector holds them.

        The vector is (band, channel, statistic) flattened, band slowest. Worth
        having: a trained dense head's first row of weights is only readable
        against these.
        """
        return tuple(
            f"band{band}_ch{channel}_{statistic}"
            for band in range(self.num_bands)
            for channel in range(self.num_channels)
            for statistic in STATISTICS
        )

    # -------------    the tree    -------------------------------------
    def _as_three_axes(self, x_data: NDArray) -> NDArray:
        """(batch, length) is promoted to a single channel, and remembered"""
        self.squeezed = x_data.ndim == 2
        if self.squeezed:
            return x_data[:, :, None]

        assert x_data.ndim == 3, (
            f"expected (batch, length) or (batch, length, channels), got "
            f"shape {x_data.shape}"
        )
        return x_data

    def _check_length(self, length: int) -> None:
        stride = 2 ** self.num_levels
        assert length % stride == 0, (
            f"{self.num_levels} levels halve the signal {self.num_levels} "
            f"times, so its length must divide by {stride}, got {length}"
        )

        # the deepest split is the tight one: a filter reading more positions
        # than the band holds wraps over the same samples twice, which loses
        # orthogonality quietly rather than loudly
        deepest = length // 2 ** (self.num_levels - 1)
        assert deepest >= self.num_taps, (
            f"{self.family} reads {self.num_taps} positions, but level "
            f"{self.num_levels - 1} sees only {deepest} of them. Either "
            f"shorten the tree or pass at least "
            f"{self.num_taps * 2 ** (self.num_levels - 1)} samples."
        )

        # entropy is normalised by log(band length), so a band of one
        # coefficient would divide by zero, and a band of two carries almost
        # no temporal information worth reading
        assert length // stride >= 2, (
            f"{self.num_levels} levels leave {length // stride} coefficients "
            "per band, too few for a temporal entropy. Shorten the tree."
        )

    def forward(self, x_data: NDArray) -> tuple[NDArray, NDArray]:
        """
        Returns
        -------
        (features, bands)
            features (batch, num_features), bands
            (batch, num_bands, length // num_bands, channels) low to high
        """
        x_data = self._as_three_axes(x_data)
        assert x_data.shape[-1] == self.num_channels, (
            f"built for {self.num_channels} channels, got {x_data.shape[-1]}"
        )
        self._check_length(x_data.shape[1])

        self.in_shape = x_data.shape
        batch, _, channels = self.in_shape

        current = x_data[:, None, :, :]
        for level in self.levels:
            bands, band_length = current.shape[1], current.shape[2]
            low, high = level(current.reshape(batch * bands, band_length, channels))
            current = np.stack([low, high], axis=1).reshape(
                batch, bands * 2, band_length // 2, channels
            )

        self.bands = current[:, self.band_order]
        self.features = self._statistics(self.bands)

        return self.features, self.bands

    def _statistics(self, bands: NDArray) -> NDArray:
        """
        Pool each band over time into log energy and temporal entropy.

        The eps floor goes inside the squared coefficients rather than onto the
        total, so the same stabilised quantity feeds both statistics and both
        derivatives. Without it a silent band gives log(0) in the entropy.
        """
        stabilized = bands ** 2 + self.eps
        total = stabilized.sum(axis=2, keepdims=True)
        band_length = bands.shape[2]

        self.total = total
        self.proportion = stabilized / total
        self.entropy = -(self.proportion * np.log(self.proportion)).sum(axis=2) / np.log(
            band_length
        )

        log_energy = np.log(total[:, :, 0, :] / band_length)

        return np.stack([log_energy, self.entropy], axis=-1).reshape(
            bands.shape[0], self.num_features
        )

    def _statistics_backward(self, grad_features: NDArray) -> NDArray:
        """
        Gradient of both statistics with respect to the band coefficients.

        Both run through u = c**2 + eps, so both end in a factor of 2c and the
        chain rule is taken in u. With p = u / S and S the band total,
        d(log S/N)/du = 1/S, and d(entropy)/du = (f - log p) / (S log N) where
        f = sum p log p, which is -entropy * log N. That substitution is the
        only reason this does not need the un-normalised sum kept around.
        """
        batch, num_bands, band_length, channels = self.bands.shape
        grad = grad_features.reshape(batch, num_bands, channels, len(STATISTICS))

        grad_log_energy = grad[..., 0][:, :, None, :]
        grad_entropy = grad[..., 1][:, :, None, :]

        log_length = np.log(band_length)
        from_energy = grad_log_energy / self.total
        from_entropy = (
            grad_entropy
            * (-self.entropy[:, :, None, :] * log_length - np.log(self.proportion))
            / (self.total * log_length)
        )

        return 2 * self.bands * (from_energy + from_entropy)

    def backward(
        self,
        grad_features: NDArray = None,
        grad_bands: NDArray = None,
    ) -> NDArray:
        """
        Gradient with respect to the input signal.

        Both outputs are optional, since a classifier head reads only the
        features and a downstream sequence model reads only the bands. Passing
        both sums the two paths, which is correct: they are two readings of the
        same coefficients, not two separate tensors.
        """
        assert (grad_features is not None) or (grad_bands is not None), (
            "backward needs a gradient for at least one of the two outputs"
        )
        assert self.bands is not None, "call forward before backward"

        grad = (
            np.zeros_like(self.bands)
            if grad_features is None
            else self._statistics_backward(grad_features)
        )
        if grad_bands is not None:
            grad = grad + grad_bands

        # undo the frequency reordering before unwinding the tree, since the
        # tree was built in natural order
        restored = np.empty_like(grad)
        restored[:, self.band_order] = grad

        batch, _, channels = self.in_shape
        current = restored
        for level in reversed(self.levels):
            bands, band_length = current.shape[1], current.shape[2]
            pair = current.reshape(batch * (bands // 2), 2, band_length, channels)
            current = level.backward(pair[:, 0], pair[:, 1]).reshape(
                batch, bands // 2, band_length * 2, channels
            )

        gradient = current[:, 0]

        return gradient[:, :, 0] if self.squeezed else gradient

    # -------------    Layer plumbing    ------------------------------
    @property
    def _parts(self) -> dict[str, Layer]:
        return {f"level_{index}": level for index, level in enumerate(self.levels)}

    def update_weights(self, **gradients: dict[str, NDArray]) -> None:
        parts = self._parts
        for name, part_gradients in gradients.items():
            parts[name].update_weights(**part_gradients)

    def purge(self) -> None:
        self.bands = None
        self.features = None
        self.total = None
        self.proportion = None
        self.entropy = None
        for part in self._parts.values():
            part.purge()

    def get_weights(self) -> dict[str, NDArray]:
        return {name: part.get_weights() for name, part in self._parts.items()}

    def get_gradients(self) -> dict[str, dict[str, NDArray]]:
        return {name: part.get_gradients() for name, part in self._parts.items()}

    def zero_gradients(self) -> None:
        for part in self._parts.values():
            part.zero_gradients()

    @property
    def num_parameters(self) -> int:
        return sum(part.num_parameters for part in self._parts.values())

    def __str__(self):
        return (
            f"wavelet packet encoder, {self.num_levels} levels, "
            f"{self.num_bands} bands, {self.num_channels} channels, "
            f"{self.family} init, {self.num_features} features"
        )

    def __repr__(self):
        return self.__str__()
