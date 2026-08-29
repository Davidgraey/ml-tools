"""
Learnable Multi-Scale Wavelet Transformer, https://arxiv.org/abs/2504.08801

The paper replaces dot-product self-attention with a learnable Haar cascade.
Mixing across positions is done by the cascade itself rather than by a score
matrix, so the cost is linear in sequence length instead of quadratic.

The filter families and the single level transform live in wavelet.py, which
signal_encoder.py also builds on. Haar is the two tap case of the Daubechies
family, so nothing here is specific to it: family="db2" runs a four tap
decomposition with no other change, and family="sym4" runs an eight tap one
whose phase is nearly linear, which keeps a feature at the position it
occurred.
"""

import numpy as np
from numpy.typing import NDArray
from ml_tools.models.layers.layers import (
    Layer,
    FullyConnectedLayer,
    NormalizeLayer,
)
from ml_tools.models.layers.wavelet import (
    LearnableWaveletSynthesis,
    LearnableWaveletTransform,
)


class LMWTAttention(Layer):
    """
    LMWT mixing block, a drop-in replacement for a self-attention block.

    Analysis cascades num_levels times, each level halving the sequence and
    setting aside a detail band. Every band gets its own channel mix, so the
    model can weight coarse structure against fine structure per scale, and
    synthesis cascades back to full length. Residual and norm wiring follows
    FourierAttention, so the two are interchangeable in a stack.

    Only the approximation is split, which is what makes this a cascade rather
    than a packet tree: bands come out logarithmically spaced, fine at the top
    and coarse at the bottom. signal_encoder.py splits both halves instead,
    when uniform frequency resolution matters more than cheapness.

    The learned taps are readable on their own: printing a level's low pass
    filter says what "coarse" came to mean for this task.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_levels: int = 2,
        family: str = "haar",
    ):
        """
        Parameters
        ----------
        hidden_dim : channel width, unchanged from input to output
        num_levels : depth of the cascade. The sequence must divide by
            2 ** num_levels, since each level halves it.
        family : which orthogonal filter initializes the taps, any key of
            WAVELET_FAMILIES. haar is the paper's choice, db2 and up trade
            more taps for smoother bands, and sym4 and up spend the same taps
            on near-linear phase instead of minimum phase. Prefer a symlet
            when where something happens in the sequence matters as much as
            that it happened.
        """
        super().__init__()
        assert num_levels >= 1, f"num_levels must be at least 1, got {num_levels}"

        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.family = family

        self.analysis = [
            LearnableWaveletTransform(hidden_dim, family) for _ in range(num_levels)
        ]
        self.synthesis = [
            LearnableWaveletSynthesis(hidden_dim, family) for _ in range(num_levels)
        ]
        self.detail_mix = [
            FullyConnectedLayer(ni=hidden_dim, no=hidden_dim, activation_type="linear")
            for _ in range(num_levels)
        ]
        self.approx_mix = FullyConnectedLayer(
            ni=hidden_dim, no=hidden_dim, activation_type="linear"
        )

        self.norm_a = NormalizeLayer(ni=hidden_dim, shift_scale=False)
        self.fc = FullyConnectedLayer(
            ni=hidden_dim, no=hidden_dim, activation_type="relu"
        )
        self.norm_b = NormalizeLayer(ni=hidden_dim, shift_scale=True)

        self.declare_shapes(inputs=((hidden_dim,),), outputs=((hidden_dim,),))

    @property
    def _parts(self) -> dict[str, Layer]:
        """
        Sub-layers by name, the keys the optimizer round-trips.

        get_gradients hands back one nested dict per name and update_weights is
        called with those same names as keywords, so this single mapping keeps
        the two from drifting apart as levels are added.
        """
        parts = {
            "approx_mix": self.approx_mix,
            "norm_a": self.norm_a,
            "fc": self.fc,
            "norm_b": self.norm_b,
        }
        for index in range(self.num_levels):
            parts[f"analysis_{index}"] = self.analysis[index]
            parts[f"synthesis_{index}"] = self.synthesis[index]
            parts[f"detail_mix_{index}"] = self.detail_mix[index]
        return parts

    def _cascade(self, x_data: NDArray) -> NDArray:
        """analysis down, mix each band, synthesis back up"""
        current = x_data
        details = []
        for level in self.analysis:
            current, detail = level(current)
            details.append(detail)

        current = self.approx_mix(current)
        details = [mix(band) for mix, band in zip(self.detail_mix, details)]

        for level, band in zip(reversed(self.synthesis), reversed(details)):
            current = level(current, band)

        return current

    def _cascade_backward(self, incoming_grad: NDArray) -> NDArray:
        """
        Reverse of _cascade. Synthesis ran deepest first, so it unwinds
        shallowest first, and each step yields the gradient of the band it
        consumed.
        """
        grad = incoming_grad
        grad_details = []
        for level in self.synthesis:
            grad, grad_detail = level.backward(grad)
            grad_details.append(grad_detail)

        grad = self.approx_mix.backward(grad)
        grad_details = [
            mix.backward(band) for mix, band in zip(self.detail_mix, grad_details)
        ]

        for level, band in zip(reversed(self.analysis), reversed(grad_details)):
            grad = level.backward(grad, band)

        return grad

    def forward(self, x_data: NDArray) -> NDArray:
        assert x_data.ndim == 3, (
            f"expected (batch, sequence, hidden), got shape {x_data.shape}"
        )
        assert x_data.shape[-1] == self.hidden_dim, (
            f"built for hidden_dim {self.hidden_dim}, got {x_data.shape[-1]}"
        )

        stride = 2 ** self.num_levels
        assert x_data.shape[1] % stride == 0, (
            f"{self.num_levels} levels halve the sequence {self.num_levels} "
            f"times, so it must divide by {stride}, got {x_data.shape[1]}"
        )

        # the deepest level is the tight one, and it fails for a reason worth
        # naming: a filter reading more positions than the band holds wraps
        # over the same samples twice, which breaks orthogonality quietly
        # rather than loudly
        num_taps = self.analysis[0].num_taps
        deepest = x_data.shape[1] // 2 ** (self.num_levels - 1)
        assert deepest >= num_taps, (
            f"{self.family} reads {num_taps} positions, but level "
            f"{self.num_levels - 1} sees only {deepest} of them. Either "
            f"shorten the cascade or pass at least "
            f"{num_taps * 2 ** (self.num_levels - 1)} positions."
        )

        self.input = x_data
        mixed = self.norm_a(self._cascade(x_data) + x_data)
        self.output = self.norm_b(self.fc(mixed) + mixed)

        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        grad = self.norm_b.backward(incoming_gradient)
        grad = self.fc.backward(grad) + grad

        grad = self.norm_a.backward(grad)
        grad = self._cascade_backward(grad) + grad

        self.gradient = grad
        return grad

    def update_weights(self, **gradients: dict[str, NDArray]) -> None:
        parts = self._parts
        for name, part_gradients in gradients.items():
            parts[name].update_weights(**part_gradients)

    def purge(self) -> None:
        self.input = None
        self.output = None
        self.gradient = None
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
            f"LMWT mixer, hidden {self.hidden_dim}, {self.num_levels} levels, "
            f"{self.family} init"
        )

    def __repr__(self):
        return self.__str__()
