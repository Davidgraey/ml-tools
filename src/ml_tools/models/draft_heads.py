"""
Output heads for the DFlash drafter.

The drafter's spectral path -- the Prefix-FFT cache, the shared block spectrum,
the gate, the convolution, the mask slots -- never asks what a token is.
Everything that does is bundled here, and it is a small bundle that always
varies together: how wide the fan-out is, what the loss means, how a block of
per-slot predictions is walked into one coherent path, and what counts as
agreeing with the target.

Splitting the agreement test from the prefix reduction is the reason this is a
head rather than a flag. Deciding whether one slot agrees is the only part of
verification that depends on the output type; taking the longest agreeing prefix
is not, so that stays on the drafter and is shared.
"""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np

from ml_tools.models.layers.layers import GLOBAL_DTYPE, FullyConnectedLayer


class DraftHead(ABC):
    """
    What the drafter needs from a head.

    width is the fan-out the drafter's unary projection must produce. The four
    abstract methods are exactly the places where being discrete matters: the
    loss, the teacher forced selector pass, the inference walk, and the per-slot
    agreement test.

    The selector projection lives here rather than on the drafter because every
    parameter that conditions one slot on the slot before it belongs to the head
    -- a head is free to spend those parameters however its output type wants.

    rng is threaded in so a head built by the drafter draws from the drafter's
    own stream, which keeps initialisation identical to the era when these
    parameters were declared inline.
    """

    width: int

    def __init__(self,
                 hidden_dim: int,
                 block_size: int,
                 selector_rank: int,
                 rng=None):
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.selector_rank = selector_rank
        self.RNG = rng if rng is not None else np.random.RandomState(42)
        self.fc_selector = FullyConnectedLayer(
            hidden_dim, selector_rank, "linear"
        )

    @abstractmethod
    def loss(self, prediction: NDArray, targets: NDArray) -> tuple[float, NDArray]:
        """mean cost over every slot of every block, and d cost / d prediction"""

    @abstractmethod
    def refine_forward(self,
                       prediction: NDArray,
                       hidden: NDArray,
                       predecessors: NDArray,
                       targets: NDArray) -> tuple[float, NDArray, NDArray]:
        """
        Teacher forced pass over the block, taking each slot's predecessor from
        the target sequence so the walk disappears and all slots train at once.

        Returns the cost and the two upstream gradients, which the caller adds
        to its own to train the heads jointly or discards to train them side by
        side.
        """

    @abstractmethod
    def propose(self,
                prediction: NDArray,
                hidden: NDArray,
                anchor: NDArray) -> NDArray:
        """the sequential walk, at inference: one pass drafted, then committed"""

    @abstractmethod
    def accepted(self, draft: NDArray, verified: NDArray) -> NDArray:
        """(batch, slot) boolean -- whether each drafted slot stands"""

    def get_gradients(self) -> dict:
        return {"fc_selector": self.fc_selector.get_gradients()}

    def update_weights(self, fc_selector: dict) -> None:
        self.fc_selector.update_weights(**fc_selector)

    def zero_gradients(self) -> None:
        self.fc_selector.zero_gradients()

    def purge(self) -> None:
        self.fc_selector.purge()

    @property
    def num_parameters(self) -> int:
        return self.fc_selector.num_parameters


class CategoricalHead(DraftHead):
    """
    Drafting over a vocabulary, which is DFlash2 as published and the only case
    where speculation is free: a drafted token either equals the verified one or
    it does not, so the accepted output is distributed exactly as the target's.

    The selector reranks the unary top-k into a coherent path. Each transition
    scores as

        unary(candidate)
          + dot(predecessor(previous) * projection(hidden), successor(candidate))

    a rank-r bilinear form conditioned on the token actually chosen at the slot
    before. The draft model still ran once: only this walk is sequential, over k
    candidates rather than a vocabulary.
    """

    def __init__(self,
                 hidden_dim: int,
                 block_size: int,
                 vocab_size: int,
                 selector_rank: int = 256,
                 selector_top_k: int = 16,
                 sample_from_anchor: bool = False,
                 rng=None):
        super().__init__(hidden_dim, block_size, selector_rank, rng)
        self.vocab_size = vocab_size
        self.width = vocab_size
        self.selector_top_k = min(selector_top_k, vocab_size)
        self.sample_from_anchor = sample_from_anchor

        scale = 1.0 / np.sqrt(hidden_dim)
        self.predecessor_codebook = (
            self.RNG.normal(scale=scale, size=(vocab_size, selector_rank))
        ).astype(GLOBAL_DTYPE)
        self.successor_codebook = (
            self.RNG.normal(scale=scale, size=(vocab_size, selector_rank))
        ).astype(GLOBAL_DTYPE)
        self._candidates = None
        self.zero_gradients()

    @staticmethod
    def loss(prediction: NDArray, targets: NDArray) -> tuple[float, NDArray]:
        """
        Mean cross entropy over every slot, taking integer labels so no one-hot
        over the vocabulary is ever materialised.

        At block_size 1 this is exactly a decoder's next token loss, which is
        the cheapest way to check the block machinery changed the shape of the
        objective and not the objective.
        """
        flat = prediction.reshape(-1, prediction.shape[-1])
        gold = targets.reshape(-1)
        rows = np.arange(gold.size)
        shifted = flat - flat.max(axis=-1, keepdims=True)
        log_partition = np.log(np.exp(shifted).sum(axis=-1))
        value = float((log_partition - shifted[rows, gold]).mean())

        gradient = np.exp(shifted - log_partition[:, None])
        gradient[rows, gold] -= 1.0
        return value, (gradient / gold.size).reshape(prediction.shape)

    def refine_forward(self, prediction, hidden, predecessors, targets):
        """
        Candidates are the unary top-k with the gold token forced in, replacing
        the weakest when it is missing -- without it a slot has no positive
        class and teaches nothing.
        """
        flat_logits = prediction.reshape(-1, self.block_size, self.vocab_size)
        flat_hidden = hidden.reshape(-1, self.block_size, self.hidden_dim)
        previous = predecessors.reshape(-1, self.block_size)
        gold = targets.reshape(-1, self.block_size)

        candidates = np.argpartition(
            flat_logits, -self.selector_top_k, axis=-1
        )[..., -self.selector_top_k:]
        weakest = np.argmin(
            np.take_along_axis(flat_logits, candidates, axis=-1), axis=-1
        )[..., None]
        found = (candidates == gold[..., None]).any(axis=-1)
        np.put_along_axis(
            candidates,
            weakest,
            np.where(
                found,
                np.take_along_axis(candidates, weakest, axis=-1)[..., 0],
                gold,
            )[..., None],
            axis=-1,
        )
        self._candidates = candidates

        unary = np.take_along_axis(flat_logits, candidates, axis=-1)
        projection = self.fc_selector(flat_hidden)
        query = self.predecessor_codebook[previous] * projection
        successor = self.successor_codebook[candidates]
        score = unary + np.einsum("nskr,nsr->nsk", successor, query)

        gold_slot = np.argmax(candidates == gold[..., None], axis=-1)[..., None]
        shifted = score - score.max(axis=-1, keepdims=True)
        log_partition = np.log(np.exp(shifted).sum(axis=-1))
        value = float((
            log_partition
            - np.take_along_axis(shifted, gold_slot, axis=-1)[..., 0]
        ).mean())

        dscore = np.exp(shifted - log_partition[..., None])
        np.put_along_axis(
            dscore,
            gold_slot,
            np.take_along_axis(dscore, gold_slot, axis=-1) - 1.0,
            axis=-1,
        )
        dscore /= previous.size

        dquery = np.einsum("nsk,nskr->nsr", dscore, successor)
        np.add.at(
            self.gradient_successor_codebook,
            candidates.reshape(-1),
            (dscore[..., None] * query[:, :, None, :]).reshape(
                -1, self.selector_rank
            ),
        )
        np.add.at(
            self.gradient_predecessor_codebook,
            previous.reshape(-1),
            (dquery * projection).reshape(-1, self.selector_rank),
        )
        dhidden = self.fc_selector.backward(
            dquery * self.predecessor_codebook[previous]
        )

        dprediction = np.zeros_like(flat_logits)
        np.put_along_axis(dprediction, candidates, dscore, axis=-1)
        return (
            value,
            dprediction.reshape(prediction.shape),
            dhidden.reshape(hidden.shape),
        )

    def propose(self, prediction, hidden, anchor):
        top_k = self.selector_top_k
        candidates = np.argpartition(prediction, -top_k, axis=-1)[..., -top_k:]
        unary = np.take_along_axis(prediction, candidates, axis=-1)
        projection = self.fc_selector(hidden)

        previous = np.asarray(anchor).reshape(-1)
        chosen = []
        for slot in range(prediction.shape[1]):
            query = self.predecessor_codebook[previous] * projection[:, slot]
            successor = self.successor_codebook[candidates[:, slot]]
            score = unary[:, slot] + np.einsum("bkr,br->bk", successor, query)
            pick = np.argmax(score, axis=-1)
            previous = np.take_along_axis(
                candidates[:, slot], pick[:, None], axis=-1
            )[:, 0]
            chosen.append(previous)

        path = np.stack(chosen, axis=1)
        return path if self.sample_from_anchor else path[:, 1:]

    def accepted(self, draft, verified):
        return np.asarray(draft) == np.asarray(verified)

    @property
    def candidates(self) -> NDArray | None:
        """the candidate set the last refine_forward scored, for inspection"""
        return self._candidates

    def get_gradients(self) -> dict:
        gradients = super().get_gradients()
        gradients.update({
            "gradient_predecessor_codebook": self.gradient_predecessor_codebook,
            "gradient_successor_codebook": self.gradient_successor_codebook,
        })
        return gradients

    def update_weights(self,
                       gradient_predecessor_codebook: NDArray,
                       gradient_successor_codebook: NDArray,
                       **inherited) -> None:
        super().update_weights(**inherited)
        self.predecessor_codebook -= gradient_predecessor_codebook
        self.successor_codebook -= gradient_successor_codebook

    def zero_gradients(self) -> None:
        super().zero_gradients()
        self.gradient_predecessor_codebook = np.zeros_like(
            self.predecessor_codebook
        )
        self.gradient_successor_codebook = np.zeros_like(self.successor_codebook)

    def purge(self) -> None:
        super().purge()
        self._candidates = None

    @property
    def num_parameters(self) -> int:
        return (
            super().num_parameters
            + self.predecessor_codebook.size
            + self.successor_codebook.size
        )

    def __str__(self):
        return (
            f"categorical head, vocab {self.vocab_size}, top-k "
            f"{self.selector_top_k} at rank {self.selector_rank}"
        )


class ContinuousHead(DraftHead):
    """
    Real valued drafting, a mean and a log variance per channel.

    Three departures from the categorical head, in rising order of how much they
    matter.

    The codebook lookups become linear maps. A codebook is an embedding indexed
    by a token, and a real valued predecessor has nothing to index, so
    predecessor and successor become fc_predecessor, output_dim -> rank, and
    fc_correction, rank -> output_dim. The rank r gated form survives intact; it
    simply emits one vector rather than a scalar per candidate, and costs
    2 * output_dim * rank instead of 2 * vocab * rank.

    Reranking becomes refinement. With no finite candidate set to reorder, the
    walk instead adds a correction to each slot conditioned on the value
    committed at the slot before -- the same information the categorical
    selector conditions on, applied additively.

    And agreement becomes a tolerance, which is a change in kind rather than in
    dtype. Discrete speculation is exact. Real values are never equal, so
    acceptance here is a bounded error contract: atol and rtol are part of the
    output specification, not tuning knobs, and turning them up buys throughput
    by giving up fidelity. Note that the tolerance deliberately ignores this
    head's own predicted variance -- a drafter graded on acceptance would
    otherwise learn to predict a larger one.
    """

    def __init__(self,
                 hidden_dim: int,
                 block_size: int,
                 output_dim: int,
                 selector_rank: int = 256,
                 absolute_tolerance: float = 1e-2,
                 relative_tolerance: float = 1e-2,
                 logvar_floor: float = -8.0,
                 sample_from_anchor: bool = False,
                 rng=None):
        super().__init__(hidden_dim, block_size, selector_rank, rng)
        self.output_dim = output_dim
        self.width = 2 * output_dim
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.logvar_floor = logvar_floor
        self.sample_from_anchor = sample_from_anchor

        self.fc_predecessor = FullyConnectedLayer(
            output_dim, selector_rank, "linear"
        )
        self.fc_correction = FullyConnectedLayer(
            selector_rank, output_dim, "linear"
        )
        self.zero_gradients()

    def _split(self, prediction: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """
        mean, floored log variance, and the mask of entries the floor left free.

        The floor matters at initialisation: exp(-logvar) on a random log
        variance makes the likelihood enormous and the first steps wild.
        """
        mean = prediction[..., :self.output_dim]
        raw = prediction[..., self.output_dim:]
        return mean, np.maximum(raw, self.logvar_floor), raw > self.logvar_floor

    def mean_and_sigma(self, prediction: NDArray) -> tuple[NDArray, NDArray]:
        """
        The draft as a caller wants to read it: the predicted value per slot and
        the head's own standard deviation on it.

        Useful for deciding where to spend the target model, and worth watching
        during training -- sigma tracks the model's own residual, so a value far
        above the data's noise floor says the mean is still undertrained rather
        than that the variance is miscalibrated.
        """
        mean, logvar, _ = self._split(prediction)
        return mean, np.exp(0.5 * logvar)

    def loss(self, prediction: NDArray, targets: NDArray) -> tuple[float, NDArray]:
        """
        Mean Gaussian negative log likelihood over every slot and channel.

        At block_size 1 this reduces to a plain one step heteroscedastic
        regression loss, the same reduction the categorical head has to a
        decoder's cross entropy.
        """
        mean, logvar, active = self._split(prediction)
        inverse = np.exp(-logvar)
        residual = targets - mean
        count = targets.size
        value = float(
            0.5 * (logvar + residual ** 2 * inverse + np.log(2 * np.pi)).mean()
        )

        dmean = -residual * inverse / count
        dlogvar = 0.5 * (1.0 - residual ** 2 * inverse) / count * active
        return value, np.concatenate([dmean, dlogvar], axis=-1)

    def refine_forward(self, prediction, hidden, predecessors, targets):
        """
        The correction is scored against the same inverse variance the head
        predicted, so a channel the head is unsure about pulls proportionally
        less. The variance itself takes no gradient from this term -- it is
        trained by loss(), and letting the refinement widen it would only
        slacken its own objective.
        """
        mean, logvar, _ = self._split(prediction)
        shape = (-1, self.block_size, self.output_dim)
        flat_mean = mean.reshape(shape)
        gold = targets.reshape(shape)
        inverse = np.exp(-logvar).reshape(shape)

        predecessor = self.fc_predecessor(predecessors.reshape(shape))
        projection = self.fc_selector(
            hidden.reshape(-1, self.block_size, self.hidden_dim)
        )
        gated = predecessor * projection
        refined = flat_mean + self.fc_correction(gated)

        residual = gold - refined
        count = gold.size
        value = float((0.5 * residual ** 2 * inverse).mean())
        drefined = -residual * inverse / count

        dgated = self.fc_correction.backward(drefined)
        dhidden = self.fc_selector.backward(dgated * predecessor)
        self.fc_predecessor.backward(dgated * projection)

        dprediction = np.concatenate(
            [drefined.reshape(mean.shape), np.zeros_like(logvar)], axis=-1
        )
        return value, dprediction, dhidden.reshape(hidden.shape)

    def propose(self, prediction, hidden, anchor):
        mean, _, _ = self._split(prediction)
        projection = self.fc_selector(hidden)

        previous = np.asarray(anchor, dtype=np.float64).reshape(
            -1, self.output_dim
        )
        chosen = []
        for slot in range(mean.shape[1]):
            gated = self.fc_predecessor(previous) * projection[:, slot]
            previous = mean[:, slot] + self.fc_correction(gated)
            chosen.append(previous)

        path = np.stack(chosen, axis=1)
        return path if self.sample_from_anchor else path[:, 1:]

    def accepted(self, draft, verified):
        draft = np.asarray(draft)
        verified = np.asarray(verified)
        tolerance = (
            self.absolute_tolerance
            + self.relative_tolerance * np.abs(verified)
        )
        return (np.abs(draft - verified) <= tolerance).all(axis=-1)

    def get_gradients(self) -> dict:
        gradients = super().get_gradients()
        gradients.update({
            "fc_predecessor": self.fc_predecessor.get_gradients(),
            "fc_correction": self.fc_correction.get_gradients(),
        })
        return gradients

    def update_weights(self,
                       fc_predecessor: dict,
                       fc_correction: dict,
                       **inherited) -> None:
        super().update_weights(**inherited)
        self.fc_predecessor.update_weights(**fc_predecessor)
        self.fc_correction.update_weights(**fc_correction)

    def zero_gradients(self) -> None:
        super().zero_gradients()
        self.fc_predecessor.zero_gradients()
        self.fc_correction.zero_gradients()

    def purge(self) -> None:
        super().purge()
        self.fc_predecessor.purge()
        self.fc_correction.purge()

    @property
    def num_parameters(self) -> int:
        return (
            super().num_parameters
            + self.fc_predecessor.num_parameters
            + self.fc_correction.num_parameters
        )

    def __str__(self):
        return (
            f"continuous head, {self.output_dim} channels at rank "
            f"{self.selector_rank}, tolerance "
            f"{self.absolute_tolerance} + {self.relative_tolerance} relative"
        )
