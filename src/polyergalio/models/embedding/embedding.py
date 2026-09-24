"""
Token embeddings: a trainable lookup table from token id to vector.
"""
from typing import Optional

import numpy as np
from polyergalio.models.constants import ANY_SHAPE, GLOBAL_DTYPE
from polyergalio.models.layers.basal_layers import Layer, xavier
from numpy.typing import NDArray


class TextEmbedding(Layer):
    """
    trainable vector lookup table, like torch.nn.Embedding.

    Row i of `weights` is the vector for token id i.
    Forward indexes the table; backward scatters the incoming gradient back onto the rows that were read,
    summing over repeated tokens.

    Input shape: (...) integer token ids
    Output shape: (..., embedding_dim)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None, special_tokens: Optional[dict] = None):
        """
        Parameters
        ----------
        num_embeddings : vocabulary size, the number of rows in the table
        embedding_dim : width of each token's vector
        padding_idx : optional id whose vector is fixed at zero and never trained
        """
        super().__init__()
        if padding_idx is not None and not 0 <= padding_idx < num_embeddings:
            raise ValueError(f"padding_idx must fall in [0, {num_embeddings}), got {padding_idx}")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.special_tokens = special_tokens

        self.declare_shapes(inputs=(ANY_SHAPE,), outputs=((embedding_dim,),))

        self.weights = xavier(self.RNG, ni=num_embeddings, no=embedding_dim).astype(GLOBAL_DTYPE)

        if padding_idx is not None:
            self.weights[padding_idx] = 0.0

        self.token_ids = None
        self.output = None
        self.zero_gradients()

    def infer_output_shapes(self, input_shapes: tuple[tuple, ...]) -> tuple[tuple, ...]:
        return ((*input_shapes[0], self.embedding_dim),)

    def forward(self, token_ids: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        token_ids : (...) integer ids in [0, num_embeddings)
        mask : unused -- padded positions are handled by padding_idx; accepted
            for pass-through compatibility with the graph

        Returns
        -------
        (..., embedding_dim) the looked-up vectors
        """
        token_ids = np.asarray(token_ids)
        if not np.issubdtype(token_ids.dtype, np.integer):
            if not np.array_equal(token_ids, np.round(token_ids)):
                raise TypeError(f"token ids must be integers, got dtype {token_ids.dtype}")
            token_ids = token_ids.astype(int)

        if token_ids.size and (token_ids.min() < 0 or token_ids.max() >= self.num_embeddings):
            raise IndexError(
                f"token ids must fall in [0, {self.num_embeddings}), "
                f"got range [{token_ids.min()}, {token_ids.max()}]"
            )
        self.token_ids = token_ids
        self.output = self.weights[token_ids]
        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        """
        Returns
        -------
        zeros shaped like the token ids -- discrete ids carry no gradient
        """
        self.gradient_weights = np.zeros_like(self.weights)
        np.add.at(
            self.gradient_weights,
            self.token_ids.reshape(-1),
            incoming_grad.reshape(-1, self.embedding_dim),
        )
        if self.padding_idx is not None:
            self.gradient_weights[self.padding_idx] = 0.0
        return np.zeros(self.token_ids.shape, dtype=GLOBAL_DTYPE)

    def update_weights(self, gradient_weights: NDArray) -> None:
        self.weights -= gradient_weights

    def get_weights(self, for_serialize: bool = False):
        if for_serialize:
            return {"weights": self.weights}
        return self.weights

    def set_weights(self, weights: dict) -> None:
        if weights is not None:
            self.weights = np.array(weights["weights"], dtype=GLOBAL_DTYPE)

    def get_gradients(self) -> dict[str, NDArray]:
        return {"gradient_weights": self.gradient_weights}

    def zero_gradients(self) -> None:
        self.gradient_weights = np.zeros_like(self.weights)

    def purge(self) -> None:
        self.token_ids = None
        self.output = None

    @property
    def num_parameters(self) -> int:
        return self.weights.size

    def __str__(self):
        padding = "" if self.padding_idx is None else f", padding_idx {self.padding_idx}"
        return f"TextEmbedding, {self.num_embeddings} tokens x {self.embedding_dim}{padding}"

    def __repr__(self):
        return self.__str__()
