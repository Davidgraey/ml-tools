import numpy as np
from typing import Optional
from numpy.typing import NDArray
from ml_tools.models.layers.basal_layers import Layer


class RopeEmbedding(Layer):
    """
    Rotary positional embedding: a fixed, parameterless rotation of adjacent
    channel pairs by an angle proportional to the token's position.
    """

    def __init__(self,
                 sequence_length: int,
                 embedding_dimension: int,
                 base_freq: int = 10000):
        """
        Parameters
        ----------
        sequence_length : the ceiling this rotation table is built for. A
            shorter sequence is fine, it takes the leading rows
        embedding_dimension : channel width of the incoming embedding
        base_freq : wavelength base, 10,000 in the paper
        """
        super().__init__()
        assert embedding_dimension % 2 == 0, (
            f"embedding_dimension must be even, got {embedding_dimension}"
        )

        self.sequence_length: int = sequence_length
        self.embedding_dimension: int = embedding_dimension
        self.base_freq: int = base_freq
        # rotation is elementwise over position, so the sequence axis is free
        # up to the ceiling the rotation matrix was built for
        self.declare_shapes(
            inputs=((embedding_dimension,),),
            outputs=((embedding_dimension,),),
        )

        self.build_rope_array(sequence_length, embedding_dimension, base_freq)

    def build_rope_array(self,
                         sequence_length: int,
                         embedding_dimension: int,
                         base_freq: int = 10000
    ) -> None:
        """
        Precompute the sine and cosine tables, shaped (sequence, dimension / 2).

        Cached on the instance rather than returned, since the table is fixed
        for the life of the layer and every forward pass reads the same rows.

        Parameters
        ----------
        sequence_length : rows to build, the longest sequence to be accepted
        embedding_dimension : channel width, giving dimension / 2 rotation pairs
        base_freq : wavelength base, 10,000 in the paper

        Notes
        -----
        arange(0, d, 2)
        """
        inv_freq = 1.0 / (
            base_freq ** (np.arange(0, embedding_dimension, 2) / embedding_dimension)
        )
        position_int = np.arange(sequence_length, dtype=float)
        rotation_angle = np.einsum("i,j->ij", position_int, inv_freq)

        self.pos_sine = np.sin(rotation_angle)
        self.pos_cosine = np.cos(rotation_angle)

    def forward(self, input_data: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        input_data : (sequence, dimension) or (batch, sequence, dimension)
        mask : unused -- each position is rotated by its own index alone,
            independent of every other position. Accepted for pass-through
            compatibility with the graph.

        Returns
        -------
        the input with each channel pair rotated by its position's angle. Not
        a sum -- nothing is added to the embedding, it is rotated in place.
        """
        assert input_data.ndim >= 2, (
            f"expected at least (sequence, dimension), got {input_data.shape}"
        )
        sequence, dimension = input_data.shape[-2:]
        assert dimension == self.embedding_dimension, (
            f"built for embedding_dimension {self.embedding_dimension}, got "
            f"{dimension}"
        )
        assert sequence <= self.sequence_length, (
            f"rotation table holds {self.sequence_length} positions, got a "
            f"sequence of {sequence}"
        )

        # the table is (sequence, dimension / 2), which broadcasts against a
        # leading batch axis, so one path serves both ranks
        _sin = self.pos_sine[:sequence]
        _cos = self.pos_cosine[:sequence]

        even_channels = input_data[..., 0::2]
        odd_channels = input_data[..., 1::2]

        self.output = np.zeros_like(input_data)
        self.output[..., 0::2] = even_channels * _cos - odd_channels * _sin
        self.output[..., 1::2] = even_channels * _sin + odd_channels * _cos

        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        """
        A rotation's Jacobian is the rotation, so the vector-Jacobian product
        is the transpose, which for a rotation is the inverse: the same angles
        applied in the opposite direction.
        """
        sequence = incoming_gradient.shape[-2]
        _sin = self.pos_sine[:sequence]
        _cos = self.pos_cosine[:sequence]

        even_position = incoming_gradient[..., 0::2]
        odd_position = incoming_gradient[..., 1::2]

        self.gradient = np.zeros_like(incoming_gradient)
        self.gradient[..., 0::2] = even_position * _cos + odd_position * _sin
        self.gradient[..., 1::2] = -even_position * _sin + odd_position * _cos

        return self.gradient

    @property
    def rope_array(self) -> tuple[NDArray, NDArray]:
        """the (sine, cosine) tables, copied so callers cannot corrupt them"""
        return (self.pos_sine.copy(), self.pos_cosine.copy())

    def purge(self) -> None:
        """
        Clear the pass state. The rotation tables survive: they are fixed
        constants, not activations, and rebuilding them per pass would be waste.
        """
        self.output = None
        self.gradient = None

    def update_weights(self, **kwargs) -> None:
        pass

    def zero_gradients(self) -> None:
        pass

    def get_weights(self, for_serialize: bool = False):
        return {} if for_serialize else None

    def set_weights(self, weights: dict) -> None:
        pass

    def get_gradients(self) -> dict[str, NDArray]:
        """
        Empty by contract. The rotation is fixed, so there is nothing to learn
        and nothing for an optimizer to step -- reporting the input gradient
        here would invite SGD to treat it as an update.
        """
        return {}

    @property
    def num_parameters(self) -> int:
        return 0

    def __str__(self):
        return (
            f"RoPE embedding matrix of {self.sequence_length} length, "
            f"at {self.embedding_dimension} embedding dimension"
        )

    def __repr__(self):
        return self.__str__()


class SinusoidEmbedding(Layer):
    """
    Fixed sine and cosine positional embedding, added to the input rather than
    rotated into it.
    Sine is applied to even positons, cosine applied to odds

    """

    def __init__(self,
                 sequence_length: int,
                 embedding_dimension: int,
                 base_freq: int = 10000):
        """
        Parameters
        ----------
        sequence_length : the ceiling this table is built for. A shorter
            sequence is fine, it takes the leading rows
        embedding_dimension : channel width of the incoming embedding
        base_freq : wavelength base, 10,000 in the paper
        """
        super().__init__()
        assert embedding_dimension % 2 == 0, (
            f"embedding_dimension must be even, got {embedding_dimension}"
        )

        self.sequence_length: int = sequence_length
        self.embedding_dimension: int = embedding_dimension
        self.base_freq: int = base_freq
        self.declare_shapes(
            inputs=((embedding_dimension,),),
            outputs=((embedding_dimension,),),
        )

        self.build_sinusoid_array(sequence_length, embedding_dimension, base_freq)

    def build_sinusoid_array(self,
                             sequence_length: int,
                             embedding_dimension: int,
                             base_freq: int = 10000
    ) -> None:
        """
        Precompute the table, shaped (sequence, dimension).

        Cached on the instance rather than returned, since it is fixed for the
        life of the layer and every forward pass reads the same rows.

        Parameters
        ----------
        sequence_length : rows to build, the longest sequence to be accepted
        embedding_dimension : channel width, giving dimension / 2 frequencies
        base_freq : wavelength base, 10,000 in the paper
        """
        inv_freq = 1.0 / (
            base_freq ** (np.arange(0, embedding_dimension, 2) / embedding_dimension)
        )
        position_int = np.arange(sequence_length, dtype=float)
        rotation_angle = np.einsum("i,j->ij", position_int, inv_freq)

        self.pos_table = np.zeros((sequence_length, embedding_dimension))
        self.pos_table[:, 0::2] = np.sin(rotation_angle)
        self.pos_table[:, 1::2] = np.cos(rotation_angle)

    def forward(self, input_data: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        input_data : (sequence, dimension) or (batch, sequence, dimension)
        mask : unused -- each position adds its own row of the fixed table,
            independent of every other position. Accepted for pass-through
            compatibility with the graph.

        Returns
        -------
        the input with the position table added. A sum, unlike RoPE, so norms
        are not preserved.
        """
        assert input_data.ndim >= 2, (
            f"expected at least (sequence, dimension), got {input_data.shape}"
        )
        sequence, dimension = input_data.shape[-2:]
        assert dimension == self.embedding_dimension, (
            f"built for embedding_dimension {self.embedding_dimension}, got "
            f"{dimension}"
        )
        assert sequence <= self.sequence_length, (
            f"position table holds {self.sequence_length} positions, got a "
            f"sequence of {sequence}"
        )

        # (sequence, dimension) broadcasts against a leading batch axis, so one
        # path serves both ranks
        self.output = input_data + self.pos_table[:sequence]

        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        """
        Adding a constant has an identity Jacobian, so the gradient passes
        through untouched. The table is fixed, so it absorbs none of it.
        """
        self.gradient = incoming_gradient

        return self.gradient

    @property
    def sinusoid_array(self) -> NDArray:
        """the table, copied so callers cannot corrupt it"""
        return self.pos_table.copy()

    def purge(self) -> None:
        """
        Clear the pass state. The table survives: it is a fixed constant, not
        an activation, and rebuilding it per pass would be waste.
        """
        self.output = None
        self.gradient = None

    def update_weights(self, **kwargs) -> None:
        pass

    def zero_gradients(self) -> None:
        pass

    def get_weights(self, for_serialize: bool = False):
        return {} if for_serialize else None

    def set_weights(self, weights: dict) -> None:
        pass

    def get_gradients(self) -> dict[str, NDArray]:
        """empty by contract, the table is fixed and there is nothing to learn"""
        return {}

    @property
    def num_parameters(self) -> int:
        return 0

    def __str__(self):
        return (
            f"sinusoid embedding table of {self.sequence_length} length, "
            f"at {self.embedding_dimension} embedding dimension"
        )

    def __repr__(self):
        return self.__str__()
