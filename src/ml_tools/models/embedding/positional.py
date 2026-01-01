import numpy as np
from numpy.typing import NDArray

class RopeEmbedding:
    def __init__(self, sequence_length: int, embedding_dimension: int):
        """
        Constructing a decoupled, rotary positional embedding mechanism

        Parameters
        ----------
        sequence_length : the length of sequence for which we're building this
            rotary matrix
        embedding_dimension : the dimension of the prior layer's embedding
            process
        """
        self.sequence_length: int = sequence_length
        self.embedding_dimension: int = embedding_dimension

        self._rope_array = self.build_rope_array(sequence_length, embedding_dimension)

    @staticmethod
    def build_rope_array(
        sequence_length: int, embedding_dimension: int, base_freq: int = 10000
    ) -> NDArray:
        """
        Construct our rope array
        Parameters
        ----------
        sequence_length : the sequence length to build the rotation matrix -
            could be the max seq length of the model
        embedding_dimension : the embedding dimension of the array
        base_freq : base freq for rotary - 10,000 is standard (from paper)

        Returns
        -------
        the rotation matrix - this can be persisted and reused
        """
        inv_freq = 1.0 / (
            base_freq ** (np.arange(0, embedding_dimension, 2) / embedding_dimension)
        )
        position_int = np.arange(sequence_length, dtype=float)
        rotation_angle = np.einsum("i,j->ij", position_int, inv_freq)

        pos_sine = np.sin(rotation_angle)
        pos_cosine = np.cos(rotation_angle)
        embeddings = np.concat([pos_sine, pos_cosine], -1)

        return embeddings

    def __call__(self, input_data: NDArray) -> NDArray:
        """
        Some basic assumptions: Data in input data is structured:
        1) Dimensions are: (num_samples, sequence_length,embedding_dimension)
        2) both sequence_length and embedding dimension are the same as
            those used to initalize this embedding object.

        Parameters
        ----------
        input_data : array of the input data, after the first embedding step:
            (num_samples, sequence_length,embedding_dimension)

        Returns
        -------
        the summed / combined given input_embedding + the RoPE positional
        embeddings
        """
        num_samples, seq_len, embedding_dim = input_data.shape
        assert seq_len <= self.sequence_length
        assert embedding_dim == self.embedding_dimension

        return np.einsum("bnd,nd->bnd", input_data, self.rope_array)

    def __str__(self):
        return (
            f"RoPE embedding matrix of {self.sequence_length} length, "
            f"at {self.embedding_dimension} embedding dimension"
        )

    @property
    def rope_array(self):
        return self._rope_array.copy()
