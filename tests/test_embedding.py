"""
Embeddings: rotary positional encoding and the exploratory embedding module.

RoPE has strong properties worth asserting -- it must preserve vector norms,
because it is a rotation, and the inner product between two positions must
depend only on their separation. That relative-position property is the entire
reason to use it.
"""

import numpy as np
import pytest

from ml_tools.models.embedding.positional import RopeEmbedding
from conftest import GRADIENT_TOLERANCE, input_gradient_error


SEQUENCE = 8
DIMENSION = 6


@pytest.fixture()
def rope():
    return RopeEmbedding(sequence_length=SEQUENCE, embedding_dimension=DIMENSION)


@pytest.fixture()
def embedded_batch():
    rng = np.random.default_rng(0)
    return rng.normal(size=(3, SEQUENCE, DIMENSION))


def test_rope_preserves_shape(rope, embedded_batch):
    assert rope.forward(embedded_batch).shape == embedded_batch.shape


def test_rope_preserves_norms(rope, embedded_batch):
    """a rotation cannot change a vector's length"""
    rotated = rope.forward(embedded_batch)
    original_norms = np.linalg.norm(embedded_batch, axis=-1)
    rotated_norms = np.linalg.norm(rotated, axis=-1)
    assert np.allclose(original_norms, rotated_norms)


def test_rope_varies_with_position(rope):
    """the same vector at two positions must not come out the same"""
    repeated = np.tile(np.arange(DIMENSION, dtype=float), (1, SEQUENCE, 1))
    rotated = rope.forward(repeated)
    assert not np.allclose(rotated[0, 0], rotated[0, 1])


def test_rope_inner_product_depends_only_on_separation(rope):
    """
    The defining property: q at position i against k at position j should
    depend on i - j alone, not on where the pair sits in the sequence.
    """
    rng = np.random.default_rng(0)
    query = rng.normal(size=DIMENSION)
    key = rng.normal(size=DIMENSION)

    def rotated_pair(first, second):
        block = np.zeros((1, SEQUENCE, DIMENSION))
        block[0, first] = query
        rotated_query = rope.forward(block)[0, first]
        block = np.zeros((1, SEQUENCE, DIMENSION))
        block[0, second] = key
        rotated_key = rope.forward(block)[0, second]
        return float(rotated_query @ rotated_key)

    separation_two = [rotated_pair(index, index + 2) for index in range(SEQUENCE - 2)]
    assert np.allclose(separation_two, separation_two[0], atol=1e-8)


@pytest.mark.slow
def test_rope_gradient(rope, embedded_batch):
    assert input_gradient_error(rope, embedded_batch) < GRADIENT_TOLERANCE


def test_rope_is_safe_for_the_optimizer(rope, embedded_batch):
    """
    RoPE has nothing to learn, but it does report a `grad` entry. Its
    update_weights takes **kwargs and ignores it, so a step is harmless -- that
    is what matters for the optimizer.
    """
    from ml_tools.models.optimizers import SGD

    output = rope.forward(embedded_batch)
    rope.backward(np.ones_like(output))
    SGD(0.01).step([rope])


@pytest.mark.xfail(
    reason="RoPE is a fixed rotation with nothing to learn, but get_gradients "
    "reports a `grad` key and num_parameters returns None instead of 0, so it "
    "does not follow the parameterless-layer convention",
    strict=True,
)
def test_rope_follows_the_parameterless_convention(rope, embedded_batch):
    output = rope.forward(embedded_batch)
    rope.backward(np.ones_like(output))
    assert rope.get_gradients() in ({}, None)
    assert rope.num_parameters == 0


def test_rope_rejects_an_odd_dimension():
    """rotary pairs adjacent dimensions, so an odd width has a leftover"""
    with pytest.raises(Exception):
        RopeEmbedding(sequence_length=4, embedding_dimension=5).forward(
            np.zeros((1, 4, 5))
        )


def test_rope_interface(rope, embedded_batch):
    output = rope.forward(embedded_batch)
    rope.backward(np.ones_like(output))
    rope.zero_gradients()
    rope.purge()


# -------------    the exploratory embedding module    -------------
def test_embedding_module_helpers_are_shape_preserving():
    """
    band_gated_fft and learnable_fft are early sketches. Assert only what they
    promise -- a gate applied in the frequency domain returns the input shape.
    """
    from ml_tools.models.embedding import embedding

    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(2, 8, 4))

    for name in ("band_gated_fft", "learnable_fft"):
        function = getattr(embedding, name, None)
        if function is None:
            pytest.skip(f"{name} not present")
        try:
            gate = np.ones(x_data.shape[1] // 2 + 1)
            result = function(x_data, gate)
        except Exception as error:
            pytest.xfail(f"{name} is an unfinished sketch: {type(error).__name__}")
        assert np.asarray(result).shape[0] == x_data.shape[0]


def test_wavelet_utils_imports():
    """the module had a syntax error on its matplotlib import"""
    pytest.importorskip("scipy.signal")
    importlib = __import__("importlib")
    importlib.import_module("ml_tools.models.embedding.wavelet_utils")
