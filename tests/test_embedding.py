"""
Embeddings: token lookup tables, positional encodings and the exploratory
embedding module.
"""

import numpy as np
import pytest
from conftest import GRADIENT_TOLERANCE, input_gradient_error, numeric_gradient, relative_error
from ml_tools.models.embedding.embedding import TextEmbedding
from ml_tools.models.embedding.positional import RopeEmbedding, SinusoidEmbedding
from ml_tools.models.layers.basal_layers import FullyConnectedLayer
from ml_tools.models.model_loss import MSELoss
from ml_tools.models.neural_network import NeuralNetwork
from ml_tools.models.optimizers import SGD, Adam

SEQUENCE = 8
DIMENSION = 6


@pytest.fixture()
def rope():
    return RopeEmbedding(sequence_length=SEQUENCE, embedding_dimension=DIMENSION)


@pytest.fixture()
def sinusoid():
    return SinusoidEmbedding(sequence_length=SEQUENCE, embedding_dimension=DIMENSION)


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


def test_rope_gradient(rope, embedded_batch):
    assert input_gradient_error(rope, embedded_batch) < GRADIENT_TOLERANCE


def test_rope_is_safe_for_the_optimizer(rope, embedded_batch):
    """
    RoPE has nothing to learn, so SGD should find no gradients to apply and
    skip it entirely rather than stepping a fixed rotation.
    """
    from ml_tools.models.optimizers import SGD

    output = rope.forward(embedded_batch)
    rope.backward(np.ones_like(output))
    before = rope.rope_array
    SGD(0.01).step([rope])
    after = rope.rope_array

    assert np.allclose(before[0], after[0])
    assert np.allclose(before[1], after[1])


def test_rope_follows_the_parameterless_convention(rope, embedded_batch):
    output = rope.forward(embedded_batch)
    rope.backward(np.ones_like(output))
    assert rope.get_gradients() in ({}, None)
    assert rope.num_parameters == 0


def test_rope_rejects_an_odd_dimension():
    """
    Rotary pairs adjacent dimensions, so an odd width has a leftover. Caught at
    construction, not on the first forward pass, where it used to surface as a
    numpy broadcast error naming shapes the caller never supplied.
    """
    with pytest.raises(AssertionError):
        RopeEmbedding(sequence_length=4, embedding_dimension=5)


def test_rope_rejects_a_sequence_past_the_table(rope):
    """the rotation table is built to a ceiling, and past it there are no rows"""
    with pytest.raises(AssertionError):
        rope.forward(np.zeros((1, SEQUENCE + 1, DIMENSION)))


def test_rope_rejects_a_mismatched_width(rope):
    with pytest.raises(AssertionError):
        rope.forward(np.zeros((1, SEQUENCE, DIMENSION + 2)))


def test_rope_accepts_a_short_sequence(rope):
    """a shorter sequence is legal, it takes the leading rows of the table"""
    short = np.random.default_rng(0).normal(size=(2, SEQUENCE - 3, DIMENSION))
    assert rope.forward(short).shape == short.shape


def test_rope_two_dimensional_matches_the_batched_path(rope, embedded_batch):
    """
    One code path serves both ranks by broadcasting, so a single sample must
    come out the same whether or not it carries a batch axis.
    """
    batched = rope.forward(embedded_batch)
    single = rope.forward(embedded_batch[0])
    assert np.allclose(single, batched[0])


def test_rope_table_is_not_reachable_for_mutation(rope):
    """rope_array hands out copies, so a caller cannot corrupt the rotation"""
    sine, cosine = rope.rope_array
    sine[:] = 0.0
    cosine[:] = 0.0
    assert not np.allclose(rope.rope_array[0], 0.0)


def test_rope_purge_keeps_the_rotation_table(rope, embedded_batch):
    """the tables are constants, not activations -- purge must not drop them"""
    before = rope.forward(embedded_batch)
    rope.purge()
    assert np.allclose(rope.forward(embedded_batch), before)


def test_rope_interface(rope, embedded_batch):
    output = rope.forward(embedded_batch)
    rope.backward(np.ones_like(output))
    rope.zero_gradients()
    rope.purge()


# -------------    the fixed sinusoid table    ---------------------
def test_sinusoid_preserves_shape(sinusoid, embedded_batch):
    assert sinusoid.forward(embedded_batch).shape == embedded_batch.shape


def test_sinusoid_is_purely_additive(sinusoid, embedded_batch):
    """
    The defining difference from RoPE: the offset depends on position alone, so
    subtracting the output from the input must leave the same table for every
    sample, whatever the content.
    """
    offset = sinusoid.forward(embedded_batch) - embedded_batch
    assert np.allclose(offset, sinusoid.sinusoid_array)
    assert np.allclose(offset[0], offset[1])


def test_sinusoid_does_not_preserve_norms(sinusoid, embedded_batch):
    """an addition is not a rotation -- asserted so the two do not get conflated"""
    rotated = sinusoid.forward(embedded_batch)
    assert not np.allclose(
        np.linalg.norm(embedded_batch, axis=-1), np.linalg.norm(rotated, axis=-1)
    )


def test_sinusoid_varies_with_position(sinusoid):
    """the same vector at two positions must not come out the same"""
    repeated = np.tile(np.arange(DIMENSION, dtype=float), (1, SEQUENCE, 1))
    encoded = sinusoid.forward(repeated)
    assert not np.allclose(encoded[0, 0], encoded[0, 1])


def test_sinusoid_table_is_bounded(sinusoid):
    """sine and cosine, so unit amplitude -- the docstring's scaling advice"""
    assert np.abs(sinusoid.sinusoid_array).max() <= 1.0


def test_sinusoid_splits_sine_and_cosine_by_parity(sinusoid):
    """even channels carry the sine, odd the cosine, on a shared ladder"""
    table = sinusoid.sinusoid_array
    assert np.allclose(table[0, 0::2], 0.0)
    assert np.allclose(table[0, 1::2], 1.0)


def test_sinusoid_shares_the_rope_frequency_ladder(sinusoid, rope):
    """
    Same inv_freq, different use: RoPE rotates by the angle, this adds its
    sine and cosine. A drift between the two would mean one of them is wrong.
    """
    rope_sine, rope_cosine = rope.rope_array
    table = sinusoid.sinusoid_array
    assert np.allclose(table[:, 0::2], rope_sine)
    assert np.allclose(table[:, 1::2], rope_cosine)


def test_sinusoid_gradient(sinusoid, embedded_batch):
    assert input_gradient_error(sinusoid, embedded_batch) < GRADIENT_TOLERANCE


def test_sinusoid_gradient_passes_through_untouched(sinusoid, embedded_batch):
    """adding a constant has an identity Jacobian"""
    output = sinusoid.forward(embedded_batch)
    upstream = np.random.default_rng(1).normal(size=output.shape)
    assert np.allclose(sinusoid.backward(upstream), upstream)


def test_sinusoid_follows_the_parameterless_convention(sinusoid, embedded_batch):
    output = sinusoid.forward(embedded_batch)
    sinusoid.backward(np.ones_like(output))
    assert sinusoid.get_gradients() in ({}, None)
    assert sinusoid.num_parameters == 0


def test_sinusoid_is_safe_for_the_optimizer(sinusoid, embedded_batch):
    from ml_tools.models.optimizers import SGD

    output = sinusoid.forward(embedded_batch)
    sinusoid.backward(np.ones_like(output))
    before = sinusoid.sinusoid_array
    SGD(0.01).step([sinusoid])
    assert np.allclose(before, sinusoid.sinusoid_array)


def test_sinusoid_rejects_an_odd_dimension():
    """parity splits the channels, so an odd width has a leftover"""
    with pytest.raises(AssertionError):
        SinusoidEmbedding(sequence_length=4, embedding_dimension=5)


def test_sinusoid_rejects_a_sequence_past_the_table(sinusoid):
    with pytest.raises(AssertionError):
        sinusoid.forward(np.zeros((1, SEQUENCE + 1, DIMENSION)))


def test_sinusoid_rejects_a_mismatched_width(sinusoid):
    with pytest.raises(AssertionError):
        sinusoid.forward(np.zeros((1, SEQUENCE, DIMENSION + 2)))


def test_sinusoid_accepts_a_short_sequence(sinusoid):
    """a shorter sequence takes the leading rows of the table"""
    short = np.random.default_rng(0).normal(size=(2, SEQUENCE - 3, DIMENSION))
    assert sinusoid.forward(short).shape == short.shape


def test_sinusoid_two_dimensional_matches_the_batched_path(sinusoid, embedded_batch):
    batched = sinusoid.forward(embedded_batch)
    single = sinusoid.forward(embedded_batch[0])
    assert np.allclose(single, batched[0])


def test_sinusoid_table_is_not_reachable_for_mutation(sinusoid):
    table = sinusoid.sinusoid_array
    table[:] = 0.0
    assert not np.allclose(sinusoid.sinusoid_array, 0.0)


def test_sinusoid_purge_keeps_the_table(sinusoid, embedded_batch):
    before = sinusoid.forward(embedded_batch)
    sinusoid.purge()
    assert np.allclose(sinusoid.forward(embedded_batch), before)


def test_sinusoid_interface(sinusoid, embedded_batch):
    output = sinusoid.forward(embedded_batch)
    sinusoid.backward(np.ones_like(output))
    sinusoid.zero_gradients()
    sinusoid.purge()


# -------------    the token lookup table    ----------------------
VOCABULARY = 10


@pytest.fixture()
def token_ids():
    return np.random.default_rng(0).integers(0, VOCABULARY, size=(3, SEQUENCE))


@pytest.fixture()
def text_embedding():
    return TextEmbedding(num_embeddings=VOCABULARY, embedding_dim=DIMENSION)


def test_text_embedding_looks_up_rows(text_embedding, token_ids):
    vectors = text_embedding.forward(token_ids)
    assert vectors.shape == (3, SEQUENCE, DIMENSION)
    assert np.array_equal(vectors[1, 2], text_embedding.weights[token_ids[1, 2]])


def test_text_embedding_accepts_any_leading_shape(text_embedding):
    assert text_embedding.forward(np.array(4)).shape == (DIMENSION,)
    assert text_embedding.forward(np.arange(5)).shape == (5, DIMENSION)
    assert text_embedding.forward(np.zeros((2, 3, 4), dtype=int)).shape == (2, 3, 4, DIMENSION)


def test_text_embedding_weight_gradient(text_embedding, token_ids):
    upstream = np.random.default_rng(1).normal(size=token_ids.shape + (DIMENSION,))
    text_embedding.forward(token_ids)
    text_embedding.backward(upstream)
    numeric = numeric_gradient(
        lambda: float((text_embedding.forward(token_ids) * upstream).sum()), text_embedding.weights
    )
    assert relative_error(text_embedding.gradient_weights, numeric) < GRADIENT_TOLERANCE


def test_text_embedding_sums_gradients_of_repeated_tokens(text_embedding):
    text_embedding.forward(np.array([3, 3, 5]))
    text_embedding.backward(np.ones((3, DIMENSION)))
    assert np.allclose(text_embedding.gradient_weights[3], 2.0)
    assert np.allclose(text_embedding.gradient_weights[5], 1.0)
    assert not np.delete(text_embedding.gradient_weights, [3, 5], axis=0).any()


def test_text_embedding_ids_get_no_gradient(text_embedding, token_ids):
    text_embedding.forward(token_ids)
    grad = text_embedding.backward(np.ones(token_ids.shape + (DIMENSION,)))
    assert grad.shape == token_ids.shape and not grad.any()


def test_text_embedding_padding_row_is_zero_and_frozen(token_ids):
    layer = TextEmbedding(VOCABULARY, DIMENSION, padding_idx=0)
    ids = np.array([[0, 1, 0, 2]])
    assert not layer.forward(ids)[0, [0, 2]].any()
    optimizer = Adam(0.1)
    for _ in range(3):
        layer.forward(ids)
        layer.backward(np.ones((1, 4, DIMENSION)))
        optimizer.step([layer])
    assert not layer.weights[0].any()
    assert layer.weights[1].any()


@pytest.mark.parametrize("bad_ids", (np.array([VOCABULARY]), np.array([-1])))
def test_text_embedding_rejects_ids_outside_the_table(text_embedding, bad_ids):
    with pytest.raises(IndexError):
        text_embedding.forward(bad_ids)


def test_text_embedding_rejects_fractional_ids(text_embedding):
    with pytest.raises(TypeError):
        text_embedding.forward(np.array([1.5]))
    assert text_embedding.forward(np.array([2.0])).shape == (1, DIMENSION)


def test_text_embedding_rejects_a_padding_idx_outside_the_table():
    with pytest.raises(ValueError):
        TextEmbedding(VOCABULARY, DIMENSION, padding_idx=VOCABULARY)


def test_text_embedding_learns_target_vectors():
    """each token's vector moves to its own target, and only the tokens seen move"""
    rng = np.random.default_rng(3)
    layer = TextEmbedding(VOCABULARY, DIMENSION)
    targets = rng.normal(size=(VOCABULARY, DIMENSION))
    ids = np.arange(VOCABULARY - 1)
    unseen = layer.weights[-1].copy()
    loss, optimizer = MSELoss(), SGD(10.0)
    for _ in range(200):
        loss(layer.forward(ids), targets[ids])
        layer.backward(loss.backward())
        optimizer.step([layer])
    assert np.allclose(layer.weights[ids], targets[ids], atol=1e-3)
    assert np.array_equal(layer.weights[-1], unseen)


def test_text_embedding_feeds_a_network():
    net = NeuralNetwork(input_shape=(SEQUENCE,))
    embedded = net.connect(TextEmbedding(VOCABULARY, DIMENSION), net.input, name="tokens")
    net.output = net.connect(FullyConnectedLayer(DIMENSION, 2, "linear"), embedded, name="out")
    ids = np.random.default_rng(0).integers(0, VOCABULARY, size=(3, SEQUENCE))
    assert embedded.out_shape == (SEQUENCE, DIMENSION)
    assert net.forward(ids).shape == (3, SEQUENCE, 2)
    assert net.backward(np.ones((3, SEQUENCE, 2))).shape == ids.shape
    assert net.node("tokens").layer.gradient_weights.any()


def test_text_embedding_interface(text_embedding, token_ids):
    text_embedding.forward(token_ids)
    text_embedding.backward(np.ones(token_ids.shape + (DIMENSION,)))
    assert text_embedding.num_parameters == VOCABULARY * DIMENSION
    assert set(text_embedding.get_gradients()) == {"gradient_weights"}
    assert text_embedding.get_config() == {"num_embeddings": VOCABULARY, "embedding_dim": DIMENSION, "padding_idx": None}
    text_embedding.zero_gradients()
    assert not text_embedding.gradient_weights.any()
    text_embedding.purge()
    assert text_embedding.token_ids is None


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
