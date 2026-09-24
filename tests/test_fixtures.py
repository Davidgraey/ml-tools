"""
Sanity checks on the dataset fixtures conftest.py builds from
RandomDatasetGenerator: shapes, dtypes, label ranges, and reproducibility.

test_generators.py exercises the generator functions directly; this covers
the fixtures as the rest of the suite actually consumes them.
"""

import numpy as np

from ml_tools.generators import RandomDatasetGenerator

SEED = 42


# -------------    regression    ------------------------------------
def test_regression_dataset_shape_and_dtype(regression_dataset):
    x_data, y_data, _ = regression_dataset
    assert x_data.shape == (1500, 3)
    assert y_data.shape == (1500,)
    assert x_data.dtype == np.float64
    assert y_data.dtype == np.float64


def test_regression_dataset_is_reproducible():
    generator = RandomDatasetGenerator(random_seed=SEED)
    first = generator.generate(
        task="regression", num_samples=1500, num_features=3, noise_scale=1.5,
        verbose=False,
    )
    generator = RandomDatasetGenerator(random_seed=SEED)
    second = generator.generate(
        task="regression", num_samples=1500, num_features=3, noise_scale=1.5,
        verbose=False,
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


# -------------    binary    ------------------------------------------
def test_binary_dataset_shape_and_labels(binary_dataset):
    x_data, y_data, _ = binary_dataset
    assert x_data.shape == (400, 5)
    assert y_data.shape == (400,)
    assert set(np.unique(y_data)) <= {0, 1}


def test_binary_dataset_is_reproducible(generator):
    same_seed = RandomDatasetGenerator(random_seed=SEED)
    repeat = same_seed.generate(
        task="binary", num_samples=400, num_features=5, verbose=False
    )
    original = generator.generate(
        task="binary", num_samples=400, num_features=5, verbose=False
    )
    assert np.array_equal(original[0], repeat[0])
    assert np.array_equal(original[1], repeat[1])


# -------------    multiclass    ---------------------------------------
def test_multiclass_dataset_shape_and_labels(multiclass_dataset):
    x_data, y_data, _ = multiclass_dataset
    assert x_data.shape == (400, 5)
    assert y_data.shape == (400,)
    assert set(np.unique(y_data)) <= set(range(4))


# -------------    multilabel    ----------------------------------------
def test_multilabel_dataset_shape_and_labels(multilabel_dataset):
    x_data, y_data, _ = multilabel_dataset
    assert x_data.shape == (300, 4)
    assert y_data.shape == (300, 4)
    assert set(np.unique(y_data)) <= {0, 1}
    # ensure_label defaults True: every row has at least one active label
    assert (y_data.sum(axis=1) > 0).all()


# -------------    clustering    -----------------------------------------
def test_clustering_dataset_shape_and_labels(clustering_dataset):
    x_data, y_data, meta = clustering_dataset
    assert x_data.shape == (600, 2)
    assert y_data.shape == (600,)
    # outlier_fraction defaults to 0, so no -1 labels here
    assert set(np.unique(y_data)) <= set(range(4))
    assert meta["outlier_count"] == 0


# -------------    signal    ---------------------------------------------
def test_signal_dataset_shape_and_labels(signal_dataset):
    x_data, y_data, meta = signal_dataset
    assert x_data.shape == (100, 128)
    assert y_data.shape == (100,)
    assert set(np.unique(y_data)) <= set(range(5))
    assert meta["sample_rate"] == 1000


# -------------    image    -----------------------------------------------
def test_image_dataset_shape_and_labels(image_dataset):
    x_data, y_data, meta = image_dataset
    assert x_data.shape == (80, 16, 16)
    assert y_data.shape == (80,)
    assert set(np.unique(y_data)) <= set(range(4))
    assert meta["image_size"] == 16


# -------------    sequence    --------------------------------------------
def test_sequence_dataset_shape_and_meta(sequence_dataset):
    """
    X is the encoder input, y is the decoder target (content + EOS,
    right-padded); both share a batch dimension of 150 and a token axis
    padded out to the batch's longest sequence.
    """
    x_data, y_data, meta = sequence_dataset
    assert x_data.shape[0] == 150
    assert y_data.shape[0] == 150
    assert x_data.ndim == 2
    assert y_data.ndim == 2
    assert meta["sequence_task"] == "sort"
    assert meta["decoder_input"].shape == y_data.shape
    assert meta["encoder_padding_mask"].shape == x_data.shape
    assert meta["decoder_padding_mask"].shape == y_data.shape
    # padding masks are boolean-valued (1 for real content, 0 for padding)
    assert set(np.unique(meta["encoder_padding_mask"])) <= {0, 1}


def test_sequence_dataset_is_reproducible():
    generator = RandomDatasetGenerator(random_seed=SEED)
    first = generator.generate(
        task="sequence", num_samples=150, sequence_task="sort", vocab_size=12,
        min_seq_length=4, max_seq_length=10, verbose=False,
    )
    generator = RandomDatasetGenerator(random_seed=SEED)
    second = generator.generate(
        task="sequence", num_samples=150, sequence_task="sort", vocab_size=12,
        min_seq_length=4, max_seq_length=10, verbose=False,
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


# -------------    calibration_data    -------------------------------------
def test_calibration_data_shape_and_correlation(calibration_data):
    """scores and labels share shape, and the score should correlate with the label"""
    scores, labels = calibration_data
    assert scores.shape == labels.shape == (600, 1)
    assert set(np.unique(labels)) <= {0, 1}
    positive_mean = scores[labels == 1].mean()
    negative_mean = scores[labels == 0].mean()
    assert positive_mean > negative_mean
