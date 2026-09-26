"""
Dataset generators: the same seed gives the same data, and the planted
structure is actually recoverable, since the rest of the suite relies on both.
"""

import numpy as np
import pytest

from polyergalio.generators.data_generators import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SEQUENCE_TASKS,
    TOKEN_OFFSET,
    RandomDatasetGenerator,
    to_int_classes,
    to_onehot,
)


TABULAR_TASKS = ("regression", "binary", "multiclass", "multilabel", "clustering")
ALL_TASKS = TABULAR_TASKS + ("signal", "image", "sequence")


@pytest.mark.parametrize("task", ALL_TASKS)
def test_same_seed_reproduces_the_dataset(task):
    first = RandomDatasetGenerator(random_seed=7).generate(
        task, num_samples=30, verbose=False
    )[0]
    second = RandomDatasetGenerator(random_seed=7).generate(
        task, num_samples=30, verbose=False
    )[0]
    assert np.array_equal(first, second)


# -------------    label encodings    ------------------------------
def test_onehot_round_trips():
    classes = np.array([0, 2, 1, 2])
    encoded = to_onehot(classes, num_classes=3)
    assert encoded.shape == (4, 3)
    assert np.array_equal(to_int_classes(encoded), classes)


# -------------    tabular tasks    --------------------------------
def test_regression_target_follows_the_planted_weights(generator):
    x_data, y_data, meta = generator.generate(
        "regression", num_samples=2000, num_features=4, noise_scale=0.01,
        verbose=False,
    )
    predicted = x_data @ meta["weights"] + meta["bias"]
    assert np.corrcoef(predicted, y_data)[0, 1] > 0.99


def test_clustering_samples_sit_near_their_centroid(generator):
    x_data, y_data, meta = generator.generate(
        "clustering", num_samples=600, num_features=2, num_clusters=4,
        noise_scale=0.2, verbose=False,
    )
    for index, centroid in enumerate(meta["centroids"]):
        members = x_data[y_data == index]
        assert np.linalg.norm(members.mean(axis=0) - centroid) < 0.3


# -------------    signal and image tasks    -----------------------
def test_signal_classes_are_separable(generator):
    """
    Spectral features should classify the families well above chance,
    otherwise the task is not learnable and tests using it prove nothing.
    """
    x_data, y_data, meta = generator.generate(
        "signal", num_samples=400, signal_length=256, num_classes=5,
        noise_scale=0.0, verbose=False,
    )
    features = np.abs(np.fft.rfft(x_data, axis=-1))
    features = features / (features.max(axis=-1, keepdims=True) + 1e-12)

    train, test = slice(0, 200), slice(200, 400)
    classes = len(meta["class_names"])
    centroids = np.array(
        [features[train][y_data[train] == k].mean(0) for k in range(classes)]
    )
    predicted = np.argmin(
        ((features[test][:, None, :] - centroids[None]) ** 2).sum(-1), axis=1
    )
    assert (predicted == y_data[test]).mean() > 0.5


def test_image_classes_are_separable_by_structure(generator):
    """
    Raw pixels classify poorly because the shapes move and rotate, so this
    uses position and scale invariant descriptors -- the signal is structural.
    """
    x_data, y_data, meta = generator.generate(
        "image", num_samples=400, image_size=24, num_classes=4, noise_scale=0.0,
        verbose=False,
    )
    size = x_data.shape[1]
    axis = np.linspace(-1, 1, size)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")

    descriptors = []
    for image in x_data:
        mask = image > 0.5
        rows, columns = grid_y[mask], grid_x[mask]
        centre_y, centre_x = rows.mean(), columns.mean()
        radius = np.sqrt((rows - centre_y) ** 2 + (columns - centre_x) ** 2)
        largest = radius.max() if radius.max() > 0 else 1.0
        descriptors.append([
            mask.sum() / (np.pi * (largest * size / 2) ** 2 + 1e-9),
            radius.mean() / largest,
            radius.std() / largest,
            (radius < 0.3 * largest).sum() / mask.sum(),
        ])
    descriptors = np.asarray(descriptors)

    train, test = slice(0, 200), slice(200, 400)
    classes = len(meta["class_names"])
    centroids = np.array(
        [descriptors[train][y_data[train] == k].mean(0) for k in range(classes)]
    )
    predicted = np.argmin(
        ((descriptors[test][:, None, :] - centroids[None]) ** 2).sum(-1), axis=1
    )
    assert (predicted == y_data[test]).mean() > 0.7


# -------------    sequence task    --------------------------------
def content_and_target(x_data, y_data, meta, row):
    """A row's real tokens, EOS stripped from both source and target."""
    source = x_data[row, :meta["source_lengths"][row] - 1]
    target = y_data[row, :meta["target_lengths"][row] - 1]
    return source, target


@pytest.mark.parametrize("task", SEQUENCE_TASKS)
def test_sequence_shapes_and_padding(task, generator):
    x_data, y_data, meta = generator.generate(
        "sequence", num_samples=60, sequence_task=task, vocab_size=12,
        min_seq_length=4, max_seq_length=10, verbose=False,
    )
    assert x_data.shape == y_data.shape
    assert meta["decoder_input"].shape == y_data.shape
    encoder_mask, decoder_mask = meta["encoder_padding_mask"], meta["decoder_padding_mask"]
    assert (x_data[~encoder_mask] == PAD_ID).all()
    assert (y_data[~decoder_mask] == PAD_ID).all()
    # every real content run ends in EOS, right where its length says it does
    for row, length in enumerate(meta["source_lengths"]):
        assert x_data[row, length - 1] == EOS_ID
    for row, length in enumerate(meta["target_lengths"]):
        assert y_data[row, length - 1] == EOS_ID


def test_sequence_decoder_input_is_target_shifted_by_one(generator):
    """decoder_input is BOS + target; comparing it to y one step over is
    exactly the teacher-forcing shift a decoder trains against."""
    _, y_data, meta = generator.generate(
        "sequence", num_samples=40, sequence_task="copy", verbose=False
    )
    decoder_input = meta["decoder_input"]
    assert (decoder_input[:, 0] == BOS_ID).all()
    for row, length in enumerate(meta["target_lengths"]):
        assert np.array_equal(decoder_input[row, 1:length], y_data[row, :length - 1])


def test_sort_target_is_source_sorted(generator):
    x_data, y_data, meta = generator.generate(
        "sequence", num_samples=100, sequence_task="sort", verbose=False
    )
    for row in range(len(x_data)):
        source, target = content_and_target(x_data, y_data, meta, row)
        assert np.array_equal(np.sort(source), target)
        assert (np.diff(target) >= 0).all()


def test_add_target_decodes_to_the_correct_sum(generator):
    x_data, y_data, meta = generator.generate(
        "sequence", num_samples=200, sequence_task="add", num_digits=3, verbose=False
    )
    for row in range(len(x_data)):
        length = meta["target_lengths"][row] - 1
        digits = y_data[row, :length] - TOKEN_OFFSET
        predicted = int("".join(str(d) for d in digits))
        a_value, b_value = meta["operands"][row]
        assert predicted == a_value + b_value == meta["sums"][row]
