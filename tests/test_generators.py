"""
Dataset generators.

The point of a generator is that a test can rely on it, so the properties
asserted here are the ones the rest of the suite depends on: the same seed
gives the same data, labels are balanced, and the planted structure is
actually recoverable. A generator that quietly produced unlearnable data would
make every downstream test meaningless.
"""

import numpy as np
import pytest

from ml_tools.generators.data_generators import (
    BOS_ID,
    EOS_ID,
    IMAGE_SHAPES,
    PAD_ID,
    SEQUENCE_TASKS,
    SIGNAL_FAMILIES,
    TOKEN_OFFSET,
    RandomDatasetGenerator,
    sequence_exact_match,
    to_int_classes,
    to_multilabel,
    to_onehot,
    token_accuracy,
)


TABULAR_TASKS = ("regression", "binary", "multiclass", "multilabel", "clustering")
ALL_TASKS = TABULAR_TASKS + ("signal", "image", "sequence")


# -------------    the dispatcher    -------------------------------
@pytest.mark.parametrize("task", ALL_TASKS)
def test_every_task_returns_a_triple(task, generator):
    x_data, y_data, meta = generator.generate(
        task, num_samples=40, num_features=4, verbose=False
    )
    assert len(x_data) == len(y_data)
    assert isinstance(meta, dict)


def test_unknown_task_raises(generator):
    with pytest.raises(ValueError):
        generator.generate("nonsense", verbose=False)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_same_seed_reproduces_the_dataset(task):
    """
    The old periodic_signal_gen reached for the global np.random and so
    silently ignored the seed. Everything here must honour it.
    """
    first = RandomDatasetGenerator(random_seed=7).generate(
        task, num_samples=30, verbose=False
    )[0]
    second = RandomDatasetGenerator(random_seed=7).generate(
        task, num_samples=30, verbose=False
    )[0]
    assert np.array_equal(first, second)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_different_seeds_differ(task):
    first = RandomDatasetGenerator(random_seed=7).generate(
        task, num_samples=30, verbose=False
    )[0]
    other = RandomDatasetGenerator(random_seed=8).generate(
        task, num_samples=30, verbose=False
    )[0]
    assert not np.array_equal(first, other)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_output_is_finite(task, generator):
    x_data, _, _ = generator.generate(task, num_samples=40, verbose=False)
    assert np.isfinite(x_data).all()


# -------------    label encodings    ------------------------------
def test_onehot_round_trips():
    classes = np.array([0, 2, 1, 2])
    encoded = to_onehot(classes, num_classes=3)
    assert encoded.shape == (4, 3)
    assert np.array_equal(to_int_classes(encoded), classes)


def test_onehot_infers_the_class_count():
    assert to_onehot(np.array([0, 3])).shape == (2, 4)


def test_multilabel_sets_every_listed_index():
    encoded = to_multilabel([[0, 2], [1]], num_classes=3)
    assert np.array_equal(encoded, [[1, 0, 1], [0, 1, 0]])


def test_to_int_classes_round_trips_multilabel():
    """
    is_multilabel rounds each label on its own -- unlike the single-class
    branch, there is no axis to argmax over, since more than one label can
    be active per row.
    """
    encoded = to_multilabel([[0, 2], [1]], num_classes=3)
    assert np.array_equal(to_int_classes(encoded, is_multilabel=True), encoded)


@pytest.mark.parametrize("task", ("binary", "multiclass", "signal", "image"))
def test_onehot_flag_widens_the_target(task, generator):
    _, y_data, _ = generator.generate(
        task, num_samples=40, num_classes=3, onehot=True, verbose=False
    )
    assert y_data.ndim == 2


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


def test_multilabel_guarantees_at_least_one_label(generator):
    _, y_data, _ = generator.generate(
        "multilabel", num_samples=300, num_classes=5, ensure_label=True,
        verbose=False,
    )
    assert (y_data.sum(axis=1) >= 1).all()


def test_class_counts_are_reported(generator):
    _, y_data, meta = generator.generate(
        "multiclass", num_samples=200, num_classes=4, verbose=False
    )
    assert meta["class_counts"].sum() == len(y_data)


# -------------    signal task    ----------------------------------
def test_signal_shape_and_balance(generator):
    x_data, y_data, meta = generator.generate(
        "signal", num_samples=100, signal_length=64, num_classes=5, verbose=False
    )
    assert x_data.shape == (100, 64)
    assert tuple(meta["class_names"]) == SIGNAL_FAMILIES
    assert meta["class_counts"].min() >= 100 // 5 - 1


def test_signal_frequencies_stay_below_nyquist(generator):
    _, _, meta = generator.generate(
        "signal", num_samples=200, sample_rate=1000, freq_high=10_000,
        verbose=False,
    )
    assert (meta["frequencies"] < meta["nyquist"]).all()


def test_tone_frequency_is_recoverable(generator):
    """an FFT should find the planted frequency within one bin"""
    x_data, y_data, meta = generator.generate(
        "signal", num_samples=200, signal_length=512, sample_rate=1000,
        num_classes=5, noise_scale=0.0, verbose=False,
    )
    tones = y_data == list(meta["class_names"]).index("tone")
    axis = np.fft.rfftfreq(512, 1 / 1000)
    peaks = axis[np.argmax(np.abs(np.fft.rfft(x_data[tones], axis=-1)), axis=-1)]
    assert np.abs(peaks - meta["frequencies"][tones]).max() <= axis[1]


def test_chirp_sweeps_upward(generator):
    """
    A chirp's instantaneous frequency has to rise, which means integrating the
    sweep rate rather than multiplying a varying frequency by t.
    """
    x_data, y_data, meta = generator.generate(
        "signal", num_samples=200, signal_length=512, sample_rate=1000,
        num_classes=5, noise_scale=0.0, verbose=False,
    )
    chirps = x_data[y_data == list(meta["class_names"]).index("chirp")]
    half = chirps.shape[1] // 2
    axis = np.fft.rfftfreq(half, 1 / 1000)

    first = axis[np.argmax(np.abs(np.fft.rfft(chirps[:, :half], axis=-1)), axis=-1)]
    second = axis[np.argmax(np.abs(np.fft.rfft(chirps[:, half:], axis=-1)), axis=-1)]
    assert (second > first).mean() > 0.8


def test_damped_family_decays(generator):
    x_data, y_data, meta = generator.generate(
        "signal", num_samples=200, signal_length=512, num_classes=5,
        noise_scale=0.0, verbose=False,
    )
    damped = x_data[y_data == list(meta["class_names"]).index("damped")]
    half = damped.shape[1] // 2
    early = np.abs(damped[:, :half]).mean(axis=1)
    late = np.abs(damped[:, half:]).mean(axis=1)
    assert (late < early).all()


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


# -------------    image task    -----------------------------------
def test_image_shape_and_balance(generator):
    x_data, y_data, meta = generator.generate(
        "image", num_samples=80, image_size=16, num_classes=4, verbose=False
    )
    assert x_data.shape == (80, 16, 16)
    assert tuple(meta["class_names"]) == IMAGE_SHAPES
    assert meta["class_counts"].min() >= 80 // 4 - 1


def test_shapes_stay_inside_the_frame(generator):
    """the centre is drawn so the shape cannot be clipped by the border"""
    x_data, _, _ = generator.generate(
        "image", num_samples=120, image_size=32, num_classes=4, noise_scale=0.0,
        verbose=False,
    )
    border = np.concatenate(
        [x_data[:, 0, :], x_data[:, -1, :], x_data[:, :, 0], x_data[:, :, -1]],
        axis=1,
    )
    assert border.max() < 0.5


def test_image_metadata_matches_the_samples(generator):
    x_data, _, meta = generator.generate(
        "image", num_samples=40, image_size=16, verbose=False
    )
    assert len(meta["centres"]) == len(x_data)
    assert len(meta["angles"]) == len(x_data)
    assert len(meta["extents"]) == len(x_data)


def test_every_image_contains_a_shape(generator):
    x_data, _, _ = generator.generate(
        "image", num_samples=60, image_size=24, num_classes=4, noise_scale=0.0,
        verbose=False,
    )
    filled = (x_data > 0.5).sum(axis=(1, 2))
    assert (filled > 0).all(), "an empty image carries no label information"


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


def test_brightness_alone_is_a_weak_signal(generator):
    """
    Mean pixel value carries a little information, since a disc covers more
    area than a ring, but it must not be enough to solve the task.
    """
    x_data, y_data, meta = generator.generate(
        "image", num_samples=400, image_size=24, num_classes=4, noise_scale=0.0,
        verbose=False,
    )
    brightness = x_data.mean(axis=(1, 2))[:, None]
    train, test = slice(0, 200), slice(200, 400)
    classes = len(meta["class_names"])
    centroids = np.array(
        [brightness[train][y_data[train] == k].mean(0) for k in range(classes)]
    )
    predicted = np.argmin(
        ((brightness[test][:, None, :] - centroids[None]) ** 2).sum(-1), axis=1
    )
    accuracy = (predicted == y_data[test]).mean()
    assert accuracy < 0.6, f"brightness alone reaches {accuracy:.2f}"


# -------------    sequence task    --------------------------------
def _content_and_target(x_data, y_data, meta, row):
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


def test_copy_target_matches_source(generator):
    x_data, y_data, meta = generator.generate(
        "sequence", num_samples=100, sequence_task="copy", verbose=False
    )
    for row in range(len(x_data)):
        source, target = _content_and_target(x_data, y_data, meta, row)
        assert np.array_equal(source, target)


def test_reverse_target_is_source_reversed(generator):
    x_data, y_data, meta = generator.generate(
        "sequence", num_samples=100, sequence_task="reverse", verbose=False
    )
    for row in range(len(x_data)):
        source, target = _content_and_target(x_data, y_data, meta, row)
        assert np.array_equal(source[::-1], target)


def test_sort_target_is_source_sorted(generator):
    x_data, y_data, meta = generator.generate(
        "sequence", num_samples=100, sequence_task="sort", verbose=False
    )
    for row in range(len(x_data)):
        source, target = _content_and_target(x_data, y_data, meta, row)
        assert np.array_equal(np.sort(source), target)
        assert (np.diff(target) >= 0).all()


def test_cipher_target_follows_the_planted_map(generator):
    x_data, y_data, meta = generator.generate(
        "sequence", num_samples=100, sequence_task="cipher", vocab_size=12, verbose=False
    )
    cipher_map = meta["cipher_map"]
    assert sorted(cipher_map.tolist()) == list(range(TOKEN_OFFSET, TOKEN_OFFSET + 12))
    for row in range(len(x_data)):
        source, target = _content_and_target(x_data, y_data, meta, row)
        assert np.array_equal(cipher_map[source - TOKEN_OFFSET], target)


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


def test_add_requires_room_for_digits_and_separator(generator):
    with pytest.raises(ValueError):
        generator.generate("sequence", sequence_task="add", vocab_size=5, verbose=False)


def test_sequence_task_must_be_known(generator):
    with pytest.raises(ValueError):
        generator.generate("sequence", sequence_task="nonsense", verbose=False)


def test_vocab_size_must_allow_at_least_two_tokens(generator):
    with pytest.raises(ValueError):
        generator.generate("sequence", vocab_size=1, verbose=False)


def test_seq_length_bounds_are_validated(generator):
    with pytest.raises(ValueError):
        generator.generate("sequence", min_seq_length=10, max_seq_length=4, verbose=False)


def test_token_accuracy_and_exact_match_against_self(generator):
    _, y_data, meta = generator.generate(
        "sequence", num_samples=50, sequence_task="sort", verbose=False
    )
    mask = meta["decoder_padding_mask"]
    assert token_accuracy(y_data, y_data, mask) == 1.0
    assert sequence_exact_match(y_data, y_data, mask) == 1.0

    wrong = y_data.copy()
    wrong[:, 0] = wrong[:, 0] + 1
    assert token_accuracy(wrong, y_data, mask) < 1.0
    assert sequence_exact_match(wrong, y_data, mask) == 0.0


# -------------    configuration edges    --------------------------
@pytest.mark.parametrize("task", ("signal", "image"))
@pytest.mark.parametrize("num_classes", (1, 2, 4))
def test_class_count_selects_a_prefix_of_the_families(task, num_classes, generator):
    _, y_data, meta = generator.generate(
        task, num_samples=24, num_classes=num_classes, verbose=False
    )
    assert len(meta["class_names"]) == num_classes
    assert set(np.unique(y_data)) <= set(range(num_classes))


@pytest.mark.parametrize("task", ("signal", "image"))
def test_fewer_samples_than_classes(task, generator):
    x_data, _, _ = generator.generate(
        task, num_samples=3, num_classes=5, verbose=False
    )
    assert len(x_data) == 3


@pytest.mark.parametrize("task", ("signal", "image"))
def test_zero_classes_is_rejected(task, generator):
    with pytest.raises(ValueError):
        generator.generate(task, num_samples=10, num_classes=0, verbose=False)


def test_tiny_signal_length(generator):
    x_data, _, _ = generator.generate(
        "signal", num_samples=6, signal_length=8, verbose=False
    )
    assert x_data.shape == (6, 8)


def test_tiny_image_size(generator):
    x_data, _, _ = generator.generate(
        "image", num_samples=6, image_size=4, verbose=False
    )
    assert x_data.shape == (6, 4, 4)


# -------------    the older loose functions    --------------------
@pytest.mark.xfail(
    reason="make_phase_mix_dataset ends with `Y = Y - np.array(Y)`, subtracting "
    "Y from itself, so the targets are all zero",
    strict=True,
)
def test_phase_mix_targets_are_not_all_zero():
    from ml_tools.generators.periodic_signal_gen import make_phase_mix_dataset

    _, targets = make_phase_mix_dataset(n_samples=8, signal_len=32)
    assert np.abs(targets).max() > 0


@pytest.mark.xfail(
    reason="both functions in periodic_signal_gen use the global np.random, so "
    "they cannot be seeded and are not reproducible",
    strict=True,
)
def test_multifreq_dataset_is_reproducible():
    from ml_tools.generators.periodic_signal_gen import make_multifreq_dataset

    np.random.seed(0)
    first = make_multifreq_dataset(batch_size=4, seq_len=16, hidden_dim=4)[0]
    second = make_multifreq_dataset(batch_size=4, seq_len=16, hidden_dim=4)[0]
    assert np.array_equal(first, second)
