"""
TextProcessor distortions, against a word-level stand-in tokenizer so no
SentencePiece model has to be trained.
"""

import numpy as np
import pytest

pytest.importorskip("sentencepiece")

from polyergalio.encoders.text_encoders import (
    IMPLEMENTED_TASKS,
    DistortionTask,
    TextProcessor,
    split_sentences,
)
from polyergalio.encoders.tokenizer import SpecialTokens

MAX_LENGTH = 32

CORPUS = [
    "the river runs past the old mill. children fish there in the morning. "
    "by evening the water turns gold.",
    "a small dog waits by the door. it hears footsteps on the street. "
    "the door opens and the dog runs out.",
    "rain fell all night. the roads were quiet by dawn.",
]


class WordTokenizer:
    """one id per whitespace-separated word, assigned on first sight"""

    def __init__(self, vocab_size: int = 200):
        self.special_tokens = SpecialTokens()
        self.vocab_size = vocab_size
        self.ids: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        for word in text.split():
            self.ids.setdefault(word, self.special_tokens.TOKEN_OFFSET + len(self.ids))
        return [self.ids[word] for word in text.split()]

    def decode(self, ids: list[int]) -> str:
        words = {value: key for key, value in self.ids.items()}
        return " ".join(words[i] for i in ids if i in words)

    def get_vocab_size(self) -> int:
        return self.vocab_size


@pytest.fixture
def processor():
    processor = TextProcessor(WordTokenizer(), max_length=MAX_LENGTH, random_seed=0)
    processor.fit(CORPUS)
    return processor


def structural_positions(processor, input_ids):
    return np.isin(input_ids, [processor.special.CLS, processor.special.SEP])


# -------------    shared contract    ----------------------------
@pytest.mark.parametrize("task", IMPLEMENTED_TASKS, ids=lambda task: task.value)
def test_every_task_returns_the_shared_fields(processor, task):
    batch = processor.distort_batch(CORPUS, task)
    for name in ("input_ids", "attention_mask", "segment_ids", "target_mask", "labels"):
        assert batch[name].shape == (len(CORPUS), MAX_LENGTH)
    assert batch["task"] == task.value


@pytest.mark.parametrize("task", IMPLEMENTED_TASKS, ids=lambda task: task.value)
def test_targets_never_fall_on_padding(processor, task):
    batch = processor.distort_batch(CORPUS, task)
    assert not (batch["target_mask"] & (batch["attention_mask"] == 0)).any()


@pytest.mark.parametrize(
    "task", [task for task in DistortionTask if task not in IMPLEMENTED_TASKS], ids=lambda task: task.value
)
def test_stubbed_tasks_raise(processor, task):
    with pytest.raises(NotImplementedError):
        processor.distort(CORPUS[0], task)


def test_split_sentences():
    assert split_sentences("One. Two! Three? four") == ["One.", "Two!", "Three?", "four"]


# -------------    cloze    ----------------------------
def test_cloze_labels_hold_the_original_ids(processor):
    sample = processor.distort(CORPUS[0], DistortionTask.CLOZE)
    original = processor.single_sequence(CORPUS[0])
    np.testing.assert_array_equal(sample["labels"], original)
    assert sample["target_mask"].any()


def test_cloze_never_targets_structural_tokens(processor):
    sample = processor.distort(CORPUS[0], DistortionTask.CLOZE)
    assert not sample["target_mask"][structural_positions(processor, sample["labels"])].any()


def test_cloze_leaves_untargeted_positions_alone(processor):
    sample = processor.distort(CORPUS[1], DistortionTask.CLOZE)
    untouched = ~sample["target_mask"]
    np.testing.assert_array_equal(sample["input_ids"][untouched], sample["labels"][untouched])


# -------------    replaced token detection    ----------------------------
def test_replaced_token_labels_mark_changed_positions(processor):
    sample = processor.distort(CORPUS[0], DistortionTask.REPLACED_TOKEN)
    original = processor.single_sequence(CORPUS[0])
    np.testing.assert_array_equal(sample["labels"], (sample["input_ids"] != original).astype(int))


def test_replaced_token_scores_every_content_position(processor):
    sample = processor.distort(CORPUS[0], DistortionTask.REPLACED_TOKEN)
    content = ~structural_positions(processor, processor.single_sequence(CORPUS[0]))
    np.testing.assert_array_equal(sample["target_mask"], content)


# -------------    span boundary    ----------------------------
def test_span_tokens_point_at_observed_boundaries(processor):
    sample = processor.distort(CORPUS[0], DistortionTask.SPAN_BOUNDARY)
    targets = np.flatnonzero(sample["target_mask"])
    assert targets.size

    for position in targets:
        left, right = sample["span_left"][position], sample["span_right"][position]
        assert left < position < right
        assert not sample["target_mask"][left] and not sample["target_mask"][right]
        assert sample["span_offset"][position] == position - left


def test_span_fields_are_empty_outside_spans(processor):
    sample = processor.distort(CORPUS[0], DistortionTask.SPAN_BOUNDARY)
    outside = ~sample["target_mask"]
    for name in ("span_left", "span_right", "span_offset"):
        assert (sample[name][outside] == -1).all()


def test_spans_mask_roughly_mask_prob_of_the_content(processor):
    sample = processor.distort(CORPUS[1], DistortionTask.SPAN_BOUNDARY)
    content = (~structural_positions(processor, sample["labels"])).sum()
    assert sample["target_mask"].sum() == max(1, round(processor.mask_prob * content))


# -------------    next sentence    ----------------------------
def test_next_sentence_is_scored_at_cls_only(processor):
    sample = processor.distort(CORPUS[0], DistortionTask.NEXT_SENTENCE)
    assert sample["target_mask"][0] and sample["target_mask"].sum() == 1
    assert sample["labels"][0] in (0, 1)


def test_next_sentence_segments_split_at_the_first_sep(processor):
    sample = processor.distort(CORPUS[0], DistortionTask.NEXT_SENTENCE)
    first_sep = np.flatnonzero(sample["input_ids"] == processor.special.SEP)[0]
    assert (sample["segment_ids"][: first_sep + 1] == 0).all()
    assert (sample["segment_ids"][first_sep + 1:] == 1).all()


def test_a_positive_pair_keeps_the_true_successor(processor):
    processor.negative_sentence_prob = 0.0
    sample = processor.distort("alpha beta. gamma delta.", DistortionTask.NEXT_SENTENCE)
    expected, _ = processor.pair_sequence(
        processor.tokenizer.encode("alpha beta."), processor.tokenizer.encode("gamma delta.")
    )
    np.testing.assert_array_equal(sample["input_ids"], expected)
    assert sample["labels"][0] == 1


def test_negatives_need_a_fitted_pool():
    processor = TextProcessor(WordTokenizer(), max_length=MAX_LENGTH, negative_sentence_prob=1.0)
    with pytest.raises(ValueError, match="fit"):
        processor.distort(CORPUS[0], DistortionTask.NEXT_SENTENCE)


def test_a_long_pair_is_trimmed_to_fit(processor):
    long_text = " ".join(f"w{i}" for i in range(40)) + ". " + " ".join(f"v{i}" for i in range(40)) + "."
    processor.negative_sentence_prob = 0.0
    sample = processor.distort(long_text, DistortionTask.NEXT_SENTENCE)
    assert sample["input_ids"].size == MAX_LENGTH


# -------------    processor interface    ----------------------------
def test_encode_is_undistorted_and_padded(processor):
    encoded = processor.encode(CORPUS)
    assert encoded["text"].shape == (len(CORPUS), MAX_LENGTH)
    np.testing.assert_array_equal(
        encoded["text"][0, : processor.single_sequence(CORPUS[0]).size], processor.single_sequence(CORPUS[0])
    )


def test_inverse_recovers_the_text(processor):
    encoded = processor.encode(["rain fell all night."])
    assert processor.inverse(encoded["text"])["text"][0] == "rain fell all night."
