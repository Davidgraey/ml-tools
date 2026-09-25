"""
Spectre encoder example: a small masked-language-model (MLM) demo.

Fits a tokenizer on a synthetic corpus, then wires one Spectre transformer
block into a NeuralNetwork graph:

    TextEmbedding -> [prenorm RMSNorm -> SpectreAttention -> FC(swish) ->
    FC(swish) -> RMSNorm -> Dropout -> residual via LatentSum] -> branches
    into a PoolingLayer (sentence-level output, shown but not trained) and
    a token-prediction head (trained against a masked-token objective).

The pooling branch mirrors a language-transformer's pooler (BERT's [CLS]
pooling head): it's a legitimate encoder output, just not one an MLM
objective trains against, since MLM needs a prediction at every position,
not one pooled vector. Both branches share the same encoder, and the
network computes both every forward pass -- only the token-prediction head
feeds the loss.

Run: python spectre_encoder_example.py
"""

import tempfile
import time
from pathlib import Path

import numpy as np
from polyergalio.encoders.tokenizer import (
    SentencePieceTokenizer,
    fit_tokenizer,
    pad_sequences,
)
from polyergalio.generators.data_generators import token_accuracy
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.embedding.embedding import TextEmbedding
from polyergalio.models.embedding.positional import RopeEmbedding
from polyergalio.models.layers.basal_layers import (
    DropoutLayer,
    FullyConnectedLayer,
    RMSNormLayer,
)
from polyergalio.models.layers.mixture_layers import PoolingLayer
from polyergalio.models.layers.operator_layers import LatentSum, MaskGather
from polyergalio.models.layers.spectre_layers import SpectreAttention
from polyergalio.models.model_loss import CrossEntropyLoss
from polyergalio.models.neural_network import NeuralNetwork
from polyergalio.models.optimizers import SGD

TARGET_VOCAB_SIZE = 30_000
CORPUS_LINES = 4000
SEQUENCE_LENGTH = 600
HIDDEN_DIM = 800
FFN_HIDDEN = 1200
NUM_HEADS = 4
DROPOUT_PROB = 0.1
MASK_PROB = 0.25
BATCH_SIZE = 10
LEARNING_RATE = 0.005
TRAIN_STEPS = 5

BASE_WORDS = (
    "the a an of to in on at for with and or but if then else when while "
    "is are was were be been being have has had do does did will would "
    "can could may might must shall should not no yes very quite rather "
    "cat dog bird fish tree river mountain city street house door window "
    "light dark red blue green yellow small large fast slow quiet loud "
    "run walk jump swim fly climb write read speak listen watch build "
    "day night morning evening year month week hour minute second time "
    "person people child adult friend stranger neighbor family group team "
    "work play rest sleep eat drink cook clean fix break make create end"
).split()


def generate_corpus(rng: np.random.Generator, num_lines: int) -> list[str]:
    """
    Synthetic sentences: mostly common words, with random letter-strings
    mixed in for subword diversity BPE can build a larger vocabulary from.

    Parameters
    ----------
    rng : source of randomness
    num_lines : number of sentences to generate

    Returns
    -------
    one sentence per line
    """
    lines = []
    for _ in range(num_lines):
        length = rng.integers(5, 13)
        words = []
        for _ in range(length):
            if rng.random() < 0.4:
                letters = rng.integers(97, 123, size=rng.integers(3, 10))
                words.append("".join(chr(letter) for letter in letters))
            else:
                words.append(rng.choice(BASE_WORDS))
        lines.append(" ".join(words))
    return lines


def build_dataset(
    tokenizer: SentencePieceTokenizer, corpus: list[str], rng: np.random.Generator
) -> tuple:
    """
    Encode a slice of the corpus into a padded, masked MLM batch.

    Returns
    -------
    masked_ids : (BATCH_SIZE, SEQUENCE_LENGTH) inputs with masked positions
        replaced by the MASK token
    target_ids : (BATCH_SIZE, SEQUENCE_LENGTH) the original token ids
    attention_mask : (BATCH_SIZE, SEQUENCE_LENGTH) 1 for real content
    mlm_mask : (BATCH_SIZE, SEQUENCE_LENGTH) bool, the positions being predicted
    """
    encoded = [tokenizer.encode(line) for line in corpus[:BATCH_SIZE]]
    padded, mask = pad_sequences(
        encoded, max_length=SEQUENCE_LENGTH, pad_token=tokenizer.special_tokens.PAD
    )
    target_ids = np.asarray(padded)
    attention_mask = np.asarray(mask)

    mlm_mask = (rng.random(target_ids.shape) < MASK_PROB) & (attention_mask == 1)
    masked_ids = target_ids.copy()
    masked_ids[mlm_mask] = tokenizer.special_tokens.MASK
    return masked_ids, target_ids, attention_mask, mlm_mask


def onehot_targets(target_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """direct index assignment -- see the module docstring for why not to_onehot"""
    targets = np.zeros((*target_ids.shape, vocab_size))
    row_idx = np.arange(target_ids.shape[0])
    targets[row_idx, target_ids] = 1.0
    return targets


def build_network(vocab_size: int, padding_idx: int) -> NeuralNetwork:
    """
    TextEmbedding -> one Spectre transformer block -> [pooled, mlm_head].

    Parameters
    ----------
    vocab_size : the tokenizer's actual fitted vocabulary size
    padding_idx : id whose embedding row is fixed at zero

    Returns
    -------
    NeuralNetwork whose output node is the MLM head: (batch, sequence, vocab_size)
    """
    net = NeuralNetwork(name="spectre_encoder", input_shape=(SEQUENCE_LENGTH,))
    embedding = net.connect(
        TextEmbedding(TARGET_VOCAB_SIZE, HIDDEN_DIM, padding_idx=padding_idx),
        net.input,
        name="embedding",
    )
    positional = net.connect(
        RopeEmbedding(sequence_length=SEQUENCE_LENGTH, embedding_dimension=HIDDEN_DIM),
        embedding,
        name="positional_emb",
    )

    prenorm = net.connect(RMSNormLayer(HIDDEN_DIM), positional, name="prenorm")
    attention = net.connect(
        SpectreAttention(
            SEQUENCE_LENGTH,
            HIDDEN_DIM,
            num_heads=NUM_HEADS,
            memory_tokens=128,
            use_wrm=True,
        ),
        prenorm,
        name="attention",
    )
    ffn_1 = net.connect(
        FullyConnectedLayer(HIDDEN_DIM, FFN_HIDDEN, "swish"), attention, name="ffn_1"
    )
    ffn_2 = net.connect(
        FullyConnectedLayer(FFN_HIDDEN, HIDDEN_DIM, "swish"), ffn_1, name="ffn_2"
    )
    postnorm = net.connect(RMSNormLayer(HIDDEN_DIM), ffn_2, name="postnorm")
    dropped = net.connect(DropoutLayer(DROPOUT_PROB), postnorm, name="dropout")
    block_out = net.connect(LatentSum(), dropped, embedding, name="residual")

    net.connect(PoolingLayer(), block_out, name="pooled")
    gathered = net.connect(MaskGather(), block_out, name="mask_gather")
    net.output = net.connect(
        FullyConnectedLayer(HIDDEN_DIM, TARGET_VOCAB_SIZE, "linear", is_output=True),
        gathered,
        name="mlm_head",
    )
    return net


def main():
    rng = np.random.default_rng(0)
    corpus = generate_corpus(rng, CORPUS_LINES)

    with tempfile.TemporaryDirectory() as workdir:
        corpus_path = Path(workdir) / "corpus.txt"
        corpus_path.write_text("\n".join(corpus))

        model_path = fit_tokenizer(
            corpus_paths=[str(corpus_path)],
            model_prefix=str(Path(workdir) / "spectre_tokenizer"),
            vocab_size=TARGET_VOCAB_SIZE,
            hard_vocab_limit=False,
        )
        tokenizer = SentencePieceTokenizer(model_path)

    vocab_size = tokenizer.get_vocab_size()
    print(f"requested vocab_size={TARGET_VOCAB_SIZE}, corpus fit {vocab_size}")

    # TODO: add in the encoder.text_encoders.py
    masked_ids, target_ids, attention_mask, mlm_mask = build_dataset(
        tokenizer, corpus, rng
    )
    target_ids_gathered = target_ids[mlm_mask]
    targets = onehot_targets(target_ids_gathered, vocab_size)

    net = build_network(vocab_size, padding_idx=tokenizer.special_tokens.PAD)
    print(net.summary())

    loss_fn = CrossEntropyLoss(task=ClassificationTask.MULTINOMIAL)
    optimizer = SGD(LEARNING_RATE)

    net.eval()
    logits = net.forward(masked_ids, mask=attention_mask, target_mask=mlm_mask)
    predicted_ids = np.argmax(logits, axis=-1)
    print(predicted_ids.shape, target_ids_gathered.shape, mlm_mask.shape)
    print(
        f"masked-token accuracy before training: {token_accuracy(predicted_ids, target_ids_gathered):.3f}"
    )

    net.train()
    for step in range(TRAIN_STEPS):
        net.zero_gradients()
        st = time.time()
        logits = net.forward(masked_ids, mask=attention_mask, target_mask=mlm_mask)
        loss = loss_fn(logits, targets)
        net.backward(loss_fn.backward())
        optimizer.step(net.layers)
        if step % 5 == 0:
            print(f"step {step:4d}  loss {loss:.4f}")
            print(f"training step took {time.time() - st}")

    net.eval()
    st = time.time()
    logits = net.forward(masked_ids, mask=attention_mask, target_mask=mlm_mask)
    predicted_ids = np.argmax(logits, axis=-1)
    print(f"eval forward step {time.time() - st}")
    print(
        f"masked-token accuracy after training: {token_accuracy(predicted_ids, target_ids_gathered):.3f}"
    )
    print(f"pooled sentence representation shape: {net.activations['pooled'].shape}")


if __name__ == "__main__":
    main()
