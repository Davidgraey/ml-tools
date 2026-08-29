"""
Train DFlashSpectreAttention exactly as a decoder trains, and measure what it
buys in accepted block length.

The synthetic target is a second order chain, which is the point: the unary head
reads the context only up to the anchor, so a later slot cannot be resolved
without the token chosen at the slot before, and that is precisely what the
selector conditions on. The noise floor caps the accepted length at something
below the block size, so the ceiling is a number rather than a hope.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ml_tools.models.blocks import DFlashSpectreAttention
from ml_tools.models.optimizers import SGD

RNG = np.random.default_rng(7)
VOCAB, HIDDEN, LENGTH, BLOCK = 16, 32, 20, 4
NOISE = 0.05

TABLE = RNG.integers(0, VOCAB, size=(VOCAB, VOCAB))
EMBED = RNG.normal(scale=0.6, size=(VOCAB, HIDDEN))
POSITION = RNG.normal(scale=0.1, size=(LENGTH, HIDDEN))


def sample_sequences(count, rng):
    """
    A second order chain: the next token is fixed by the previous two, apart
    from a noise floor. Second order is the point -- the unary head sees the
    context only up to the anchor, so resolving a later slot needs the token
    chosen at the slot before, which is exactly what the selector conditions on.
    """
    ids = np.zeros((count, LENGTH), dtype=np.int64)
    ids[:, :2] = rng.integers(0, VOCAB, size=(count, 2))
    for t in range(2, LENGTH):
        step = TABLE[ids[:, t - 2], ids[:, t - 1]]
        noisy = rng.random(count) < NOISE
        ids[:, t] = np.where(noisy, rng.integers(0, VOCAB, size=count), step)
    return ids


def to_hidden(ids):
    return EMBED[ids] + POSITION[None, : ids.shape[1]]


def build(zero_successor):
    layer = DFlashSpectreAttention(
        sequence_length=LENGTH, hidden_dim=HIDDEN, vocab_size=VOCAB,
        num_heads=4, block_size=BLOCK, selector_rank=32, selector_top_k=6,
        sample_from_anchor=True,
    )
    if zero_successor:
        layer.successor_codebook[...] = 0.0
    return layer


def measure(layer, ids, use_selector):
    """mean accepted prefix length over every start position a block fits"""
    hidden = to_hidden(ids)
    lengths, slot_hits = [], np.zeros(BLOCK)
    starts = range(2, LENGTH - BLOCK + 1)
    for start in starts:
        layer.reset_cache()
        layer.prefill(hidden[:, :start])
        logits, block_hidden = layer.draft_block()
        if use_selector:
            path = layer.select_path(logits, block_hidden, ids[:, start - 1])
        else:
            path = np.argmax(logits, axis=-1)
        truth = ids[:, start:start + BLOCK]
        lengths.append(layer.accept_length(path, truth))
        slot_hits += (path == truth).mean(axis=0)
    layer.reset_cache()
    return float(np.mean(lengths)), slot_hits / len(starts)


def report(tag, layer, ids):
    unary, unary_slots = measure(layer, ids, False)
    both, both_slots = measure(layer, ids, True)
    print(f"   {tag:26s} unary {unary:.3f}   + selector {both:.3f}")
    print(f"       per-slot accuracy unary    {np.round(unary_slots, 3)}")
    print(f"       per-slot accuracy selector {np.round(both_slots, 3)}")
    return unary, both


def train(layer, epochs, batches, learning_rate, joint):
    optimizer = SGD(learning_rate=learning_rate)
    rng = np.random.default_rng(11)
    history = []
    for epoch in range(epochs):
        block_running, select_running = 0.0, 0.0
        for _ in range(batches):
            ids = sample_sequences(16, rng)
            hidden = to_hidden(ids)
            labels = layer.block_targets(ids, BLOCK, 1)
            prevs = layer.block_targets(ids, BLOCK, 0)

            logits = layer.draft_forward(hidden)
            block_cost, dlogits = layer.block_loss(logits, labels)
            layer.zero_gradients()
            select_cost, dselect, _ = layer.select_forward(
                logits, layer.draft_hidden, prevs, labels
            )
            layer.draft_backward(dlogits + dselect if joint else dlogits)
            optimizer.step([layer])

            block_running += block_cost
            select_running += select_cost
        history.append((block_running / batches, select_running / batches))
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"       epoch {epoch:3d}  block CE {history[-1][0]:.4f}   "
                  f"selector CE {history[-1][1]:.4f}")
    return history


holdout = sample_sequences(256, np.random.default_rng(99))
ceiling = sum(pow(1.0 - NOISE, m) for m in range(1, BLOCK + 1))
print(f"second order chain, vocab {VOCAB}, noise {NOISE}, block {BLOCK}")
print(f"uniform CE would be ln(vocab) = {np.log(VOCAB):.4f}")
print(f"a drafter that knew the chain exactly would accept {ceiling:.3f} of "
      f"{BLOCK}, the rest being noise no model can see")
print()

for joint in (False, True):
    name = "selector alongside" if not joint else "selector joint with unary"
    print("=" * 68)
    print(name)
    print("=" * 68)
    layer = build(zero_successor=False)
    before = report("before training", layer, holdout)
    print("   training:")
    train(layer, epochs=120, batches=16, learning_rate=0.05, joint=joint)
    after = report("after training", layer, holdout)
    print(f"   accepted length {before[1]:.3f} -> {after[1]:.3f} with selector, "
          f"{before[0]:.3f} -> {after[0]:.3f} unary only")
    print()
