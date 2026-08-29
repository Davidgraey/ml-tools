"""
The same drafter, drafting real values instead of tokens.

The target is a two channel vector autoregression of order two, so as in the
categorical experiment a later slot cannot be resolved from the context alone --
it needs the value committed at the slot before, which is what the refinement
conditions on.

Acceptance here is a tolerance, not equality, so the ceiling is computable: a
step is acceptable when every channel's innovation falls inside the tolerance,
and the expected accepted length is the sum of that probability raised to each
slot index. Printed alongside the measurement, because a speedup number without
its ceiling means very little.
"""

import sys
from math import erf, sqrt
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ml_tools.models.blocks import DFlashSpectreAttention
from ml_tools.models.draft_heads import ContinuousHead
from ml_tools.models.optimizers import SGD

RNG = np.random.default_rng(7)
CHANNELS, HIDDEN, LENGTH, BLOCK = 2, 32, 20, 4
INNOVATION = 0.02
TOLERANCE = 0.05

FIRST = np.array([[0.6, -0.35], [0.25, 0.55]])
SECOND = np.array([[-0.3, 0.15], [-0.1, -0.25]])
EMBED = RNG.normal(scale=0.6, size=(CHANNELS, HIDDEN))
POSITION = RNG.normal(scale=0.1, size=(LENGTH, HIDDEN))


def sample_series(count, rng):
    series = np.zeros((count, LENGTH, CHANNELS))
    series[:, :2] = rng.normal(scale=0.5, size=(count, 2, CHANNELS))
    for step in range(2, LENGTH):
        series[:, step] = (
            series[:, step - 1] @ FIRST.T
            + series[:, step - 2] @ SECOND.T
            + rng.normal(scale=INNOVATION, size=(count, CHANNELS))
        )
    return series


def to_hidden(series):
    return series @ EMBED + POSITION[None, :series.shape[1]]


def ceiling():
    """expected accepted length if the chain itself were known exactly"""
    per_channel = erf(TOLERANCE / (INNOVATION * sqrt(2.0)))
    per_step = per_channel ** CHANNELS
    return sum(per_step ** slot for slot in range(1, BLOCK + 1))


def build():
    head = ContinuousHead(
        HIDDEN, BLOCK, CHANNELS, selector_rank=32,
        absolute_tolerance=TOLERANCE, relative_tolerance=0.0,
        sample_from_anchor=True,
    )
    layer = DFlashSpectreAttention(
        sequence_length=LENGTH, hidden_dim=HIDDEN, num_heads=4,
        block_size=BLOCK, head=head,
    )
    # start the log variance at zero rather than at random: exp(-logvar) on a
    # random draw makes the first likelihoods enormous and the first steps wild
    layer.fc_unary.weights[:, CHANNELS:] = 0.0
    layer.fc_unary.bias[:, CHANNELS:] = 0.0
    layer.zero_gradients()
    return layer


def measure(layer, series):
    hidden = to_hidden(series)
    refined_lengths, plain_lengths = [], []
    sigma = []
    for start in range(2, LENGTH - BLOCK + 1):
        layer.reset_cache()
        layer.prefill(hidden[:, :start])
        prediction, block_hidden = layer.draft_block()
        mean, spread = layer.head.mean_and_sigma(prediction)
        sigma.append(spread.mean())

        truth = series[:, start:start + BLOCK]
        refined = layer.select_path(prediction, block_hidden, series[:, start - 1])
        refined_lengths.append(layer.accept_length(refined, truth))
        plain_lengths.append(layer.accept_length(mean, truth))
    layer.reset_cache()
    return (float(np.mean(plain_lengths)), float(np.mean(refined_lengths)),
            float(np.mean(sigma)))


def train(layer, epochs, batches, learning_rate):
    optimizer = SGD(learning_rate=learning_rate)
    rng = np.random.default_rng(11)
    for epoch in range(epochs):
        likelihood, refinement = 0.0, 0.0
        for _ in range(batches):
            series = sample_series(16, rng)
            labels = layer.block_targets(series, BLOCK, 1)
            predecessors = layer.block_targets(series, BLOCK, 0)

            prediction = layer.draft_forward(to_hidden(series))
            cost, gradient = layer.block_loss(prediction, labels)
            layer.zero_gradients()
            refine_cost, drefine, _ = layer.select_forward(
                prediction, layer.draft_hidden, predecessors, labels
            )
            layer.draft_backward(gradient + drefine)
            optimizer.step([layer])

            likelihood += cost
            refinement += refine_cost
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"       epoch {epoch:3d}  NLL {likelihood / batches:8.4f}   "
                  f"refinement {refinement / batches:.4f}")


holdout = sample_series(256, np.random.default_rng(99))
print(f"order two VAR, {CHANNELS} channels, innovation {INNOVATION}, "
      f"block {BLOCK}")
print(f"tolerance {TOLERANCE} absolute, so a drafter that knew the chain "
      f"exactly would accept {ceiling():.3f} of {BLOCK}")
print()

layer = build()
before = measure(layer, holdout)
print(f"   before training   mean only {before[0]:.3f}   refined {before[1]:.3f}"
      f"   predicted sigma {before[2]:.4f}")
print("   training:")
train(layer, epochs=120, batches=16, learning_rate=0.02)
after = measure(layer, holdout)
print(f"   after training    mean only {after[0]:.3f}   refined {after[1]:.3f}"
      f"   predicted sigma {after[2]:.4f}  (true {INNOVATION})")
print()
print(f"   accepted length {before[1]:.3f} -> {after[1]:.3f} refined, "
      f"{before[0]:.3f} -> {after[0]:.3f} mean only, ceiling {ceiling():.3f}")
