# tests

Grouped by concept, one module per area:

| module | covers |
| --- | --- |
| `test_activations.py` | activations and the VJP derivative contract |
| `test_layers.py` | fully connected, norms, dropout, the Hartley/FFT layers, operators |
| `test_losses.py` | every loss, and that `backward` is the derivative of `forward` |
| `test_blocks.py` | the FNet-style and SPECTRE mixers, plus the FFT adjoints |
| `test_optimizers.py` | SGD, the gradient/update naming contract, end-to-end descent |
| `test_clustering.py` | PLSOM, GPLSOM growth and pruning, centroid network, metrics |
| `test_transforms.py` | probability calibration, PCA, MCA |
| `test_encoders.py` | the audio pipeline and encoder utilities |
| `test_generators.py` | every dataset task, reproducibility, separability |
| `test_embedding.py` | rotary positional encoding |
| `test_network.py` | the NeuralNetwork graph container |
| `test_utilities.py` | shared helpers, distance axioms, package import health |

## Running

Configuration lives in `pyproject.toml`, so from the repo root:

    pytest

`pythonpath` is set there, so no editable install is needed.

Finite-difference gradient checks carry the `slow` marker. To skip them:

    pytest -m "not slow"

That drops roughly a quarter of the cases and finishes in a few seconds.

## Conventions

Datasets come from `RandomDatasetGenerator` via fixtures in `conftest.py`,
never from hand-written literals, so any failure is reproducible from the seed.

`conftest.py` also holds the numerical helpers the suite is built on:

- `numeric_gradient` / `numeric_gradient_complex` — central differences,
  perturbing the array in place so the code under test actually sees it
- `relative_error` — floored at 1, so it degrades to an absolute check when
  both gradients are near zero
- `input_gradient_error` / `parameter_gradient_error` — compare a layer's
  backward pass against finite differences of its own forward pass
- `as_float64` — promote a layer's parameters, since float32 lacks the
  precision for central differences

Upstream gradients are drawn from a separate seed (`UPSTREAM_SEED`). Sharing
the data seed makes the upstream vector equal to the input, which for a
normalisation layer lies in its null space — the true gradient vanishes and the
comparison proves nothing.

## Expected failures

Known defects are marked `xfail(strict=True)` with a reason rather than
omitted, so they stay visible and flip to a failure the moment they are fixed.
They currently cover: `DropoutLayer`'s train/eval scaling and its stale mask,
`LatentStack` at rank 4, nested blocks through `SGD`, the four tabular encoder
modules that cannot be imported, `norm_euclidian_distance` failing the metric
axioms, `RopeEmbedding` not following the parameterless convention, and the two
unseeded functions in `periodic_signal_gen`.
