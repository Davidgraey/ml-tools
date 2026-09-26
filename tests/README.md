# tests
Tests should only be maintained if they're TRULY adding value -- don't let a conding agent add shit just because.

Try to keep this a lean suite: each module keeps only the checks that guard real correctness --
backward passes against finite differences, serialization round trips, graph wiring, planted-structure recovery, 
and end-to-end learning.

| module | covers |
| --- | --- |
| `test_activations.py` | every derivative is the VJP of its activation, modReLU, sigmoid stability |
| `test_losses.py` | every loss's `backward` is the derivative of `forward` |
| `test_optimizers.py` | the gradient/update naming contract, SGD step size, a classifier learning |
| `test_layers.py` | fully connected, norms, dropout, Hartley/FFT layers, LatentStack |
| `test_spectre.py` | SPECTRE encoder/decoder gradients, causal decoding, prefill/decode, head layers |
| `test_hyena.py` | Hyena gradients, causal convolution, causality and padding |
| `test_decision_layers.py` | DecisionHead gradients, decision loss, decoding, calibration, end-to-end |
| `test_embedding.py` | rotary and sinusoid encodings, the token lookup table |
| `test_network.py` | NeuralNetwork wiring checks, fan-out gradients, a shape sweep over every layer |
| `test_serialization.py` | round trips for every layer and for whole graphs |
| `test_weight_initialization.py` | variance rules and the activation-to-rule dispatch |
| `test_clustering.py` | PLSOM/GPLSOM lattice bookkeeping, cluster metrics, centroid network |
| `test_transforms.py` | probability calibration, PCA, MCA |
| `test_supervised.py` | GradientDescent and the boosted tree model on each task |
| `test_generators.py` | reproducibility and recoverable planted structure |
| `test_encoders.py` | the audio pipeline and encoder utilities |
| `test_text_encoders.py` | TextProcessor distortions |
| `test_utilities.py` | shared helpers and distance metrics |

## Running

From the repo root, with the package installed (`pip install -e .[test]`):

    pytest

## Conventions

Datasets come from `RandomDatasetGenerator` via fixtures in `conftest.py`,
never from hand-written literals, so any failure should be reproducible from the seed.

`conftest.py` also holds the numerical helpers the suite is built on:

- `numeric_gradient` / `numeric_gradient_complex` — central differences,
  perturbing the array in place so the code under test actually sees it
- `relative_error` — floored at 1, so it degrades to an absolute check when
  both gradients are near zero
- `input_gradient_error` / `parameter_gradient_error` — compare a layer's
  backward pass against finite differences of its own forward pass
- `as_float64` — promote a layer's parameters, since float32 lacks the
  precision for central differences

Upstream gradients are drawn from a separate seed (`UPSTREAM_SEED`).
