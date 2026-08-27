# cuRuptures

**GPU-accelerated change-point detection with CuPy.**

cuRuptures provides GPU implementations of exact change-point detection
algorithms with an API inspired by
[ruptures](https://github.com/deepcharles/ruptures).

The library is designed particularly for **batched change-point detection**,
where many independent time series can be processed simultaneously on a CUDA
GPU.

> **Status:** Experimental / alpha. The API may change before the first stable
> release.

---

## Why cuRuptures?

Change-point detection algorithms often require evaluating many candidate
segments. For large collections of time series, this can become
computationally expensive.

cuRuptures uses **CuPy and CUDA** to evaluate segment costs and dynamic
programming states across many signals in parallel.

The package currently provides two complementary batched exact solvers:

- `BatchPelt` — batched PELT with per-series pruning.
- `BatchOptimalPartitioning` — unpruned exact optimal partitioning designed
  around regular GPU-parallel operations.

An important design observation behind cuRuptures is that pruning does not
automatically imply faster execution on a GPU. Dense, regular computations can
sometimes outperform irregular pruning operations even when more candidate
segments are evaluated.

---

## Features

- GPU-accelerated L2 segment cost using CuPy
- Single-series PELT
- Batched L2 cost evaluation
- Batched PELT with independent pruning for every series
- Batched exact optimal partitioning without pruning
- Univariate time series
- Multivariate time series
- NumPy input support
- CuPy GPU backend
- `fit`, `predict`, and `fit_predict` style APIs
- Correctness validation against `ruptures`
- Randomized correctness tests
- GPU scaling benchmarks

---

## Installation

cuRuptures requires an NVIDIA GPU and a working CUDA installation.

### CUDA 13

```bash
pip install "curuptures[cuda13]"
```

### CUDA 12

```bash
pip install "curuptures[cuda12]"
```

For development from a local source checkout:

```bash
cd curuptures
pip install -e ".[cuda13,test]"
```

Use `cuda12` instead of `cuda13` when appropriate for your CUDA environment.

---

## Quick start

### Single-series PELT

```python
import numpy as np

from curuptures import Pelt


signal = np.concatenate([
    np.zeros(100),
    np.ones(100) * 5,
    np.ones(100) * -3,
])

breakpoints = (
    Pelt(
        model="l2",
        min_size=2,
        jump=5,
    )
    .fit_predict(
        signal,
        pen=10,
    )
)

print(breakpoints)
```

The final signal endpoint is included in the returned breakpoint list.

---

## Batched change-point detection

The main use case for cuRuptures is processing multiple independent time
series simultaneously.

Input arrays can have shape:

```text
(n_series, n_samples)
```

for univariate signals, or:

```text
(n_series, n_samples, n_features)
```

for multivariate signals.

### Batched PELT

```python
import numpy as np

from curuptures import BatchPelt


signals = np.stack([
    np.concatenate([
        np.zeros(100),
        np.ones(100) * 5,
    ]),
    np.concatenate([
        np.zeros(100),
        np.ones(100) * -4,
    ]),
])


breakpoints = (
    BatchPelt(
        model="l2",
        min_size=2,
        jump=5,
    )
    .fit_predict(
        signals,
        pen=10,
    )
)

print(breakpoints)
```

Each element of `breakpoints` contains the detected change points for one
series.

---

## GPU optimal partitioning

cuRuptures also provides:

```python
from curuptures import BatchOptimalPartitioning
```

This solver performs exact penalized optimal partitioning without PELT
pruning.

Example:

```python
breakpoints = (
    BatchOptimalPartitioning(
        model="l2",
        min_size=2,
        jump=5,
    )
    .fit_predict(
        signals,
        pen=10,
    )
)
```

Although this algorithm evaluates more candidate segments than PELT, its
regular batched computation can map particularly well to GPU hardware.

---

## Which batched solver should I use?

### `BatchOptimalPartitioning`

Consider this solver when:

- many independent time series are processed together,
- signals are short to moderately long,
- GPU throughput is more important than minimizing candidate evaluations,
- dense regular GPU operations are advantageous.

### `BatchPelt`

Consider this solver when:

- signals are extremely long,
- pruning removes a substantial fraction of candidate locations,
- the quadratic candidate growth of optimal partitioning becomes limiting.

Both algorithms solve the same penalized change-point objective when used with
equivalent parameters.

---
## L2 cost

For a segment from sample \(s\) to \(e\), cuRuptures currently uses the L2
piecewise-constant cost:

$$
C(s,e)
=
\sum_{t=s}^{e-1}
\left\|
x_t-\bar{x}_{s:e}
\right\|_2^2
$$

Prefix sums allow segment costs to be evaluated efficiently on the GPU.

The current release supports:

```text
model="l2"
```

Additional cost functions are planned for future releases.
---

## Preliminary performance observations

Development benchmarks have shown substantial benefits from batching change
point workloads on the GPU.

Representative development measurements from several batch-size and
signal-length scaling experiments are shown below:

| Samples per series | Batch size | BatchPelt | BatchOptimalPartitioning | Optimal/Pelt |
|---:|---:|---:|---:|---:|
| 4,000 | 25 | 1.519 s | 0.110 s | 13.79x |
| 16,000 | 10 | 5.531 s | 0.454 s | 12.18x |
| 32,000 | 5 | 12.259 s | 0.903 s | 13.58x |
| 64,000 | 5 | 23.395 s | 2.083 s | 11.23x |
| 128,000 | 5 | 45.949 s | 5.831 s | 7.88x |
| 256,000 | 5 | 96.573 s | 22.145 s | 4.36x |

\* Longer-signal experiments used a smaller batch size.

These measurements are intended to illustrate algorithmic behavior rather
than provide universal performance claims. Performance depends on GPU model,
batch size, signal length, change-point structure, penalty, `jump`, CUDA
environment, and system load.

Development measurements were performed on a shared NVIDIA GPU, so formal
benchmark results should be reproduced under controlled hardware conditions
before publication-quality speedup claims are made.

A particularly interesting observation is that
`BatchOptimalPartitioning` can outperform `BatchPelt` despite evaluating more
candidate segments. As signal length becomes very large, however, its
quadratic candidate growth becomes increasingly visible and its advantage
decreases.

---

## Correctness validation

cuRuptures is tested against the established `ruptures` package.

The current test suite includes:

- known change-point examples,
- comparisons against `ruptures.Pelt`,
- multiple penalties and jump values,
- univariate signals,
- multivariate signals,
- randomized batches,
- no-change signals,
- signals containing many strong change points,
- signal lengths not divisible by `jump`,
- agreement between batched PELT and batched optimal partitioning.

Run the tests with:

```bash
pytest -q
```

The current development suite contains:

```text
32 tests
```

---

## Public API

```python
from curuptures import (
    CostL2,
    BatchCostL2,
    Pelt,
    BatchPelt,
    BatchOptimalPartitioning,
)
```

### `CostL2`

GPU L2 segment cost for one signal.

### `BatchCostL2`

Vectorized GPU L2 segment cost for batches of independent signals.

### `Pelt`

Single-series PELT implementation.

### `BatchPelt`

GPU batched PELT with independent candidate pruning for each signal.

### `BatchOptimalPartitioning`

GPU batched exact optimal partitioning without pruning.

---

## Project structure

```text
curuptures/
├── benchmarks/
├── src/
│   └── curuptures/
│       ├── __init__.py
│       ├── pelt.py
│       ├── batch_pelt.py
│       ├── batch_optimal_partitioning.py
│       └── costs/
│           ├── __init__.py
│           ├── l2.py
│           └── batch_l2.py
├── tests/
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## Roadmap

Planned work includes:

- additional change-point cost functions,
- additional segmentation algorithms,
- improved GPU memory profiling,
- broader batch-size and signal-length benchmarks,
- controlled CPU/GPU benchmarking,
- API documentation,
- continuous integration,
- PyPI releases,
- examples and tutorials.

Potential future costs include L1 and kernel-based costs.

---

## Relationship to ruptures

[ruptures](https://github.com/deepcharles/ruptures) is a mature Python library
for offline change-point detection.

cuRuptures does not aim to replace its broad algorithm collection. Instead,
the project focuses on GPU-oriented implementations and especially batched
workloads that can benefit from CUDA parallelism.

`ruptures` is also used as a correctness reference in the cuRuptures test
suite.

---

## License

cuRuptures is released under the MIT License.

See [LICENSE](LICENSE).

---
## Citation

If you use cuRuptures in research, please cite the software using the
repository's [`CITATION.cff`](CITATION.cff) metadata.

GitHub can also generate formatted citation information directly from the
repository.
