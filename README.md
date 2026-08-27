# cuRuptures ⚡

**GPU-Accelerated Change-Point Detection using NVIDIA CUDA and CuPy**

[![PyPI](https://img.shields.io/pypi/v/curuptures?label=PyPI&color=blue)](https://pypi.org/project/curuptures/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-GPU-green)](https://developer.nvidia.com/cuda-toolkit)
[![CuPy](https://img.shields.io/badge/CuPy-GPU%20Arrays-orange)](https://cupy.dev/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/MominaLAli/curuptures)](https://github.com/MominaLAli/curuptures/releases)
[![Package Checks](https://github.com/MominaLAli/curuptures/actions/workflows/package-checks.yml/badge.svg)](https://github.com/MominaLAli/curuptures/actions/workflows/package-checks.yml)

---

## 🚀 Overview

**cuRuptures** is a GPU-accelerated change-point detection library powered by
[CuPy](https://cupy.dev/) and NVIDIA CUDA.

It is designed especially for **batched change-point detection**, where many
independent time series can be processed simultaneously on a GPU.

cuRuptures provides an API inspired by
[ruptures](https://github.com/deepcharles/ruptures), while focusing on
GPU-oriented implementations and high-throughput workloads.

Designed for:

- Time-series segmentation
- Change-point detection
- Batched signal analysis
- Large-scale monitoring
- Scientific computing
- Machine-learning pipelines
- GPU-accelerated time-series analytics

---

## ✨ Features

✅ GPU acceleration with CUDA and CuPy

✅ L2 piecewise-constant segment cost

✅ Single-series PELT

✅ Batched PELT with independent per-series pruning

✅ Exact batched optimal partitioning

✅ Batched GPU segment-cost evaluation

✅ Univariate and multivariate signals

✅ NumPy input support

✅ CuPy GPU backend

✅ `ruptures`-style API

- `fit()`
- `predict()`
- `fit_predict()`

✅ Correctness validation against `ruptures`

✅ Randomized and edge-case testing

✅ Reproducible GPU benchmarks

---

## 📦 Installation

cuRuptures is available on PyPI.

### CUDA 12

```bash
pip install "curuptures[cuda12]"
```

### CUDA 13

```bash
pip install "curuptures[cuda13]"
```

Use the option matching your CUDA environment.

Verify your GPU with:

```bash
nvidia-smi
```

---

## ⚡ Quick Start

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

---

## 🚀 Batched Change-Point Detection

The main use case for cuRuptures is processing many independent signals
simultaneously.

Input arrays may have shape:

```text
(n_series, n_samples)
```

for univariate signals, or:

```text
(n_series, n_samples, n_features)
```

for multivariate signals.

### BatchPelt

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

Each element contains the detected breakpoints for one signal.

---

## ⚙️ GPU Optimal Partitioning

cuRuptures also provides:

```python
from curuptures import BatchOptimalPartitioning
```

This solver performs exact penalized optimal partitioning without PELT
pruning.

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
dense and regular batched computation can map efficiently to GPU hardware.

---

## 🧠 Which Solver Should I Use?

### `BatchOptimalPartitioning`

Recommended when:

- many independent signals are processed together,
- high GPU throughput is important,
- dense GPU operations are advantageous,
- signal lengths are moderate to large.

### `BatchPelt`

Recommended when:

- PELT pruning removes many candidate locations,
- signals are extremely long,
- quadratic candidate growth becomes limiting.

Both algorithms solve the same penalized change-point objective when used with
equivalent parameters.

---

## 📐 L2 Cost

For a segment spanning samples `s` through `e - 1`, cuRuptures currently uses
the L2 piecewise-constant cost:

<p align="center">
  <strong>
    C(s, e) = ∑<sub>t=s</sub><sup>e−1</sup>
    ‖x<sub>t</sub> − mean(x[s:e])‖<sub>2</sub><sup>2</sup>
  </strong>
</p>

where `mean(x[s:e])` denotes the mean of the samples in the segment.

Prefix sums allow segment costs to be evaluated efficiently on the GPU.

The current release supports:

```text
model="l2"
```

Additional cost functions are planned for future releases.

---

## 📊 Performance

cuRuptures is designed to exploit parallelism across both candidate segments
and independent time series.

Representative controlled benchmarks show that GPU execution becomes
increasingly advantageous as batch size and signal length increase.

For a batch of 100 signals with 1,000 samples per signal:

| Solver | Median time | Relative to CPU reference |
|---|---:|---:|
| `ruptures.Pelt` CPU loop | 3.805 s | 1.0× |
| `BatchPelt` | 0.088 s | 43.0× |
| `BatchOptimalPartitioning` | 0.045 s | 83.7× |

These values represent **batched-throughput comparisons** against running
`ruptures.Pelt` independently for each signal.

Performance depends on:

- GPU architecture
- batch size
- signal length
- penalty
- `jump`
- change-point structure
- CUDA version
- CuPy version
- GPU workload

See [`benchmarks/`](benchmarks/) for reproducible benchmark scripts and
controlled benchmark records.

---

## ✅ Correctness

cuRuptures is validated against the established
[`ruptures`](https://github.com/deepcharles/ruptures) package.

The test suite includes:

- known change-point examples,
- comparisons with `ruptures.Pelt`,
- randomized signals,
- multiple penalties,
- multiple jump values,
- univariate signals,
- multivariate signals,
- no-change signals,
- signals with multiple strong change points,
- agreement between batched GPU solvers.

Run the tests on a CUDA-enabled system:

```bash
pytest -q
```

---

## 📚 Public API

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

GPU L2 segment cost for a single signal.

### `BatchCostL2`

Vectorized GPU L2 segment cost for batches of signals.

### `Pelt`

Single-series PELT implementation.

### `BatchPelt`

GPU batched PELT with independent pruning for each signal.

### `BatchOptimalPartitioning`

GPU batched exact optimal partitioning without pruning.

---

## 🗂 Project Structure

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
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## 🛣 Roadmap

Planned development includes:

- additional change-point cost functions,
- L1 cost,
- kernel-based costs,
- additional segmentation algorithms,
- GPU kernel profiling,
- kernel fusion,
- custom CuPy/CUDA kernels,
- improved memory profiling,
- additional benchmarks,
- expanded documentation,
- tutorials and examples.

---

## 🤝 Contributing

Contributions are welcome.

See:

[CONTRIBUTING.md](CONTRIBUTING.md)

for development setup, testing, benchmarking, and pull-request guidelines.

---

## 📚 Citation

If you use cuRuptures in your research, please cite the software.

GitHub provides citation information automatically from:

[CITATION.cff](CITATION.cff)

A basic software citation is:

```bibtex
@software{curuptures2026,
  author = {Ali, Momina Liaqat},
  title = {cuRuptures: GPU-Accelerated Change-Point Detection with CuPy},
  year = {2026},
  version = {0.1.1},
  url = {https://github.com/MominaLAli/curuptures}
}
```

---

## 📄 License

cuRuptures is released under the MIT License.

See [LICENSE](LICENSE).

---

## 🔗 Links

- [PyPI](https://pypi.org/project/curuptures/)
- [GitHub](https://github.com/MominaLAli/curuptures)
- [Releases](https://github.com/MominaLAli/curuptures/releases)
- [Issues](https://github.com/MominaLAli/curuptures/issues)
- [Benchmarks](benchmarks/)
