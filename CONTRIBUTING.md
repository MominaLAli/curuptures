# Contributing to cuRuptures

Thank you for your interest in contributing to cuRuptures.

cuRuptures is an experimental GPU-accelerated change-point detection library built with CuPy. Contributions that improve correctness, performance, documentation, testing, or GPU support are welcome.

## Development setup

Clone the repository and install the package in editable mode.

For CUDA 13:

```bash
pip install -e ".[cuda13,test]"
```

For CUDA 12:

```bash
pip install -e ".[cuda12,test]"
```

Use the CUDA extra that matches your system.

## Running tests

Run the complete test suite with:

```bash
pytest -q
```

Because cuRuptures uses CuPy and CUDA, the GPU-dependent tests must be run on a system where a compatible NVIDIA GPU is available.

Changes to change-point algorithms or cost functions should include tests that verify correctness against known results or, where appropriate, against the `ruptures` reference implementation.

## Benchmarking

GPU performance measurements should be performed on an uncontended GPU whenever possible.

Please record relevant environment information, including:

- GPU model
- CUDA version
- CuPy version
- Python version
- NumPy version
- signal length
- batch size
- algorithm parameters

Benchmark scripts are available in the `benchmarks/` directory.

For example:

```bash
python benchmarks/benchmark_three_way.py
```

and:

```bash
python benchmarks/benchmark_length_scaling.py
```

Performance changes should not sacrifice numerical correctness.

CPU-versus-GPU speedups should be described carefully. In particular, batched GPU timing should not be presented as a single-series speedup when the CPU reference processes signals sequentially.

## Code changes

When modifying algorithms or cost functions:

1. Preserve existing public APIs unless a breaking change is intentional.
2. Add or update tests for behavioral changes.
3. Verify numerical agreement with the reference implementation where appropriate.
4. Avoid unnecessary CPU-GPU synchronization in performance-critical code.
5. Document any new assumptions about input shape, precision, CUDA requirements, or algorithm parameters.

## Pull requests

Before opening a pull request:

1. Run the test suite on a compatible GPU system:

   ```bash
   pytest -q
   ```

2. Check for whitespace errors:

   ```bash
   git diff --check
   ```

3. Keep changes focused on one feature or issue where practical.
4. Add or update tests for behavioral changes.
5. Update documentation when public APIs or behavior change.
6. Include benchmark evidence when claiming a performance improvement.

## Issues

Bug reports should include:

- a minimal reproducible example,
- Python version,
- NumPy version,
- CuPy version,
- CUDA version,
- GPU model,
- cuRuptures version or commit,
- the complete error message or traceback.

Feature requests are also welcome.

## Performance contributions

Performance-oriented contributions are encouraged, including work on:

- reducing CPU-GPU synchronization,
- reducing temporary GPU allocations,
- kernel fusion,
- custom CUDA or CuPy kernels,
- improved batching,
- memory profiling,
- additional GPU-friendly change-point algorithms.

Any optimization should be validated for both correctness and performance.

## Documentation contributions

Documentation improvements are welcome, including:

- installation instructions,
- examples,
- API documentation,
- benchmark explanations,
- tutorials,
- troubleshooting guidance.

Examples should use public APIs where possible.

## License

By contributing to cuRuptures, you agree that your contributions will be distributed under the MIT License used by this project.
