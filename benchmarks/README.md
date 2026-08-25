# cuRuptures Benchmarks

This directory contains reproducible benchmark scripts for evaluating
cuRuptures against `ruptures.Pelt` and for comparing the two batched GPU
solvers provided by cuRuptures.

## Benchmark goals

The benchmarks evaluate:

- CPU `ruptures.Pelt`
- GPU `BatchPelt`
- GPU `BatchOptimalPartitioning`
- batch-size scaling
- signal-length scaling
- very long signal behavior
- agreement between GPU solvers and the CPU reference

All GPU benchmark scripts explicitly synchronize the CUDA stream around
measured regions where appropriate.

## Environment

Install cuRuptures with the CUDA version matching your system.

For CUDA 13:

```bash
pip install -e ".[cuda13,test]"
