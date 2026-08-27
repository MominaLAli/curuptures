"""Optimized Rust PELT vs FusedPELT for multivariate signals.

This corrects the competitive baseline for the multivariate experiment.
The original `multivariate_scaling.csv` was generated with Python
`ruptures.Pelt`; this benchmark instead uses the optimized Rust `pelt`
implementation.

The synthetic signal construction, dimensions, penalties, jump, and
minimum segment size match the original multivariate experiment.

Produces:
    results/rust_multivariate_scaling.csv
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cupy as cp
import numpy as np

from pelt import predict as rust_predict

from curuptures.fused_endpoint_batch_pelt import (
    FusedEndpointBatchPelt,
)


RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

N = 2000
JUMP = 5
MIN_SIZE = 2

DIMENSIONS = [
    1,
    2,
    4,
    8,
    16,
    32,
]

RUST_REPEATS = 21
RUST_WARMUP = 3

GPU_REPEATS = 15
GPU_WARMUP = 3


def make_multivariate_signal(
    n_features,
    seed=123,
):
    """Match run_fused_pelt_suite.py exactly."""

    rng = np.random.default_rng(
        seed
    )

    x = rng.normal(
        0.0,
        0.5,
        size=(
            N,
            n_features,
        ),
    )

    scales = np.linspace(
        0.75,
        1.25,
        n_features,
        dtype=np.float64,
    )

    x[300:850, :] += (
        2.5
        * scales[None, :]
    )

    x[850:1450, :] -= (
        3.0
        * scales[None, :]
    )

    x[1450:, :] += (
        4.0
        * scales[None, :]
    )

    return x


def benchmark_rust(
    signal,
    penalty,
):

    for _ in range(RUST_WARMUP):
        rust_predict(
            signal,
            penalty=penalty,
            segment_cost_function="l2",
            jump=JUMP,
            minimum_segment_length=MIN_SIZE,
        )

    times = []
    result = None

    for _ in range(RUST_REPEATS):

        start = time.perf_counter_ns()

        result = rust_predict(
            signal,
            penalty=penalty,
            segment_cost_function="l2",
            jump=JUMP,
            minimum_segment_length=MIN_SIZE,
        )

        elapsed = (
            time.perf_counter_ns()
            - start
        )

        times.append(elapsed)

    return (
        result.tolist(),
        float(np.median(times))
        / 1e6,
    )


def benchmark_gpu(
    signal,
    penalty,
):

    batch = signal[None, :, :]

    for _ in range(GPU_WARMUP):

        model = FusedEndpointBatchPelt(
            model="l2",
            min_size=MIN_SIZE,
            jump=JUMP,
            threads=256,
        )

        model.fit_predict(
            batch,
            pen=penalty,
        )

    cp.cuda.Device().synchronize()

    times = []
    result = None

    for _ in range(GPU_REPEATS):

        model = FusedEndpointBatchPelt(
            model="l2",
            min_size=MIN_SIZE,
            jump=JUMP,
            threads=256,
        )

        cp.cuda.Device().synchronize()

        start = time.perf_counter_ns()

        result = model.fit_predict(
            batch,
            pen=penalty,
        )

        cp.cuda.Device().synchronize()

        elapsed = (
            time.perf_counter_ns()
            - start
        )

        times.append(elapsed)

    return (
        result[0],
        float(np.median(times))
        / 1e6,
    )


def write_csv(rows):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        RESULTS_DIR
        / "rust_multivariate_scaling.csv"
    )

    with path.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote {path}")


def main():

    print("=" * 100)
    print(
        "OPTIMIZED RUST VS FUSEDPELT "
        "MULTIVARIATE SCALING"
    )
    print("=" * 100)

    print(
        f"n={N}, jump={JUMP}, "
        f"min_size={MIN_SIZE}"
    )

    rows = []

    for dimension in DIMENSIONS:

        penalty = (
            10.0
            * dimension
        )

        signal = (
            make_multivariate_signal(
                dimension
            )
        )

        rust_result, rust_ms = (
            benchmark_rust(
                signal,
                penalty,
            )
        )

        gpu_result, gpu_ms = (
            benchmark_gpu(
                signal,
                penalty,
            )
        )

        ratio = (
            rust_ms
            / gpu_ms
        )

        exact = (
            rust_result
            == gpu_result
        )

        winner = (
            "GPU"
            if ratio > 1.0
            else "Rust"
        )

        row = {
            "n": N,
            "features": dimension,
            "penalty": penalty,
            "jump": JUMP,
            "min_size": MIN_SIZE,
            "rust_ms": rust_ms,
            "gpu_ms": gpu_ms,
            "rust_over_gpu": ratio,
            "winner": winner,
            "breakpoints": len(
                rust_result
            ),
            "exact": exact,
        }

        rows.append(row)

        print(
            f"d={dimension:2d}  "
            f"pen={penalty:6.1f}  "
            f"Rust={rust_ms:8.4f} ms  "
            f"GPU={gpu_ms:8.4f} ms  "
            f"R/G={ratio:6.3f}x  "
            f"winner={winner:4s}  "
            f"bkpts={len(rust_result)}  "
            f"exact={exact}"
        )

    write_csv(rows)


if __name__ == "__main__":
    main()
