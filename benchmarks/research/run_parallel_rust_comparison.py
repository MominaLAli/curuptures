"""Parallel optimized-Rust PELT vs batched FusedPELT.

This benchmark reuses the exact heterogeneous batch generator and
GPU timing protocol from run_rust_pelt_comparison.py.

CPU baseline:
    optimized Rust `pelt`, parallelized across independent series
    using 1, 2, 4, and 8 persistent Python worker processes.

GPU baseline:
    one batched FusedPELT call.

The ProcessPool startup cost is excluded from the measured region;
the measured time includes dispatch of each batch to already-running
workers and collection of the results.

Produces:
    results/parallel_rust_batch_crossover.csv
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pelt import predict as rust_predict

from run_rust_pelt_comparison import (
    JUMP,
    MIN_SIZE,
    PEN,
    benchmark_gpu,
    make_batch,
)


RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

BATCH_SIZES = [
    8,
    16,
    32,
    64,
    128,
]

WORKER_COUNTS = [
    1,
    2,
    4,
    8,
]

PARALLEL_REPEATS = 21
PARALLEL_WARMUP = 3


def predict_one(signal):
    """Run one optimized Rust PELT prediction."""

    return rust_predict(
        signal,
        penalty=PEN,
        segment_cost_function="l2",
        jump=JUMP,
        minimum_segment_length=MIN_SIZE,
    ).tolist()


def benchmark_parallel_rust(
    signals,
    workers,
):
    """Median batch runtime using persistent CPU workers."""

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
    ) as pool:

        # Force all workers to start before timing.
        list(
            pool.map(
                predict_one,
                signals,
                chunksize=1,
            )
        )

        # Additional warm-up passes.
        for _ in range(
            PARALLEL_WARMUP - 1
        ):
            list(
                pool.map(
                    predict_one,
                    signals,
                    chunksize=1,
                )
            )

        times = []
        result = None

        for _ in range(
            PARALLEL_REPEATS
        ):
            start = (
                time.perf_counter_ns()
            )

            result = list(
                pool.map(
                    predict_one,
                    signals,
                    chunksize=1,
                )
            )

            elapsed = (
                time.perf_counter_ns()
                - start
            )

            times.append(elapsed)

    return (
        result,
        float(np.median(times))
        / 1e9,
    )


def write_csv(rows):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        RESULTS_DIR
        / "parallel_rust_batch_crossover.csv"
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

    # Avoid nested CPU threading if any dependency enables it.
    os.environ.setdefault(
        "OMP_NUM_THREADS",
        "1",
    )

    print("=" * 100)
    print(
        "PARALLEL OPTIMIZED RUST "
        "VS BATCHED FUSEDPELT"
    )
    print("=" * 100)

    print(
        f"penalty={PEN}, "
        f"jump={JUMP}, "
        f"min_size={MIN_SIZE}"
    )

    print(
        f"parallel repeats="
        f"{PARALLEL_REPEATS}, "
        f"warmup={PARALLEL_WARMUP}"
    )

    print(
        "SLURM_CPUS_PER_TASK=",
        os.environ.get(
            "SLURM_CPUS_PER_TASK",
            "not set",
        ),
    )

    rows = []

    for batch in BATCH_SIZES:

        print()
        print("-" * 100)
        print(
            f"batch={batch}, n=2000"
        )
        print("-" * 100)

        signals = make_batch(
            batch=batch,
            n=2000,
            seed=123,
        )

        cpu_results = {}
        cpu_times = {}

        # Run all CPU measurements before
        # measuring the GPU for this batch.
        for workers in WORKER_COUNTS:

            result, runtime = (
                benchmark_parallel_rust(
                    signals,
                    workers,
                )
            )

            cpu_results[workers] = (
                result
            )

            cpu_times[workers] = (
                runtime
            )

            print(
                f"Rust-{workers}: "
                f"{runtime * 1000:.4f} ms"
            )

        gpu_result, gpu_time = (
            benchmark_gpu(signals)
        )

        print(
            f"GPU:    "
            f"{gpu_time * 1000:.4f} ms"
        )

        best_workers = min(
            WORKER_COUNTS,
            key=lambda w: cpu_times[w],
        )

        best_cpu_time = (
            cpu_times[best_workers]
        )

        ratio = (
            best_cpu_time
            / gpu_time
        )

        exact_by_workers = {
            workers: (
                cpu_results[workers]
                == gpu_result
            )
            for workers
            in WORKER_COUNTS
        }

        exact = all(
            exact_by_workers.values()
        )

        winner = (
            "GPU"
            if ratio > 1.0
            else "CPU"
        )

        print(
            f"Best CPU: "
            f"{best_cpu_time * 1000:.4f} ms "
            f"({best_workers} workers)"
        )

        print(
            f"CPU/GPU: {ratio:.3f}x"
        )

        print(
            f"winner: {winner}"
        )

        print(
            f"exact: {exact}"
        )

        row = {
            "batch": batch,
            "n": 2000,
            "rust_1_worker_ms": (
                cpu_times[1] * 1000
            ),
            "rust_2_workers_ms": (
                cpu_times[2] * 1000
            ),
            "rust_4_workers_ms": (
                cpu_times[4] * 1000
            ),
            "rust_8_workers_ms": (
                cpu_times[8] * 1000
            ),
            "best_cpu_ms": (
                best_cpu_time * 1000
            ),
            "best_workers": (
                best_workers
            ),
            "gpu_ms": (
                gpu_time * 1000
            ),
            "cpu_over_gpu": ratio,
            "winner": winner,
            "exact": exact,
        }

        rows.append(row)

    write_csv(rows)


if __name__ == "__main__":
    main()
