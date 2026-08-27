"""GPU memory/scalability benchmark for FusedPELT.

Measures CuPy-managed allocator footprint and end-to-end runtime
while varying:

1. Sequence length n at fixed batch size B=16
2. Batch size B at fixed sequence length n=2000

The reported memory is CuPy memory-pool growth after clearing the
pool following a warm-up run. It excludes CUDA context/driver
allocations and should therefore be interpreted as a practical
CuPy-managed working-memory footprint rather than total device
memory usage.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cupy as cp
import numpy as np

from curuptures.fused_endpoint_batch_pelt import (
    FusedEndpointBatchPelt,
)


SEQUENCE_LENGTHS = [
    2000,
    4000,
    8000,
    16000,
    32000,
    64000,
]

BATCH_SIZES = [
    1,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("benchmarks")
            / "research"
            / "results"
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--jump",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--penalty",
        type=float,
        default=10.0,
    )

    return parser.parse_args()


def make_signal(
    rng,
    batch,
    n,
):
    x = rng.normal(
        0.0,
        0.5,
        size=(batch, n),
    )

    p1 = n // 4
    p2 = n // 2
    p3 = 3 * n // 4

    x[:, p1:p2] += 2.0
    x[:, p2:p3] -= 2.5
    x[:, p3:] += 1.5

    return x


def measure_case(
    *,
    batch,
    n,
    threads,
    jump,
    min_size,
    penalty,
    seed=12345,
):
    rng = np.random.default_rng(seed)

    # Compile / warm up.
    warm = rng.normal(
        0.0,
        0.5,
        size=(2, 200),
    )

    warm_model = FusedEndpointBatchPelt(
        model="l2",
        min_size=min_size,
        jump=jump,
        threads=threads,
    )

    warm_model.fit_predict(
        warm,
        pen=penalty,
    )

    cp.cuda.Device().synchronize()

    pool = cp.get_default_memory_pool()

    # Remove cached allocations after warm-up.
    pool.free_all_blocks()

    x = make_signal(
        rng,
        batch,
        n,
    )

    model = FusedEndpointBatchPelt(
        model="l2",
        min_size=min_size,
        jump=jump,
        threads=threads,
    )

    before_total = pool.total_bytes()

    cp.cuda.Device().synchronize()

    t0 = time.perf_counter_ns()

    result = model.fit_predict(
        x,
        pen=penalty,
    )

    cp.cuda.Device().synchronize()

    t1 = time.perf_counter_ns()

    after_used = pool.used_bytes()
    after_total = pool.total_bytes()

    runtime_ms = (
        t1 - t0
    ) / 1e6

    growth_bytes = (
        after_total - before_total
    )

    return {
        "n": n,
        "batch": batch,
        "runtime_ms": runtime_ms,
        "pool_used_mib": (
            after_used / 2**20
        ),
        "pool_total_mib": (
            after_total / 2**20
        ),
        "pool_growth_mib": (
            growth_bytes / 2**20
        ),
        "bytes_per_series_sample": (
            growth_bytes
            / (batch * n)
        ),
        "series_completed": len(result),
    }


def write_csv(
    path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    print("=" * 80)
    print("FUSEDPELT GPU MEMORY SCALING")
    print("=" * 80)

    print()
    print("Sequence-length scaling")
    print("fixed batch = 16")
    print()

    sequence_rows = []

    for n in SEQUENCE_LENGTHS:
        row = measure_case(
            batch=16,
            n=n,
            threads=args.threads,
            jump=args.jump,
            min_size=args.min_size,
            penalty=args.penalty,
        )

        sequence_rows.append(row)

        print(
            f"n={n:6d}  "
            f"runtime={row['runtime_ms']:10.6f} ms  "
            f"growth={row['pool_growth_mib']:10.6f} MiB  "
            f"bytes/sample={row['bytes_per_series_sample']:8.3f}"
        )

    sequence_path = (
        args.output_dir
        / "memory_sequence_scaling.csv"
    )

    write_csv(
        sequence_path,
        sequence_rows,
    )

    print()
    print("Batch-size scaling")
    print("fixed n = 2000")
    print()

    batch_rows = []

    for batch in BATCH_SIZES:
        row = measure_case(
            batch=batch,
            n=2000,
            threads=args.threads,
            jump=args.jump,
            min_size=args.min_size,
            penalty=args.penalty,
        )

        batch_rows.append(row)

        print(
            f"B={batch:3d}  "
            f"runtime={row['runtime_ms']:10.6f} ms  "
            f"growth={row['pool_growth_mib']:10.6f} MiB  "
            f"bytes/sample={row['bytes_per_series_sample']:8.3f}"
        )

    batch_path = (
        args.output_dir
        / "memory_batch_scaling.csv"
    )

    write_csv(
        batch_path,
        batch_rows,
    )

    print()
    print(f"Wrote {sequence_path}")
    print(f"Wrote {batch_path}")


if __name__ == "__main__":
    main()
