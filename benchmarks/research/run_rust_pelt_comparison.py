"""
Optimized Rust PELT vs fused CUDA PELT comparisons.

Produces:
    results/rust_batch_crossover.csv
    results/rust_sequence_crossover.csv
    results/rust_comparison_metadata.json
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
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

PEN = 10.0
JUMP = 5
MIN_SIZE = 2

RUST_REPEATS = 21
GPU_REPEATS = 15
GPU_WARMUP = 3


def write_csv(filename, rows):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = RESULTS_DIR / filename

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

    print(
        f"Wrote {path}"
    )


def write_metadata():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    props = (
        cp.cuda.runtime.getDeviceProperties(
            cp.cuda.Device().id
        )
    )

    gpu_name = props["name"]

    if isinstance(
        gpu_name,
        bytes,
    ):
        gpu_name = gpu_name.decode()

    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "pelt": importlib.metadata.version(
            "pelt"
        ),
        "gpu": gpu_name,
        "cuda_runtime": int(
            cp.cuda.runtime.runtimeGetVersion()
        ),
        "penalty": PEN,
        "jump": JUMP,
        "min_size": MIN_SIZE,
        "rust_repeats": RUST_REPEATS,
        "gpu_repeats": GPU_REPEATS,
    }

    path = (
        RESULTS_DIR
        / "rust_comparison_metadata.json"
    )

    with path.open("w") as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print(
        f"Wrote {path}"
    )


def snap(value, n):

    result = int(
        round(
            value * n / JUMP
        )
        * JUMP
    )

    return min(
        max(result, JUMP),
        n - JUMP,
    )


def make_single_signal(
    n,
    seed=123,
):

    rng = np.random.default_rng(
        seed
    )

    x = rng.normal(
        0.0,
        0.5,
        n,
    )

    c1 = snap(0.15, n)
    c2 = snap(0.425, n)
    c3 = snap(0.725, n)

    x[c1:c2] += 2.5
    x[c2:c3] -= 3.0
    x[c3:] += 4.0

    return x


def make_batch(
    batch,
    n=2000,
    seed=123,
):

    rng = np.random.default_rng(
        seed
    )

    signals = np.zeros(
        (batch, n),
        dtype=np.float64,
    )

    for i in range(batch):

        kind = i % 4

        if kind == 0:

            x = rng.normal(
                0.0,
                0.4,
                n,
            )

            x[450:1350] += 3.5
            x[1350:] -= 2.5

        elif kind == 1:

            x = rng.normal(
                0.0,
                0.5,
                n,
            )

            x[300:850] += 2.5
            x[850:1450] -= 3.0
            x[1450:] += 4.0

        elif kind == 2:

            x = rng.normal(
                0.0,
                0.45,
                n,
            )

            changes = [
                250,
                500,
                750,
                1000,
                1250,
                1500,
                1750,
            ]

            levels = [
                0.0,
                2.0,
                -2.0,
                3.0,
                -1.5,
                2.5,
                -3.0,
                1.5,
            ]

            starts = [0] + changes
            ends = changes + [n]

            for start, end, level in zip(
                starts,
                ends,
                levels,
            ):
                x[start:end] += level

        else:

            x = rng.normal(
                0.0,
                0.9,
                n,
            )

            x[400:1050] += 1.2
            x[1050:1600] -= 1.0
            x[1600:] += 1.5

        signals[i] = x

    return signals


def benchmark_rust_signal(signal):

    for _ in range(3):
        rust_predict(
            signal,
            penalty=PEN,
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
            penalty=PEN,
            segment_cost_function="l2",
            jump=JUMP,
            minimum_segment_length=MIN_SIZE,
        )

        times.append(
            time.perf_counter_ns()
            - start
        )

    return (
        result.tolist(),
        float(
            np.median(times)
        ) / 1e9,
    )


def benchmark_rust_batch(signals):

    for signal in signals:
        rust_predict(
            signal,
            penalty=PEN,
            segment_cost_function="l2",
            jump=JUMP,
            minimum_segment_length=MIN_SIZE,
        )

    times = []
    result = None

    for _ in range(RUST_REPEATS):

        start = time.perf_counter_ns()

        result = [
            rust_predict(
                signal,
                penalty=PEN,
                segment_cost_function="l2",
                jump=JUMP,
                minimum_segment_length=MIN_SIZE,
            ).tolist()
            for signal in signals
        ]

        times.append(
            time.perf_counter_ns()
            - start
        )

    return (
        result,
        float(
            np.median(times)
        ) / 1e9,
    )


def benchmark_gpu(signals):

    for _ in range(GPU_WARMUP):

        model = FusedEndpointBatchPelt(
            model="l2",
            min_size=MIN_SIZE,
            jump=JUMP,
            threads=256,
        )

        model.fit_predict(
            signals,
            pen=PEN,
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
            signals,
            pen=PEN,
        )

        cp.cuda.Device().synchronize()

        times.append(
            time.perf_counter_ns()
            - start
        )

    return (
        result,
        float(
            np.median(times)
        ) / 1e9,
    )


def run_sequence():

    lengths = [
        8000,
        8500,
        9000,
        9500,
        10000,
        10500,
        11000,
        11500,
        12000,
        16000,
        24000,
        32000,
    ]

    rows = []

    print(
        "\nSingle-series sequence crossover"
    )

    for n in lengths:

        signal = make_single_signal(n)

        rust_result, rust_time = (
            benchmark_rust_signal(
                signal
            )
        )

        gpu_result, gpu_time = (
            benchmark_gpu(
                signal[None, :]
            )
        )

        gpu_result = gpu_result[0]

        ratio = (
            rust_time
            / gpu_time
        )

        row = {
            "n": n,
            "rust_ms": rust_time * 1000,
            "gpu_ms": gpu_time * 1000,
            "rust_over_gpu": ratio,
            "winner": (
                "GPU"
                if ratio > 1.0
                else "Rust"
            ),
            "exact": (
                rust_result
                == gpu_result
            ),
        }

        rows.append(row)

        print(
            f"n={n:6d}  "
            f"Rust={row['rust_ms']:.4f} ms  "
            f"GPU={row['gpu_ms']:.4f} ms  "
            f"ratio={ratio:.3f}x  "
            f"winner={row['winner']}  "
            f"exact={row['exact']}"
        )

    write_csv(
        "rust_sequence_crossover.csv",
        rows,
    )


def run_batch():

    batch_sizes = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    ]

    rows = []

    print(
        "\nBatch crossover at n=2000"
    )

    for batch in batch_sizes:

        signals = make_batch(
            batch
        )

        rust_result, rust_time = (
            benchmark_rust_batch(
                signals
            )
        )

        gpu_result, gpu_time = (
            benchmark_gpu(
                signals
            )
        )

        ratio = (
            rust_time
            / gpu_time
        )

        row = {
            "batch": batch,
            "n": 2000,
            "rust_ms": rust_time * 1000,
            "gpu_ms": gpu_time * 1000,
            "rust_over_gpu": ratio,
            "winner": (
                "GPU"
                if ratio > 1.0
                else "Rust"
            ),
            "exact": (
                rust_result
                == gpu_result
            ),
        }

        rows.append(row)

        print(
            f"batch={batch:3d}  "
            f"Rust={row['rust_ms']:.4f} ms  "
            f"GPU={row['gpu_ms']:.4f} ms  "
            f"ratio={ratio:.3f}x  "
            f"winner={row['winner']}  "
            f"exact={row['exact']}"
        )

    write_csv(
        "rust_batch_crossover.csv",
        rows,
    )


def main():

    write_metadata()
    run_sequence()
    run_batch()


if __name__ == "__main__":
    main()
