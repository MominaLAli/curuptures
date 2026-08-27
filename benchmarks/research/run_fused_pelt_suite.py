"""
Reproducible benchmark suite for the fused GPU PELT research prototype.

Outputs CSV files under:

    benchmarks/research/results/

Sections
--------
sequence
    BatchPelt vs FusedEndpointBatchPelt across sequence lengths.

batch
    Batch scaling at fixed sequence length.

cpu
    Single-series CPU ruptures.Pelt vs fused GPU.

multivariate
    Single-series multivariate CPU vs fused GPU.

penalty
    Pruning-density sensitivity.

exactness
    Randomized exactness validation against ruptures.Pelt.

all
    Run every section.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import cupy as cp
import numpy as np
import ruptures as rpt

from curuptures import BatchPelt
from curuptures.fused_endpoint_batch_pelt import (
    FusedEndpointBatchPelt,
)


RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

JUMP = 5
MIN_SIZE = 2
PEN = 10.0


def median_time(values):
    return float(
        np.median(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def write_csv(
    filename,
    rows,
):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        RESULTS_DIR
        / filename
    )

    if not rows:
        raise ValueError(
            "No rows to write"
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

    name = props["name"]

    if isinstance(
        name,
        bytes,
    ):
        name = name.decode()

    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "ruptures": getattr(
            rpt,
            "__version__",
            "unknown",
        ),
        "gpu": name,
        "cuda_runtime": int(
            cp.cuda.runtime.runtimeGetVersion()
        ),
        "jump": JUMP,
        "min_size": MIN_SIZE,
        "default_penalty": PEN,
    }

    path = (
        RESULTS_DIR
        / "metadata.json"
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


def snap(
    fraction,
    n,
):

    result = int(
        round(
            fraction
            * n
            / JUMP
        )
        * JUMP
    )

    return min(
        max(
            result,
            JUMP,
        ),
        n - JUMP,
    )


def make_heterogeneous_signals(
    batch,
    n,
    seed=123,
):

    rng = np.random.default_rng(
        seed
    )

    signals = np.zeros(
        (
            batch,
            n,
        ),
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

            c1 = snap(
                0.225,
                n,
            )

            c2 = snap(
                0.675,
                n,
            )

            x[c1:c2] += 3.5
            x[c2:] -= 2.5

        elif kind == 1:

            x = rng.normal(
                0.0,
                0.5,
                n,
            )

            c1 = snap(
                0.15,
                n,
            )

            c2 = snap(
                0.425,
                n,
            )

            c3 = snap(
                0.725,
                n,
            )

            x[c1:c2] += 2.5
            x[c2:c3] -= 3.0
            x[c3:] += 4.0

        elif kind == 2:

            x = rng.normal(
                0.0,
                0.45,
                n,
            )

            changes = [
                snap(0.125, n),
                snap(0.250, n),
                snap(0.375, n),
                snap(0.500, n),
                snap(0.625, n),
                snap(0.750, n),
                snap(0.875, n),
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

            starts = [
                0
            ] + changes

            ends = (
                changes
                + [n]
            )

            for (
                start,
                end,
                level,
            ) in zip(
                starts,
                ends,
                levels,
            ):

                x[
                    start:end
                ] += level

        else:

            x = rng.normal(
                0.0,
                0.9,
                n,
            )

            c1 = snap(
                0.20,
                n,
            )

            c2 = snap(
                0.525,
                n,
            )

            c3 = snap(
                0.80,
                n,
            )

            x[c1:c2] += 1.2
            x[c2:c3] -= 1.0
            x[c3:] += 1.5

        signals[i] = x

    return signals


def make_single_series_signal(
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

    c1 = snap(
        0.15,
        n,
    )

    c2 = snap(
        0.425,
        n,
    )

    c3 = snap(
        0.725,
        n,
    )

    x[c1:c2] += 2.5
    x[c2:c3] -= 3.0
    x[c3:] += 4.0

    return x



def benchmark_gpu_model(
    model,
    signals,
    pen,
    repeats=7,
    warmup=2,
    end_to_end=True,
):

    if not end_to_end:
        model.fit(signals)

    for _ in range(warmup):

        if end_to_end:

            model.fit_predict(
                signals,
                pen=pen,
            )

        else:

            model.predict(
                pen
            )

    cp.cuda.Device().synchronize()

    times = []
    result = None

    for _ in range(repeats):

        cp.cuda.Device().synchronize()

        start = time.perf_counter()

        if end_to_end:

            result = model.fit_predict(
                signals,
                pen=pen,
            )

        else:

            result = model.predict(
                pen
            )

        cp.cuda.Device().synchronize()

        times.append(
            time.perf_counter()
            - start
        )

    return (
        result,
        median_time(times),
    )


def run_sequence():

    lengths = [
        500,
        1000,
        2000,
        4000,
        8000,
    ]

    batch = 16

    rows = []

    print(
        "\nSequence-length scaling"
    )

    for n in lengths:

        signals = (
            make_heterogeneous_signals(
                batch,
                n,
            )
        )

        baseline_result, baseline_time = (
            benchmark_gpu_model(
                BatchPelt(
                    model="l2",
                    min_size=MIN_SIZE,
                    jump=JUMP,
                ),
                signals,
                PEN,
            )
        )

        fused_result, fused_time = (
            benchmark_gpu_model(
                FusedEndpointBatchPelt(
                    model="l2",
                    min_size=MIN_SIZE,
                    jump=JUMP,
                    threads=256,
                ),
                signals,
                PEN,
            )
        )

        exact = (
            baseline_result
            == fused_result
        )

        row = {
            "n": n,
            "batch": batch,
            "batch_pelt_ms": (
                baseline_time
                * 1000
            ),
            "fused_ms": (
                fused_time
                * 1000
            ),
            "speedup": (
                baseline_time
                / fused_time
            ),
            "exact": exact,
        }

        rows.append(row)

        print(
            f"n={n:5d}  "
            f"BatchPelt="
            f"{row['batch_pelt_ms']:.3f} ms  "
            f"Fused="
            f"{row['fused_ms']:.3f} ms  "
            f"speedup="
            f"{row['speedup']:.2f}x  "
            f"exact={exact}"
        )

    write_csv(
        "sequence_scaling.csv",
        rows,
    )


def run_batch():

    batch_sizes = [
        1,
        4,
        8,
        16,
        32,
        64,
    ]

    n = 2000

    rows = []

    print(
        "\nBatch scaling"
    )

    for batch in batch_sizes:

        signals = (
            make_heterogeneous_signals(
                batch,
                n,
            )
        )

        baseline_result, baseline_time = (
            benchmark_gpu_model(
                BatchPelt(
                    model="l2",
                    min_size=MIN_SIZE,
                    jump=JUMP,
                ),
                signals,
                PEN,
                end_to_end=False,
            )
        )

        fused_result, fused_time = (
            benchmark_gpu_model(
                FusedEndpointBatchPelt(
                    model="l2",
                    min_size=MIN_SIZE,
                    jump=JUMP,
                    threads=256,
                ),
                signals,
                PEN,
                end_to_end=False,
            )
        )

        exact = (
            baseline_result
            == fused_result
        )

        row = {
            "batch": batch,
            "n": n,
            "batch_pelt_ms": (
                baseline_time
                * 1000
            ),
            "fused_ms": (
                fused_time
                * 1000
            ),
            "speedup": (
                baseline_time
                / fused_time
            ),
            "fused_series_per_s": (
                batch
                / fused_time
            ),
            "exact": exact,
        }

        rows.append(row)

        print(
            f"batch={batch:3d}  "
            f"speedup="
            f"{row['speedup']:.2f}x  "
            f"throughput="
            f"{row['fused_series_per_s']:.1f} "
            f"series/s  "
            f"exact={exact}"
        )

    write_csv(
        "batch_scaling.csv",
        rows,
    )


def run_cpu():

    lengths = [
        500,
        1000,
        2000,
        4000,
        8000,
    ]

    rows = []

    print(
        "\nSingle-series CPU vs GPU"
    )

    for n in lengths:

        signal = (
            make_single_series_signal(
                n
            )
        )

        cpu_times = []
        cpu_result = None

        for _ in range(5):

            start = time.perf_counter()

            cpu_result = (
                rpt.Pelt(
                    model="l2",
                    min_size=MIN_SIZE,
                    jump=JUMP,
                )
                .fit(signal)
                .predict(
                    pen=PEN
                )
            )

            cpu_times.append(
                time.perf_counter()
                - start
            )

        gpu_result, gpu_time = (
            benchmark_gpu_model(
                FusedEndpointBatchPelt(
                    model="l2",
                    min_size=MIN_SIZE,
                    jump=JUMP,
                    threads=256,
                ),
                signal[None, :],
                PEN,
                repeats=9,
                warmup=3,
            )
        )

        gpu_result = (
            gpu_result[0]
        )

        cpu_time = median_time(
            cpu_times
        )

        exact = (
            cpu_result
            == gpu_result
        )

        row = {
            "n": n,
            "cpu_ms": (
                cpu_time
                * 1000
            ),
            "gpu_ms": (
                gpu_time
                * 1000
            ),
            "speedup": (
                cpu_time
                / gpu_time
            ),
            "exact": exact,
        }

        rows.append(row)

        print(
            f"n={n:5d}  "
            f"CPU={row['cpu_ms']:.3f} ms  "
            f"GPU={row['gpu_ms']:.3f} ms  "
            f"speedup={row['speedup']:.2f}x  "
            f"exact={exact}"
        )

    write_csv(
        "single_series_cpu_gpu.csv",
        rows,
    )


def make_multivariate_signal(
    n_features,
    seed=123,
):

    rng = np.random.default_rng(
        seed
    )

    n = 2000

    x = rng.normal(
        0.0,
        0.5,
        size=(
            n,
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


def run_multivariate():

    dimensions = [
        1,
        2,
        4,
        8,
        16,
        32,
    ]

    rows = []

    print(
        "\nMultivariate scaling"
    )

    for dimension in dimensions:

        pen = (
            10.0
            * dimension
        )

        signal = (
            make_multivariate_signal(
                dimension
            )
        )

        cpu_times = []
        cpu_result = None

        for _ in range(5):

            start = time.perf_counter()

            cpu_result = (
                rpt.Pelt(
                    model="l2",
                    min_size=MIN_SIZE,
                    jump=JUMP,
                )
                .fit(signal)
                .predict(
                    pen=pen
                )
            )

            cpu_times.append(
                time.perf_counter()
                - start
            )

        cpu_time = median_time(
            cpu_times
        )

        gpu_result, gpu_time = (
            benchmark_gpu_model(
                FusedEndpointBatchPelt(
                    model="l2",
                    min_size=MIN_SIZE,
                    jump=JUMP,
                    threads=256,
                ),
                signal[
                    None,
                    :,
                    :,
                ],
                pen,
                repeats=9,
                warmup=3,
            )
        )

        gpu_result = (
            gpu_result[0]
        )

        exact = (
            cpu_result
            == gpu_result
        )

        row = {
            "features": dimension,
            "penalty": pen,
            "cpu_ms": (
                cpu_time
                * 1000
            ),
            "gpu_ms": (
                gpu_time
                * 1000
            ),
            "speedup": (
                cpu_time
                / gpu_time
            ),
            "breakpoints": len(
                cpu_result
            ),
            "exact": exact,
        }

        rows.append(row)

        print(
            f"d={dimension:2d}  "
            f"speedup="
            f"{row['speedup']:.2f}x  "
            f"bkpts="
            f"{row['breakpoints']}  "
            f"exact={exact}"
        )

    write_csv(
        "multivariate_scaling.csv",
        rows,
    )


def run_penalty():

    penalties = [
        1.0,
        2.0,
        5.0,
        10.0,
        25.0,
        50.0,
        100.0,
    ]

    signals = (
        make_heterogeneous_signals(
            16,
            2000,
        )
    )

    rows = []

    print(
        "\nPenalty / pruning sensitivity"
    )

    for pen in penalties:

        baseline = BatchPelt(
            model="l2",
            min_size=MIN_SIZE,
            jump=JUMP,
        )

        baseline_result, baseline_time = (
            benchmark_gpu_model(
                baseline,
                signals,
                pen,
                end_to_end=False,
            )
        )

        stats_model = BatchPelt(
            model="l2",
            min_size=MIN_SIZE,
            jump=JUMP,
        )

        stats_model.fit(
            signals
        )

        stats_model._segment(
            pen,
            collect_stats=True,
        )

        active_density = float(
            np.mean(
                stats_model
                .candidate_stats_[
                    "active_density"
                ]
            )
        )

        fused_result, fused_time = (
            benchmark_gpu_model(
                FusedEndpointBatchPelt(
                    model="l2",
                    min_size=MIN_SIZE,
                    jump=JUMP,
                    threads=256,
                ),
                signals,
                pen,
                end_to_end=False,
            )
        )

        exact = (
            baseline_result
            == fused_result
        )

        row = {
            "penalty": pen,
            "active_density": (
                active_density
            ),
            "batch_pelt_ms": (
                baseline_time
                * 1000
            ),
            "fused_ms": (
                fused_time
                * 1000
            ),
            "speedup": (
                baseline_time
                / fused_time
            ),
            "mean_breakpoints": float(
                np.mean(
                    [
                        len(x)
                        for x
                        in baseline_result
                    ]
                )
            ),
            "exact": exact,
        }

        rows.append(row)

        print(
            f"pen={pen:6.1f}  "
            f"density="
            f"{active_density:.4f}  "
            f"speedup="
            f"{row['speedup']:.2f}x  "
            f"exact={exact}"
        )

    write_csv(
        "penalty_sensitivity.csv",
        rows,
    )


def run_exactness():

    rng = np.random.default_rng(
        2026
    )

    n_tests = 50

    length_choices = [
        200,
        350,
        500,
        750,
        1000,
    ]

    feature_choices = [
        1,
        2,
        4,
    ]

    jump_choices = [
        1,
        2,
        5,
        10,
    ]

    min_size_choices = [
        2,
        5,
        10,
    ]

    penalty_choices = [
        1.0,
        5.0,
        10.0,
        25.0,
    ]

    rows = []

    print(
        "\nRandomized exactness"
    )

    for test_id in range(
        n_tests
    ):

        n = int(
            rng.choice(
                length_choices
            )
        )

        n_features = int(
            rng.choice(
                feature_choices
            )
        )

        jump = int(
            rng.choice(
                jump_choices
            )
        )

        min_size = int(
            rng.choice(
                min_size_choices
            )
        )

        pen = float(
            rng.choice(
                penalty_choices
            )
        )

        x = rng.normal(
            0.0,
            0.7,
            size=(
                n,
                n_features,
            ),
        )

        n_changes = int(
            rng.integers(
                0,
                5,
            )
        )

        if n_changes > 0:

            possible = np.arange(
                max(
                    min_size,
                    jump,
                ),
                n
                - max(
                    min_size,
                    jump,
                ),
                jump,
            )

            if (
                possible.size
                >= n_changes
            ):

                changes = np.sort(
                    rng.choice(
                        possible,
                        size=n_changes,
                        replace=False,
                    )
                )

                for change in changes:

                    shift = rng.normal(
                        0.0,
                        2.0,
                        size=n_features,
                    )

                    x[
                        change:,
                        :,
                    ] += shift

        cpu_result = (
            rpt.Pelt(
                model="l2",
                min_size=min_size,
                jump=jump,
            )
            .fit(x)
            .predict(
                pen=pen
            )
        )

        gpu_result = (
            FusedEndpointBatchPelt(
                model="l2",
                min_size=min_size,
                jump=jump,
                threads=256,
            )
            .fit_predict(
                x[
                    None,
                    :,
                    :,
                ],
                pen=pen,
            )[0]
        )

        exact = (
            cpu_result
            == gpu_result
        )

        row = {
            "test": test_id,
            "n": n,
            "features": n_features,
            "jump": jump,
            "min_size": min_size,
            "penalty": pen,
            "exact": exact,
        }

        rows.append(row)

        if not exact:

            write_csv(
                "randomized_exactness.csv",
                rows,
            )

            raise RuntimeError(
                "Exactness failure on "
                f"test {test_id}"
            )

    print(
        f"Passed {n_tests}/{n_tests}"
    )

    write_csv(
        "randomized_exactness.csv",
        rows,
    )


SECTIONS = {
    "sequence": run_sequence,
    "batch": run_batch,
    "cpu": run_cpu,
    "multivariate": run_multivariate,
    "penalty": run_penalty,
    "exactness": run_exactness,
}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--section",
        choices=[
            *SECTIONS.keys(),
            "all",
        ],
        default="all",
    )

    args = parser.parse_args()

    write_metadata()

    if args.section == "all":

        for function in (
            SECTIONS.values()
        ):
            function()

    else:

        SECTIONS[
            args.section
        ]()


if __name__ == "__main__":
    main()
