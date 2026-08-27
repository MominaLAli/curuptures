"""Real-data validation of FusedPELT on the Turing Change Point Dataset.

Protocol
--------
Datasets:
    well_log
    brent_spot
    quality_control_3
    run_log

Preprocessing:
    Global z-score standardization independently per dimension.

PELT settings:
    penalty = log(n)
    jump = 5
    min_size = 2

Baselines:
    Optimized Rust `pelt` implementation
    FusedPELT GPU implementation

Metrics:
    Exact Rust--GPU changepoint parity
    TCPDBench F1 / precision / recall
    TCPDBench segmentation covering

The penalty and preprocessing are annotation-independent. Human
annotations are used only for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

from pelt import predict as rust_predict

from curuptures.fused_endpoint_batch_pelt import (
    FusedEndpointBatchPelt,
)


DATASETS = [
    "well_log",
    "brent_spot",
    "quality_control_3",
    "run_log",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tcpd-root",
        type=Path,
        default=Path.home() / "TCPD",
    )

    parser.add_argument(
        "--tcpdbench-root",
        type=Path,
        default=Path.home() / "TCPDBench",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path("benchmarks")
            / "research"
            / "results"
            / "tcpd_real_data.csv"
        ),
    )

    parser.add_argument(
        "--rust-repeats",
        type=int,
        default=51,
    )

    parser.add_argument(
        "--gpu-repeats",
        type=int,
        default=31,
    )

    parser.add_argument(
        "--gpu-warmup",
        type=int,
        default=5,
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
        "--threads",
        type=int,
        default=256,
    )

    return parser.parse_args()


def clean_cps(cps, n_obs):
    """Match TCPDBench changepoint cleaning."""
    return sorted(
        {
            int(cp)
            for cp in cps
            if 1 <= int(cp) < n_obs - 1
        }
    )


def load_dataset(root, name):
    path = (
        root
        / "datasets"
        / name
        / f"{name}.json"
    )

    with path.open() as f:
        obj = json.load(f)

    n = int(obj["n_obs"])
    d = int(obj["n_dim"])

    # TCPD stores each dimension as a separate entry.
    x = np.column_stack(
        [
            np.asarray(
                series["raw"],
                dtype=np.float64,
            )
            for series in obj["series"]
        ]
    )

    if x.shape != (n, d):
        raise RuntimeError(
            f"{name}: expected {(n, d)}, "
            f"found {x.shape}"
        )

    # Global z-score, independently per dimension.
    mean = x.mean(
        axis=0,
        keepdims=True,
    )

    std = x.std(
        axis=0,
        keepdims=True,
    )

    # Safely handle any constant dimensions.
    std = np.where(
        std > 0.0,
        std,
        1.0,
    )

    x = (x - mean) / std

    return x, n, d


def median_rust_runtime(
    x,
    d,
    penalty,
    jump,
    min_size,
    repeats,
):
    rust_input = (
        x[:, 0]
        if d == 1
        else x
    )

    times = []
    result = None

    for _ in range(repeats):
        t0 = time.perf_counter_ns()

        result = rust_predict(
            rust_input,
            penalty=penalty,
            segment_cost_function="l2",
            jump=jump,
            minimum_segment_length=min_size,
        )

        t1 = time.perf_counter_ns()

        times.append(
            (t1 - t0) / 1e6
        )

    return (
        result,
        float(np.median(times)),
    )


def median_gpu_runtime(
    x,
    d,
    penalty,
    jump,
    min_size,
    threads,
    warmup,
    repeats,
):
    gpu_input = (
        x[:, 0][None, :]
        if d == 1
        else x[None, :, :]
    )

    model = FusedEndpointBatchPelt(
        model="l2",
        min_size=min_size,
        jump=jump,
        threads=threads,
    )

    result = None

    for _ in range(warmup):
        result = model.fit_predict(
            gpu_input,
            pen=penalty,
        )
        cp.cuda.Device().synchronize()

    times = []

    for _ in range(repeats):
        cp.cuda.Device().synchronize()

        t0 = time.perf_counter_ns()

        result = model.fit_predict(
            gpu_input,
            pen=penalty,
        )

        cp.cuda.Device().synchronize()

        t1 = time.perf_counter_ns()

        times.append(
            (t1 - t0) / 1e6
        )

    return (
        result[0],
        float(np.median(times)),
    )


def main():
    args = parse_args()

    metrics_path = (
        args.tcpdbench_root
        / "analysis"
        / "scripts"
    )

    sys.path.insert(
        0,
        str(metrics_path),
    )

    from metrics import (
        covering,
        f_measure,
    )

    with (
        args.tcpd_root
        / "annotations.json"
    ).open() as f:
        annotations = json.load(f)

    rows = []

    print("=" * 100)
    print(
        "TCPD REAL-DATA FUSEDPELT EVALUATION"
    )
    print("=" * 100)

    print(
        f"jump={args.jump}, "
        f"min_size={args.min_size}, "
        "penalty=log(n)"
    )

    for name in DATASETS:
        x, n, d = load_dataset(
            args.tcpd_root,
            name,
        )

        penalty = math.log(n)

        rust_result, rust_ms = (
            median_rust_runtime(
                x=x,
                d=d,
                penalty=penalty,
                jump=args.jump,
                min_size=args.min_size,
                repeats=args.rust_repeats,
            )
        )

        gpu_result, gpu_ms = (
            median_gpu_runtime(
                x=x,
                d=d,
                penalty=penalty,
                jump=args.jump,
                min_size=args.min_size,
                threads=args.threads,
                warmup=args.gpu_warmup,
                repeats=args.gpu_repeats,
            )
        )

        rust_cps = clean_cps(
            rust_result,
            n,
        )

        gpu_cps = clean_cps(
            gpu_result,
            n,
        )

        exact = (
            rust_cps == gpu_cps
        )

        f1, precision, recall = (
            f_measure(
                annotations[name],
                gpu_cps,
                return_PR=True,
            )
        )

        cover = covering(
            annotations[name],
            gpu_cps,
            n,
        )

        ratio = rust_ms / gpu_ms

        row = {
            "dataset": name,
            "n": n,
            "d": d,
            "penalty": penalty,
            "jump": args.jump,
            "min_size": args.min_size,
            "threads": args.threads,
            "predicted_cps": len(
                gpu_cps
            ),
            "changepoints": (
                json.dumps(gpu_cps)
            ),
            "rust_ms": rust_ms,
            "gpu_ms": gpu_ms,
            "rust_over_gpu": ratio,
            "exact_parity": exact,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "covering": cover,
        }

        rows.append(row)

        print()
        print(name)
        print("-" * len(name))
        print(
            f"n={n}, d={d}, "
            f"penalty={penalty:.12f}"
        )
        print(
            "changepoints:",
            gpu_cps,
        )
        print(
            "exact parity:",
            exact,
        )
        print(
            f"Rust: {rust_ms:.6f} ms"
        )
        print(
            f"GPU:  {gpu_ms:.6f} ms"
        )
        print(
            f"F1={f1:.6f}, "
            f"precision={precision:.6f}, "
            f"recall={recall:.6f}, "
            f"covering={cover:.6f}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Wrote {args.output}"
    )


if __name__ == "__main__":
    main()
