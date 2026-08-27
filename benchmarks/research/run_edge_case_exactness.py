"""Edge-case exactness checks for FusedPELT.

Compares FusedPELT against the optimized Rust `pelt`
implementation on deliberately unusual inputs.

These are robustness/exactness checks rather than performance
benchmarks.

Produces:
    results/edge_case_exactness.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import cupy as cp

from pelt import predict as rust_predict

from curuptures.fused_endpoint_batch_pelt import (
    FusedEndpointBatchPelt,
)


RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)


def run_case(
    name,
    signal,
    penalty,
    jump=5,
    min_size=2,
):
    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    # Rust input:
    # (n,) for univariate
    # (n,d) for multivariate.
    rust_input = signal

    rust_result = rust_predict(
        rust_input,
        penalty=penalty,
        segment_cost_function="l2",
        jump=jump,
        minimum_segment_length=min_size,
    ).tolist()

    # GPU expects a batch dimension.
    if signal.ndim == 1:
        gpu_input = signal[None, :]
        d = 1
    else:
        gpu_input = signal[None, :, :]
        d = signal.shape[1]

    model = FusedEndpointBatchPelt(
        model="l2",
        min_size=min_size,
        jump=jump,
        threads=256,
    )

    gpu_result = model.fit_predict(
        gpu_input,
        pen=penalty,
    )[0]

    cp.cuda.Device().synchronize()

    exact = (
        rust_result
        == gpu_result
    )

    return {
        "case": name,
        "n": signal.shape[0],
        "d": d,
        "penalty": penalty,
        "jump": jump,
        "min_size": min_size,
        "rust_breakpoints": repr(
            rust_result
        ),
        "gpu_breakpoints": repr(
            gpu_result
        ),
        "exact": exact,
    }


def make_cases():

    rng = np.random.default_rng(
        20260827
    )

    cases = []

    # 1. Completely constant signal.
    cases.append(
        (
            "constant_univariate",
            np.zeros(200),
            10.0,
            5,
            2,
        )
    )

    # 2. Random signal with an extremely large
    # penalty: should favor no internal changes.
    cases.append(
        (
            "very_large_penalty",
            rng.normal(
                0.0,
                1.0,
                300,
            ),
            1.0e8,
            5,
            2,
        )
    )

    # 3. Strong alternating levels with a small
    # penalty, producing many detectable changes.
    x = rng.normal(
        0.0,
        0.05,
        300,
    )

    for start in range(
        0,
        300,
        25,
    ):
        level = (
            2.0
            if (start // 25) % 2
            else -2.0
        )

        x[
            start : start + 25
        ] += level

    cases.append(
        (
            "many_changes_small_penalty",
            x,
            0.1,
            5,
            2,
        )
    )

    # 4. Very short valid signal.
    x = np.array(
        [
            0.0,
            0.1,
            -0.1,
            0.0,
            0.1,
            4.0,
            4.1,
            3.9,
            4.0,
            4.1,
            -2.0,
            -2.1,
            -1.9,
            -2.0,
            -2.1,
            -2.0,
            -2.0,
            -2.1,
            -1.9,
            -2.0,
        ],
        dtype=np.float64,
    )

    cases.append(
        (
            "very_short_signal",
            x,
            1.0,
            5,
            2,
        )
    )

    # 5. Multivariate constant signal.
    cases.append(
        (
            "constant_multivariate",
            np.zeros(
                (200, 4),
                dtype=np.float64,
            ),
            40.0,
            5,
            2,
        )
    )

    # 6. Multivariate signal with common
    # changepoints but different feature scales.
    x = rng.normal(
        0.0,
        0.3,
        size=(500, 4),
    )

    scales = np.array(
        [0.5, 1.0, 2.0, 4.0]
    )

    x[100:250, :] += (
        2.0 * scales
    )

    x[250:400, :] -= (
        2.5 * scales
    )

    x[400:, :] += (
        3.0 * scales
    )

    cases.append(
        (
            "multivariate_scaled_features",
            x,
            40.0,
            5,
            2,
        )
    )

    # 7. Minimum segment size larger than default.
    x = rng.normal(
        0.0,
        0.25,
        400,
    )

    x[100:220] += 3.0
    x[220:] -= 2.0

    cases.append(
        (
            "larger_min_size",
            x,
            10.0,
            20,
            20,
        )
    )

    return cases


def main():

    print("=" * 100)
    print(
        "FUSEDPELT EDGE-CASE EXACTNESS"
    )
    print("=" * 100)

    rows = []

    for (
        name,
        signal,
        penalty,
        jump,
        min_size,
    ) in make_cases():

        row = run_case(
            name=name,
            signal=signal,
            penalty=penalty,
            jump=jump,
            min_size=min_size,
        )

        rows.append(row)

        print()
        print(name)
        print("-" * len(name))
        print(
            f"n={row['n']}, "
            f"d={row['d']}, "
            f"penalty={row['penalty']}, "
            f"jump={row['jump']}, "
            f"min_size={row['min_size']}"
        )
        print(
            "Rust:",
            row["rust_breakpoints"],
        )
        print(
            "GPU: ",
            row["gpu_breakpoints"],
        )
        print(
            "exact:",
            row["exact"],
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        RESULTS_DIR
        / "edge_case_exactness.csv"
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

    passed = sum(
        bool(row["exact"])
        for row in rows
    )

    print()
    print("=" * 100)
    print(
        f"Exact cases: "
        f"{passed}/{len(rows)}"
    )
    print(
        f"Wrote {path}"
    )

    if passed != len(rows):
        raise SystemExit(
            "At least one edge case failed "
            "exact parity."
        )


if __name__ == "__main__":
    main()
