import time

import cupy as cp
import numpy as np

from curuptures import (
    BatchOptimalPartitioning,
    BatchPelt,
)


N_SERIES = 5
N_SAMPLES = 256000
JUMP = 5
MIN_SIZE = 2
PENALTY = 10


def make_signals(
    n_series,
    n_samples,
    seed=42,
):
    signals = []

    for i in range(n_series):
        rng = np.random.default_rng(
            seed + i
        )

        q1 = n_samples // 4
        q2 = n_samples // 2
        q3 = 3 * n_samples // 4

        signal = np.concatenate([
            rng.normal(
                0.0,
                0.5,
                q1,
            ),
            rng.normal(
                4.0,
                0.5,
                q2 - q1,
            ),
            rng.normal(
                -3.0,
                0.5,
                q3 - q2,
            ),
            rng.normal(
                2.0,
                0.5,
                n_samples - q3,
            ),
        ])

        signals.append(signal)

    return np.stack(signals)


def run_pelt(signals):
    result = (
        BatchPelt(
            model="l2",
            min_size=MIN_SIZE,
            jump=JUMP,
        )
        .fit_predict(
            signals,
            pen=PENALTY,
        )
    )

    cp.cuda.Stream.null.synchronize()

    return result


def run_optimal(signals):
    result = (
        BatchOptimalPartitioning(
            model="l2",
            min_size=MIN_SIZE,
            jump=JUMP,
        )
        .fit_predict(
            signals,
            pen=PENALTY,
        )
    )

    cp.cuda.Stream.null.synchronize()

    return result


def timed_run(
    function,
    signals,
):
    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()

    result = function(
        signals
    )

    cp.cuda.Stream.null.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    return elapsed, result


def main():

    m = (
        N_SAMPLES
        // JUMP
    )

    candidates_per_series = (
        m * (m + 1)
    ) // 2

    batch_candidates = (
        candidates_per_series
        * N_SERIES
    )

    print()
    print(
        "cuRuptures 256k stress benchmark"
    )

    print(
        f"Batch size: {N_SERIES}"
    )

    print(
        f"N: {N_SAMPLES:,}"
    )

    print(
        f"jump: {JUMP}"
    )

    print(
        f"Grid points: {m:,}"
    )

    print(
        "Candidates / series:",
        f"{candidates_per_series:,}",
    )

    print(
        "Batch candidate-series evaluations:",
        f"{batch_candidates:,}",
    )

    signals = make_signals(
        n_series=N_SERIES,
        n_samples=N_SAMPLES,
    )

    # Small global CUDA warm-up only.
    warm = cp.arange(
        10000,
        dtype=cp.float64,
    )

    cp.sum(warm)

    cp.cuda.Stream.null.synchronize()

    print()
    print(
        "Running BatchPelt..."
    )

    pelt_time, pelt_result = timed_run(
        run_pelt,
        signals,
    )

    print(
        f"BatchPelt: {pelt_time:.6f} s"
    )

    print()
    print(
        "Running BatchOptimalPartitioning..."
    )

    optimal_time, optimal_result = timed_run(
        run_optimal,
        signals,
    )

    print(
        "BatchOptimalPartitioning:",
        f"{optimal_time:.6f} s",
    )

    match = (
        pelt_result
        == optimal_result
    )

    relative = (
        pelt_time
        / optimal_time
    )

    print()
    print(
        "Match:",
        match,
    )

    print(
        "Optimal/Pelt:",
        f"{relative:.3f}x",
    )

    print(
        "First Pelt result:",
        pelt_result[0],
    )

    print(
        "First Optimal result:",
        optimal_result[0],
    )


if __name__ == "__main__":
    main()
