import time

import cupy as cp
import numpy as np

from curuptures import (
    BatchOptimalPartitioning,
    BatchPelt,
)


REPEATS = 2
N_SERIES = 5
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


def benchmark(
    function,
    signals,
):
    # Configuration-specific warm-up.
    reference = function(
        signals
    )

    times = []

    for _ in range(REPEATS):

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

        if result != reference:
            raise RuntimeError(
                f"{function.__name__} result changed"
            )

        times.append(
            elapsed
        )

    times = np.asarray(
        times,
        dtype=np.float64,
    )

    return (
        float(np.median(times)),
        float(np.min(times)),
        reference,
    )


def theoretical_candidate_count(
    n_samples,
):
    """
    Approximate number of candidate segment evaluations
    for the unpruned solver.

    With jump=JUMP, there are roughly n_samples / JUMP
    candidate endpoints.

    Total candidates are approximately:

        1 + 2 + ... + m
        = m(m+1)/2
    """

    m = n_samples // JUMP

    per_series = (
        m * (m + 1)
    ) // 2

    total_batch = (
        per_series
        * N_SERIES
    )

    return (
        m,
        per_series,
        total_batch,
    )


def main():

    lengths = [
        32000,
        64000,
        128000,
    ]

    # Small global warm-up.
    warm_signals = make_signals(
        n_series=2,
        n_samples=2000,
        seed=9999,
    )

    run_pelt(
        warm_signals
    )

    run_optimal(
        warm_signals
    )

    print()
    print(
        "cuRuptures extreme signal-length benchmark"
    )

    print(
        f"Batch size: {N_SERIES}"
    )

    print(
        f"jump: {JUMP}"
    )

    print(
        f"Repeats: {REPEATS}"
    )

    print()

    print("=" * 126)

    print(
        f"{'N':>9}"
        f"{'grid':>9}"
        f"{'candidates/series':>20}"
        f"{'batch candidates':>20}"
        f"{'Pelt med':>14}"
        f"{'Optimal med':>14}"
        f"{'Optimal/Pelt':>15}"
        f"{'match':>10}"
    )

    print("-" * 126)

    for n_samples in lengths:

        signals = make_signals(
            n_series=N_SERIES,
            n_samples=n_samples,
            seed=42,
        )

        (
            grid_size,
            candidates_per_series,
            batch_candidates,
        ) = theoretical_candidate_count(
            n_samples
        )

        (
            pelt_median,
            pelt_min,
            pelt_result,
        ) = benchmark(
            run_pelt,
            signals,
        )

        (
            optimal_median,
            optimal_min,
            optimal_result,
        ) = benchmark(
            run_optimal,
            signals,
        )

        match = (
            pelt_result
            == optimal_result
        )

        if not match:
            raise RuntimeError(
                f"Solver mismatch at N={n_samples}"
            )

        relative = (
            pelt_median
            / optimal_median
        )

        print(
            f"{n_samples:9d}"
            f"{grid_size:9d}"
            f"{candidates_per_series:20,d}"
            f"{batch_candidates:20,d}"
            f"{pelt_median:14.6f}"
            f"{optimal_median:14.6f}"
            f"{relative:15.3f}x"
            f"{str(match):>10}"
        )

    print()

    print(
        "Optimal/Pelt = BatchPelt median / "
        "BatchOptimalPartitioning median"
    )

    print()

    print(
        "Candidate counts are theoretical unpruned "
        "segment evaluations based on the jump grid."
    )

    print(
        "They are intended to show algorithmic workload growth."
    )


if __name__ == "__main__":
    main()
