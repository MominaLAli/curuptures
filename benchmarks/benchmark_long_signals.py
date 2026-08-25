import time

import cupy as cp
import numpy as np

from curuptures import (
    BatchOptimalPartitioning,
    BatchPelt,
)


REPEATS = 3
N_SERIES = 10


def make_signals(
    n_series,
    n_samples,
    seed=42,
):
    signals = []

    for i in range(n_series):
        rng = np.random.default_rng(seed + i)

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


def run_pelt(signals, pen):
    result = (
        BatchPelt(
            model="l2",
            min_size=2,
            jump=5,
        )
        .fit_predict(
            signals,
            pen=pen,
        )
    )

    cp.cuda.Stream.null.synchronize()

    return result


def run_optimal(signals, pen):
    result = (
        BatchOptimalPartitioning(
            model="l2",
            min_size=2,
            jump=5,
        )
        .fit_predict(
            signals,
            pen=pen,
        )
    )

    cp.cuda.Stream.null.synchronize()

    return result


def benchmark_solver(
    function,
    signals,
    pen,
):
    times = []

    # Configuration-specific warm-up.
    reference = function(
        signals,
        pen,
    )

    for _ in range(REPEATS):
        cp.cuda.Stream.null.synchronize()

        start = time.perf_counter()

        result = function(
            signals,
            pen,
        )

        cp.cuda.Stream.null.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        if result != reference:
            raise RuntimeError(
                f"{function.__name__} changed result"
            )

        times.append(elapsed)

    times = np.asarray(times)

    return (
        float(np.median(times)),
        float(np.std(times, ddof=1)),
        reference,
    )


def measure_memory(
    function,
    signals,
    pen,
):
    """
    Measure CuPy memory-pool allocation after one isolated run.

    This is a development metric, not total GPU memory usage.
    """

    pool = cp.get_default_memory_pool()

    pool.free_all_blocks()

    cp.cuda.Stream.null.synchronize()

    function(
        signals,
        pen,
    )

    cp.cuda.Stream.null.synchronize()

    total_bytes = pool.total_bytes()

    return (
        total_bytes
        / (1024 ** 2)
    )


def main():

    lengths = [
        4000,
        8000,
        16000,
        32000,
    ]

    pen = 10

    # Global warm-up.
    warm_signals = make_signals(
        n_series=3,
        n_samples=1000,
        seed=9999,
    )

    run_pelt(
        warm_signals,
        pen,
    )

    run_optimal(
        warm_signals,
        pen,
    )

    print()
    print(
        "cuRuptures long-signal GPU benchmark"
    )

    print(
        f"Batch size: {N_SERIES}"
    )

    print(
        f"Repeats: {REPEATS}"
    )

    print()

    print("=" * 112)

    print(
        f"{'N':>8}"
        f"{'Pelt med':>14}"
        f"{'Optimal med':>14}"
        f"{'Optimal/Pelt':>15}"
        f"{'Pelt std':>12}"
        f"{'Optimal std':>13}"
        f"{'Pelt MB':>12}"
        f"{'Optimal MB':>12}"
        f"{'match':>9}"
    )

    print("-" * 112)

    for n_samples in lengths:

        signals = make_signals(
            n_series=N_SERIES,
            n_samples=n_samples,
            seed=42,
        )

        (
            pelt_median,
            pelt_std,
            pelt_result,
        ) = benchmark_solver(
            run_pelt,
            signals,
            pen,
        )

        (
            optimal_median,
            optimal_std,
            optimal_result,
        ) = benchmark_solver(
            run_optimal,
            signals,
            pen,
        )

        match = (
            pelt_result
            == optimal_result
        )

        if not match:
            raise RuntimeError(
                f"Solver mismatch at N={n_samples}"
            )

        pelt_memory = measure_memory(
            run_pelt,
            signals,
            pen,
        )

        optimal_memory = measure_memory(
            run_optimal,
            signals,
            pen,
        )

        relative = (
            pelt_median
            / optimal_median
        )

        print(
            f"{n_samples:8d}"
            f"{pelt_median:14.6f}"
            f"{optimal_median:14.6f}"
            f"{relative:15.3f}x"
            f"{pelt_std:12.6f}"
            f"{optimal_std:13.6f}"
            f"{pelt_memory:12.2f}"
            f"{optimal_memory:12.2f}"
            f"{str(match):>9}"
        )

    print()
    print(
        "Optimal/Pelt = BatchPelt median / "
        "BatchOptimalPartitioning median"
    )

    print(
        "Memory columns report CuPy memory-pool allocation "
        "after an isolated run."
    )

    print(
        "They are not whole-system GPU memory measurements."
    )


if __name__ == "__main__":
    main()
