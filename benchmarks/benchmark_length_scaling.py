import time

import cupy as cp
import numpy as np
import ruptures as rpt

from curuptures import (
    BatchOptimalPartitioning,
    BatchPelt,
)


REPEATS = 3
N_SERIES = 25


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


def run_cpu(
    signals,
    pen,
):
    results = []

    for signal in signals:
        result = (
            rpt.Pelt(
                model="l2",
                min_size=2,
                jump=5,
            )
            .fit(signal)
            .predict(pen=pen)
        )

        results.append(result)

    return results


def run_batch_pelt(
    signals,
    pen,
):
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


def run_batch_optimal(
    signals,
    pen,
):
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


def median_time(
    function,
    signals,
    pen,
    reference,
):
    times = []

    # Unmeasured configuration-specific warm-up.
    result = function(
        signals,
        pen,
    )

    if result != reference:
        raise RuntimeError(
            f"{function.__name__} does not match reference"
        )

    for _ in range(REPEATS):

        if function is not run_cpu:
            cp.cuda.Stream.null.synchronize()

        start = time.perf_counter()

        result = function(
            signals,
            pen,
        )

        if function is not run_cpu:
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
    )


def warm_up():
    signals = make_signals(
        n_series=5,
        n_samples=500,
        seed=9999,
    )

    run_batch_pelt(
        signals,
        pen=10,
    )

    run_batch_optimal(
        signals,
        pen=10,
    )

    cp.cuda.Stream.null.synchronize()


def main():

    lengths = [
        250,
        500,
        1000,
        2000,
        4000,
    ]

    pen = 10

    print()
    print(
        "cuRuptures signal-length scaling benchmark"
    )

    print(
        f"Batch size: {N_SERIES}"
    )

    print(
        f"Repeats: {REPEATS}"
    )

    print()

    print(
        "Warming up GPU solvers..."
    )

    warm_up()

    print(
        "Warm-up complete.\n"
    )

    print("=" * 126)

    print(
        f"{'N':>8}"
        f"{'CPU med':>12}"
        f"{'Pelt med':>12}"
        f"{'Optimal med':>14}"
        f"{'Pelt xCPU':>12}"
        f"{'Optimal xCPU':>14}"
        f"{'Optimal/Pelt':>15}"
        f"{'Pelt std':>12}"
        f"{'Optimal std':>13}"
        f"{'match':>9}"
    )

    print("-" * 126)

    for n_samples in lengths:

        signals = make_signals(
            n_series=N_SERIES,
            n_samples=n_samples,
            seed=42,
        )

        # One CPU reference result.
        reference = run_cpu(
            signals,
            pen,
        )

        cpu_median, cpu_std = median_time(
            run_cpu,
            signals,
            pen,
            reference,
        )

        pelt_median, pelt_std = median_time(
            run_batch_pelt,
            signals,
            pen,
            reference,
        )

        optimal_median, optimal_std = median_time(
            run_batch_optimal,
            signals,
            pen,
            reference,
        )

        pelt_speedup = (
            cpu_median
            / pelt_median
        )

        optimal_speedup = (
            cpu_median
            / optimal_median
        )

        optimal_vs_pelt = (
            pelt_median
            / optimal_median
        )

        print(
            f"{n_samples:8d}"
            f"{cpu_median:12.6f}"
            f"{pelt_median:12.6f}"
            f"{optimal_median:14.6f}"
            f"{pelt_speedup:12.3f}x"
            f"{optimal_speedup:14.3f}x"
            f"{optimal_vs_pelt:15.3f}x"
            f"{pelt_std:12.6f}"
            f"{optimal_std:13.6f}"
            f"{str(True):>9}"
        )

    print()
    print(
        "Pelt xCPU = CPU ruptures.Pelt / GPU BatchPelt"
    )

    print(
        "Optimal xCPU = CPU ruptures.Pelt / "
        "GPU BatchOptimalPartitioning"
    )

    print(
        "Optimal/Pelt = GPU BatchPelt / "
        "GPU BatchOptimalPartitioning"
    )

    print(
        "All timings include fit(), prediction, and GPU synchronization."
    )


if __name__ == "__main__":
    main()
