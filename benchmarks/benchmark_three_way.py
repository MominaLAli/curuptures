import time

import cupy as cp
import numpy as np
import ruptures as rpt

from curuptures import (
    BatchOptimalPartitioning,
    BatchPelt,
)


REPEATS = 7


def make_signals(
    n_series,
    n_samples=1000,
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


def time_cpu(
    signals,
    pen,
    reference,
):
    times = []

    for _ in range(REPEATS):

        start = time.perf_counter()

        result = run_cpu(
            signals,
            pen,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        if result != reference:
            raise RuntimeError(
                "CPU result changed between runs"
            )

        times.append(elapsed)

    return np.asarray(times)


def time_gpu(
    function,
    signals,
    pen,
    reference,
):
    times = []

    # Unmeasured run for this exact configuration.
    result = function(
        signals,
        pen,
    )

    if result != reference:
        raise RuntimeError(
            f"{function.__name__} does not match CPU"
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
                f"{function.__name__} does not match CPU"
            )

        times.append(elapsed)

    return np.asarray(times)


def warm_up(
    n_samples,
    pen,
):
    signals = make_signals(
        n_series=10,
        n_samples=n_samples,
        seed=9999,
    )

    run_batch_pelt(
        signals,
        pen,
    )

    run_batch_optimal(
        signals,
        pen,
    )

    cp.cuda.Stream.null.synchronize()


def summarize(times):
    return {
        "median": float(
            np.median(times)
        ),
        "minimum": float(
            np.min(times)
        ),
        "std": float(
            np.std(
                times,
                ddof=1,
            )
        ),
    }


def benchmark_one(
    n_series,
    n_samples,
    pen,
):
    signals = make_signals(
        n_series=n_series,
        n_samples=n_samples,
        seed=42,
    )

    # -----------------------------------------
    # CPU reference
    # -----------------------------------------

    reference = run_cpu(
        signals,
        pen,
    )

    # -----------------------------------------
    # Repeated timings
    # -----------------------------------------

    cpu_times = time_cpu(
        signals,
        pen,
        reference,
    )

    pelt_times = time_gpu(
        run_batch_pelt,
        signals,
        pen,
        reference,
    )

    optimal_times = time_gpu(
        run_batch_optimal,
        signals,
        pen,
        reference,
    )

    cpu = summarize(cpu_times)
    pelt = summarize(pelt_times)
    optimal = summarize(optimal_times)

    return {
        "cpu": cpu,
        "pelt": pelt,
        "optimal": optimal,
        "pelt_speedup": (
            cpu["median"]
            / pelt["median"]
        ),
        "optimal_speedup": (
            cpu["median"]
            / optimal["median"]
        ),
        "optimal_vs_pelt": (
            pelt["median"]
            / optimal["median"]
        ),
        "match": True,
    }


def main():

    n_samples = 1000
    pen = 10

    batch_sizes = [
        5,
        10,
        25,
        50,
        100,
    ]

    print()
    print(
        "cuRuptures three-way benchmark"
    )

    print(
        f"Repeats per configuration: {REPEATS}"
    )

    print()

    print(
        "Warming up both GPU solvers..."
    )

    warm_up(
        n_samples=n_samples,
        pen=pen,
    )

    print(
        "Warm-up complete.\n"
    )

    print("=" * 124)

    print(
        f"{'series':>7}"
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

    print("-" * 124)

    for n_series in batch_sizes:

        result = benchmark_one(
            n_series=n_series,
            n_samples=n_samples,
            pen=pen,
        )

        print(
            f"{n_series:7d}"
            f"{result['cpu']['median']:12.6f}"
            f"{result['pelt']['median']:12.6f}"
            f"{result['optimal']['median']:14.6f}"
            f"{result['pelt_speedup']:12.3f}x"
            f"{result['optimal_speedup']:14.3f}x"
            f"{result['optimal_vs_pelt']:15.3f}x"
            f"{result['pelt']['std']:12.6f}"
            f"{result['optimal']['std']:13.6f}"
            f"{str(result['match']):>9}"
        )

    print()

    print(
        "Pelt xCPU = CPU median / BatchPelt median"
    )

    print(
        "Optimal xCPU = CPU median / "
        "BatchOptimalPartitioning median"
    )

    print(
        "Optimal/Pelt = BatchPelt median / "
        "BatchOptimalPartitioning median"
    )

    print(
        "All GPU timings include fit(), prediction, "
        "and synchronization."
    )

    print(
        "CUDA cold-start compilation is excluded."
    )


if __name__ == "__main__":
    main()
