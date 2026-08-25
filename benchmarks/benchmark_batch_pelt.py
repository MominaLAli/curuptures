import time

import cupy as cp
import numpy as np
import ruptures as rpt

from curuptures import BatchPelt


REPEATS = 7


def make_signals(
    n_series,
    n_samples=1000,
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


def cpu_batch_pelt(
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


def gpu_batch_pelt(
    signals,
    pen,
):
    model = BatchPelt(
        model="l2",
        min_size=2,
        jump=5,
    )

    result = model.fit_predict(
        signals,
        pen=pen,
    )

    cp.cuda.Stream.null.synchronize()

    return result


def warm_up_gpu(
    n_samples,
    pen,
):
    """
    Full algorithm warm-up so CUDA initialization and
    kernel compilation are excluded from steady-state timings.
    """

    signals = make_signals(
        n_series=10,
        n_samples=n_samples,
        seed=999,
    )

    gpu_batch_pelt(
        signals,
        pen,
    )

    cp.cuda.Stream.null.synchronize()


def benchmark_one(
    n_series,
    n_samples,
    pen,
    repeats=REPEATS,
):
    signals = make_signals(
        n_series=n_series,
        n_samples=n_samples,
        seed=42,
    )

    # ---------------------------------------------
    # Reference result
    # ---------------------------------------------

    cpu_reference = cpu_batch_pelt(
        signals,
        pen,
    )

    # ---------------------------------------------
    # One unmeasured GPU run at this exact batch size
    # to stabilize allocations/caches.
    # ---------------------------------------------

    gpu_reference = gpu_batch_pelt(
        signals,
        pen,
    )

    if gpu_reference != cpu_reference:
        raise RuntimeError(
            f"CPU/GPU mismatch for batch size {n_series}"
        )

    # ---------------------------------------------
    # CPU repeated timing
    # ---------------------------------------------

    cpu_times = []

    for _ in range(repeats):

        start = time.perf_counter()

        result = cpu_batch_pelt(
            signals,
            pen,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        if result != cpu_reference:
            raise RuntimeError(
                "CPU result changed between runs"
            )

        cpu_times.append(
            elapsed
        )

    # ---------------------------------------------
    # GPU repeated timing
    # ---------------------------------------------

    gpu_times = []

    for _ in range(repeats):

        cp.cuda.Stream.null.synchronize()

        start = time.perf_counter()

        result = gpu_batch_pelt(
            signals,
            pen,
        )

        cp.cuda.Stream.null.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        if result != cpu_reference:
            raise RuntimeError(
                "GPU result does not match CPU"
            )

        gpu_times.append(
            elapsed
        )

    cpu_times = np.asarray(
        cpu_times
    )

    gpu_times = np.asarray(
        gpu_times
    )

    cpu_median = float(
        np.median(cpu_times)
    )

    gpu_median = float(
        np.median(gpu_times)
    )

    cpu_min = float(
        np.min(cpu_times)
    )

    gpu_min = float(
        np.min(gpu_times)
    )

    gpu_std = float(
        np.std(
            gpu_times,
            ddof=1,
        )
    )

    speedup = (
        cpu_median
        / gpu_median
    )

    return {
        "cpu_median": cpu_median,
        "gpu_median": gpu_median,
        "cpu_min": cpu_min,
        "gpu_min": gpu_min,
        "gpu_std": gpu_std,
        "speedup": speedup,
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
        "cuRuptures BatchPelt repeated benchmark"
    )

    print(
        f"Repeats per configuration: {REPEATS}"
    )

    print()

    # ---------------------------------------------
    # Global full-algorithm GPU warm-up
    # ---------------------------------------------

    print(
        "Warming up CUDA/cuRuptures..."
    )

    warm_up_gpu(
        n_samples=n_samples,
        pen=pen,
    )

    print(
        "Warm-up complete.\n"
    )

    print("=" * 110)

    print(
        f"{'series':>8}"
        f"{'samples':>10}"
        f"{'CPU median':>14}"
        f"{'GPU median':>14}"
        f"{'GPU std':>12}"
        f"{'CPU min':>12}"
        f"{'GPU min':>12}"
        f"{'speedup':>12}"
        f"{'match':>9}"
    )

    print("-" * 110)

    for n_series in batch_sizes:

        result = benchmark_one(
            n_series=n_series,
            n_samples=n_samples,
            pen=pen,
        )

        print(
            f"{n_series:8d}"
            f"{n_samples:10d}"
            f"{result['cpu_median']:14.6f}"
            f"{result['gpu_median']:14.6f}"
            f"{result['gpu_std']:12.6f}"
            f"{result['cpu_min']:12.6f}"
            f"{result['gpu_min']:12.6f}"
            f"{result['speedup']:12.3f}x"
            f"{str(result['match']):>9}"
        )

    print()

    print(
        "speedup = median CPU time / median GPU time"
    )

    print(
        "Cold-start CUDA compilation is excluded."
    )

    print(
        "GPU timings include fit(), prediction, and synchronization."
    )


if __name__ == "__main__":
    main()
