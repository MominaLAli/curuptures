import time

import cupy as cp
import numpy as np
import ruptures as rpt

from curuptures import Pelt


def make_signal(n, seed=42):
    """
    Create a piecewise-constant noisy signal
    with three known change points.
    """

    rng = np.random.default_rng(seed)

    q1 = n // 4
    q2 = n // 2
    q3 = 3 * n // 4

    signal = np.concatenate([
        rng.normal(0.0, 0.5, q1),
        rng.normal(4.0, 0.5, q2 - q1),
        rng.normal(-3.0, 0.5, q3 - q2),
        rng.normal(2.0, 0.5, n - q3),
    ])

    return signal


def time_ruptures(signal, repeats=3):

    times = []
    result = None

    for _ in range(repeats):

        start = time.perf_counter()

        result = (
            rpt.Pelt(
                model="l2",
                min_size=2,
                jump=5,
            )
            .fit(signal)
            .predict(pen=10)
        )

        end = time.perf_counter()

        times.append(end - start)

    return np.median(times), result


def time_curuptures(signal, repeats=3):

    times = []
    result = None

    # Warm up CUDA before timing.
    warmup = cp.arange(
        1000,
        dtype=cp.float64,
    )

    cp.sum(warmup)
    cp.cuda.Stream.null.synchronize()

    for _ in range(repeats):

        cp.cuda.Stream.null.synchronize()

        start = time.perf_counter()

        result = (
            Pelt(
                model="l2",
                min_size=2,
                jump=5,
            )
            .fit(signal)
            .predict(pen=10)
        )

        cp.cuda.Stream.null.synchronize()

        end = time.perf_counter()

        times.append(end - start)

    return np.median(times), result


def main():

    sizes = [
        500,
        1000,
        2500,
        5000,
    ]

    print()
    print("cuRuptures development benchmark")
    print("=" * 78)

    print(
        f"{'N':>8}"
        f"{'ruptures CPU':>18}"
        f"{'cuRuptures GPU':>20}"
        f"{'speedup':>14}"
        f"{'match':>10}"
    )

    print("-" * 78)

    for n in sizes:

        signal = make_signal(n)

        cpu_time, cpu_bkps = time_ruptures(
            signal
        )

        gpu_time, gpu_bkps = time_curuptures(
            signal
        )

        speedup = cpu_time / gpu_time

        match = cpu_bkps == gpu_bkps

        print(
            f"{n:8d}"
            f"{cpu_time:18.6f}"
            f"{gpu_time:20.6f}"
            f"{speedup:14.3f}x"
            f"{str(match):>10}"
        )

    print()
    print(
        "speedup = CPU time / GPU time"
    )

    print(
        "Values > 1 mean cuRuptures is faster."
    )

    print(
        "Values < 1 mean ruptures is faster."
    )


if __name__ == "__main__":
    main()
