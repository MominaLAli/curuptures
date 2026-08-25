import time

import cupy as cp
import numpy as np
import ruptures as rpt

from curuptures import BatchPelt


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

        signals.append(
            signal
        )

    return np.stack(
        signals
    )


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
            .predict(
                pen=pen
            )
        )

        results.append(
            result
        )

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

    # ---------------- CPU ----------------

    start = time.perf_counter()

    cpu_result = cpu_batch_pelt(
        signals,
        pen,
    )

    cpu_time = (
        time.perf_counter()
        - start
    )

    # ---------------- GPU warmup ----------------

    warmup = cp.arange(
        1000,
        dtype=cp.float64,
    )

    cp.sum(warmup)

    cp.cuda.Stream.null.synchronize()

    # ---------------- GPU ----------------

    start = time.perf_counter()

    gpu_result = gpu_batch_pelt(
        signals,
        pen,
    )

    gpu_time = (
        time.perf_counter()
        - start
    )

    # ---------------- correctness ----------------

    match = (
        cpu_result
        == gpu_result
    )

    speedup = (
        cpu_time
        / gpu_time
    )

    return (
        cpu_time,
        gpu_time,
        speedup,
        match,
    )


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
        "cuRuptures BatchPelt benchmark"
    )

    print("=" * 78)

    print(
        f"{'series':>10}"
        f"{'samples':>10}"
        f"{'CPU (s)':>14}"
        f"{'GPU (s)':>14}"
        f"{'speedup':>12}"
        f"{'match':>10}"
    )

    print("-" * 78)

    for n_series in batch_sizes:

        (
            cpu_time,
            gpu_time,
            speedup,
            match,
        ) = benchmark_one(
            n_series=n_series,
            n_samples=n_samples,
            pen=pen,
        )

        print(
            f"{n_series:10d}"
            f"{n_samples:10d}"
            f"{cpu_time:14.6f}"
            f"{gpu_time:14.6f}"
            f"{speedup:12.3f}x"
            f"{str(match):>10}"
        )

    print()

    print(
        "speedup = CPU time / GPU time"
    )

    print(
        "Values > 1 mean BatchPelt is faster."
    )


if __name__ == "__main__":
    main()
