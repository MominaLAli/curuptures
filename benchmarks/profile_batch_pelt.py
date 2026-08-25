import cProfile
import io
import pstats
import time

import cupy as cp
import numpy as np

from curuptures import BatchPelt


def make_signals(
    n_series=100,
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


def run():
    signals = make_signals()

    # CUDA warm-up
    warmup = cp.arange(
        1000,
        dtype=cp.float64,
    )

    cp.sum(warmup)
    cp.cuda.Stream.null.synchronize()

    model = BatchPelt(
        model="l2",
        min_size=2,
        jump=5,
    )

    start = time.perf_counter()

    result = model.fit_predict(
        signals,
        pen=10,
    )

    # Important: wait for GPU work before stopping timer
    cp.cuda.Stream.null.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        "Number of series:",
        len(result),
    )

    print(
        "First result:",
        result[0],
    )

    print(
        f"Wall time: {elapsed:.6f} s"
    )


profiler = cProfile.Profile()

profiler.enable()

run()

profiler.disable()


stream = io.StringIO()

stats = pstats.Stats(
    profiler,
    stream=stream,
)

stats.sort_stats(
    "cumulative"
)

stats.print_stats(35)

print()
print(stream.getvalue())
