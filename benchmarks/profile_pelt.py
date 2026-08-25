import cProfile
import io
import pstats

import numpy as np

from curuptures import Pelt


def make_signal(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    q1 = n // 4
    q2 = n // 2
    q3 = 3 * n // 4

    return np.concatenate([
        rng.normal(0.0, 0.5, q1),
        rng.normal(4.0, 0.5, q2 - q1),
        rng.normal(-3.0, 0.5, q3 - q2),
        rng.normal(2.0, 0.5, n - q3),
    ])


def run():
    signal = make_signal()

    model = Pelt(
        model="l2",
        min_size=2,
        jump=5,
    )

    bkps = model.fit_predict(
        signal,
        pen=10,
    )

    print("Breakpoints:", bkps)


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

stats.print_stats(30)

print(stream.getvalue())
