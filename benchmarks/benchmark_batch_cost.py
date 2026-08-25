import time

import cupy as cp
import numpy as np
import ruptures as rpt

from curuptures import BatchCostL2


def make_signals(
    n_series,
    n_samples,
    seed=42,
):
    rng = np.random.default_rng(seed)

    return rng.normal(
        size=(n_series, n_samples)
    )


def cpu_batch_cost(
    signals,
    starts,
    end,
):
    """
    Compute all requested costs using ruptures on CPU.
    """

    n_series = signals.shape[0]

    result = np.empty(
        (
            n_series,
            len(starts),
        ),
        dtype=np.float64,
    )

    for i in range(n_series):

        cost = (
            rpt.costs.CostL2()
            .fit(signals[i])
        )

        for j, start in enumerate(starts):

            result[i, j] = cost.error(
                start,
                end,
            )

    return result


def gpu_batch_cost(
    signals,
    starts,
    end,
):
    """
    Compute all requested costs using one batched GPU operation.
    """

    model = BatchCostL2()

    model.fit(signals)

    result = model.error_many(
        starts=starts,
        end=end,
    )

    # Synchronize so timing includes actual GPU work.
    cp.cuda.Stream.null.synchronize()

    return cp.asnumpy(result)


def benchmark_one(
    n_series,
    n_samples,
    starts,
    end,
):
    # --------------------------------------------------
    # Generate exactly the same input for CPU and GPU.
    # --------------------------------------------------

    signals = make_signals(
        n_series=n_series,
        n_samples=n_samples,
        seed=42,
    )

    # --------------------------------------------------
    # CPU
    # --------------------------------------------------

    start_time = time.perf_counter()

    cpu_result = cpu_batch_cost(
        signals,
        starts,
        end,
    )

    cpu_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------
    # GPU
    # --------------------------------------------------

    # CUDA warm-up.
    warmup = cp.arange(
        1000,
        dtype=cp.float64,
    )

    cp.sum(warmup)

    cp.cuda.Stream.null.synchronize()

    start_time = time.perf_counter()

    gpu_result = gpu_batch_cost(
        signals,
        starts,
        end,
    )

    gpu_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------
    # Correctness
    # --------------------------------------------------

    match = np.allclose(
        cpu_result,
        gpu_result,
        rtol=1e-10,
        atol=1e-10,
    )

    max_error = np.max(
        np.abs(
            cpu_result
            - gpu_result
        )
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
        max_error,
    )


def main():

    n_samples = 5000

    # 50 candidate segment starts.
    starts = list(
        range(
            0,
            2500,
            50,
        )
    )

    end = 4500

    batch_sizes = [
        10,
        50,
        100,
        250,
        500,
        1000,
    ]

    print()
    print(
        "cuRuptures batched L2 benchmark"
    )

    print("=" * 96)

    print(
        f"{'series':>10}"
        f"{'samples':>10}"
        f"{'candidates':>12}"
        f"{'CPU (s)':>14}"
        f"{'GPU (s)':>14}"
        f"{'speedup':>12}"
        f"{'match':>10}"
        f"{'max error':>14}"
    )

    print("-" * 96)

    for n_series in batch_sizes:

        (
            cpu_time,
            gpu_time,
            speedup,
            match,
            max_error,
        ) = benchmark_one(
            n_series=n_series,
            n_samples=n_samples,
            starts=starts,
            end=end,
        )

        print(
            f"{n_series:10d}"
            f"{n_samples:10d}"
            f"{len(starts):12d}"
            f"{cpu_time:14.6f}"
            f"{gpu_time:14.6f}"
            f"{speedup:12.3f}x"
            f"{str(match):>10}"
            f"{max_error:14.3e}"
        )

    print()
    print(
        "speedup = CPU time / GPU time"
    )

    print(
        "Values > 1 mean BatchCostL2 is faster."
    )

    print(
        "Values < 1 mean the CPU implementation is faster."
    )


if __name__ == "__main__":
    main()
