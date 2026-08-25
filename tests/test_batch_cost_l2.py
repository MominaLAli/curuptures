import cupy as cp
import numpy as np
import ruptures as rpt

from curuptures import BatchCostL2


def test_batch_output_shape():
    rng = np.random.default_rng(42)

    signals = rng.normal(
        size=(16, 1000)
    )

    cost = BatchCostL2().fit(
        signals
    )

    result = cost.error_many(
        starts=[0, 100, 200, 300],
        end=800,
    )

    assert result.shape == (
        16,
        4,
    )

    assert isinstance(
        result,
        cp.ndarray,
    )


def test_batch_matches_ruptures():
    rng = np.random.default_rng(123)

    signals = rng.normal(
        size=(16, 1000)
    )

    starts = [
        0,
        50,
        100,
        250,
        500,
    ]

    end = 900

    gpu_cost = (
        BatchCostL2()
        .fit(signals)
        .error_many(
            starts=starts,
            end=end,
        )
    )

    gpu_cost = cp.asnumpy(
        gpu_cost
    )

    for series_index in range(
        signals.shape[0]
    ):

        cpu = (
            rpt.costs.CostL2()
            .fit(
                signals[
                    series_index
                ]
            )
        )

        for candidate_index, start in enumerate(
            starts
        ):

            expected = cpu.error(
                start,
                end,
            )

            actual = gpu_cost[
                series_index,
                candidate_index,
            ]

            assert np.isclose(
                actual,
                expected,
                rtol=1e-10,
                atol=1e-10,
            )


def test_batch_multivariate_matches_ruptures():
    rng = np.random.default_rng(456)

    signals = rng.normal(
        size=(8, 500, 4)
    )

    starts = [
        0,
        50,
        100,
        200,
    ]

    end = 450

    gpu_cost = (
        BatchCostL2()
        .fit(signals)
        .error_many(
            starts=starts,
            end=end,
        )
    )

    gpu_cost = cp.asnumpy(
        gpu_cost
    )

    for series_index in range(
        signals.shape[0]
    ):

        cpu = (
            rpt.costs.CostL2()
            .fit(
                signals[
                    series_index
                ]
            )
        )

        for candidate_index, start in enumerate(
            starts
        ):

            expected = cpu.error(
                start,
                end,
            )

            actual = gpu_cost[
                series_index,
                candidate_index,
            ]

            assert np.isclose(
                actual,
                expected,
                rtol=1e-10,
                atol=1e-10,
            )
