import cupy as cp
import numpy as np
import pytest
import ruptures as rpt

from curuptures import CostL2


def test_simple_signal():
    signal = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    cost = CostL2().fit(signal)

    assert np.isclose(
        cost.error(0, 5),
        10.0,
    )

    assert np.isclose(
        cost.error(0, 3),
        2.0,
    )


def test_matches_ruptures_univariate():
    rng = np.random.default_rng(42)

    signal = rng.normal(
        size=1000
    )

    cpu = rpt.costs.CostL2().fit(signal)
    gpu = CostL2().fit(signal)

    segments = [
        (0, 100),
        (10, 250),
        (100, 500),
        (250, 900),
        (0, 1000),
    ]

    for start, end in segments:

        cpu_value = cpu.error(
            start,
            end
        )

        gpu_value = gpu.error(
            start,
            end
        )

        assert np.isclose(
            cpu_value,
            gpu_value,
            rtol=1e-10,
            atol=1e-10,
        )


def test_matches_ruptures_multivariate():
    rng = np.random.default_rng(123)

    signal = rng.normal(
        size=(1000, 8)
    )

    cpu = rpt.costs.CostL2().fit(signal)
    gpu = CostL2().fit(signal)

    segments = [
        (0, 50),
        (50, 200),
        (100, 600),
        (400, 1000),
        (0, 1000),
    ]

    for start, end in segments:

        assert np.isclose(
            cpu.error(start, end),
            gpu.error(start, end),
            rtol=1e-10,
            atol=1e-10,
        )


def test_numpy_input():
    signal = np.arange(
        100,
        dtype=np.float64
    )

    cost = CostL2().fit(signal)

    assert isinstance(
        cost.signal,
        cp.ndarray,
    )


def test_cupy_input():
    signal = cp.arange(
        100,
        dtype=cp.float64
    )

    cost = CostL2().fit(signal)

    assert isinstance(
        cost.signal,
        cp.ndarray,
    )


def test_error_many():
    signal = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    cost = CostL2().fit(signal)

    gpu_values = cost.error_many(
        starts=[0, 1, 2],
        end=5,
    )

    values = cp.asnumpy(
        gpu_values
    )

    expected = np.array(
        [10.0, 5.0, 2.0]
    )

    assert np.allclose(
        values,
        expected,
    )


def test_not_fitted():
    cost = CostL2()

    with pytest.raises(RuntimeError):
        cost.error(0, 10)


def test_invalid_empty_signal():
    cost = CostL2()

    with pytest.raises(ValueError):
        cost.fit(
            np.array([])
        )


def test_invalid_segment():
    signal = np.arange(
        20,
        dtype=np.float64
    )

    cost = CostL2().fit(signal)

    with pytest.raises(ValueError):
        cost.error(10, 5)

    with pytest.raises(ValueError):
        cost.error(-1, 5)

    with pytest.raises(ValueError):
        cost.error(0, 100)
