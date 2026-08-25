import numpy as np
import pytest
import ruptures as rpt

from curuptures import BatchPelt


def test_batch_known_breakpoints():
    signals = np.stack([
        np.concatenate([
            np.zeros(50),
            np.ones(50) * 5,
            np.ones(50) * -2,
        ]),
        np.concatenate([
            np.ones(50) * 3,
            np.ones(50) * -4,
            np.ones(50) * 2,
        ]),
        np.concatenate([
            np.ones(50) * -1,
            np.ones(50) * 6,
            np.ones(50) * 1,
        ]),
    ])

    result = (
        BatchPelt(
            model="l2",
            min_size=2,
            jump=1,
        )
        .fit_predict(
            signals,
            pen=10,
        )
    )

    assert result == [
        [50, 100, 150],
        [50, 100, 150],
        [50, 100, 150],
    ]


def test_batch_matches_ruptures():
    rng = np.random.default_rng(42)

    signals = []

    for seed_shift in range(4):

        local_rng = np.random.default_rng(
            42 + seed_shift
        )

        signal = np.concatenate([
            local_rng.normal(
                0.0,
                0.5,
                100,
            ),
            local_rng.normal(
                4.0,
                0.5,
                100,
            ),
            local_rng.normal(
                -3.0,
                0.5,
                100,
            ),
        ])

        signals.append(signal)

    signals = np.stack(signals)

    gpu_results = (
        BatchPelt(
            model="l2",
            min_size=2,
            jump=5,
        )
        .fit_predict(
            signals,
            pen=5,
        )
    )

    cpu_results = []

    for signal in signals:

        result = (
            rpt.Pelt(
                model="l2",
                min_size=2,
                jump=5,
            )
            .fit(signal)
            .predict(pen=5)
        )

        cpu_results.append(
            result
        )

    assert gpu_results == cpu_results


def test_batch_multivariate_matches_ruptures():
    rng = np.random.default_rng(123)

    batches = []

    for _ in range(3):

        signal = np.vstack([
            rng.normal(
                loc=[0.0, 0.0],
                scale=0.3,
                size=(100, 2),
            ),
            rng.normal(
                loc=[3.0, -2.0],
                scale=0.3,
                size=(100, 2),
            ),
            rng.normal(
                loc=[-2.0, 4.0],
                scale=0.3,
                size=(100, 2),
            ),
        ])

        batches.append(signal)

    signals = np.stack(
        batches
    )

    gpu_results = (
        BatchPelt(
            model="l2",
            min_size=5,
            jump=5,
        )
        .fit_predict(
            signals,
            pen=10,
        )
    )

    cpu_results = []

    for signal in signals:

        result = (
            rpt.Pelt(
                model="l2",
                min_size=5,
                jump=5,
            )
            .fit(signal)
            .predict(pen=10)
        )

        cpu_results.append(
            result
        )

    assert gpu_results == cpu_results


def test_batch_pelt_validation():
    model = BatchPelt()

    with pytest.raises(RuntimeError):
        model.predict(pen=10)

    with pytest.raises(ValueError):
        BatchPelt(
            min_size=0
        )

    with pytest.raises(ValueError):
        BatchPelt(
            jump=0
        )

    with pytest.raises(NotImplementedError):
        BatchPelt(
            model="rbf"
        )
