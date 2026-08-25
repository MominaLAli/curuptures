import numpy as np
import pytest
import ruptures as rpt

from curuptures import BatchOptimalPartitioning


def cpu_reference(
    signals,
    min_size,
    jump,
    pen,
):
    results = []

    for signal in signals:
        result = (
            rpt.Pelt(
                model="l2",
                min_size=min_size,
                jump=jump,
            )
            .fit(signal)
            .predict(pen=pen)
        )

        results.append(result)

    return results


def test_known_breakpoints():
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
    ])

    result = (
        BatchOptimalPartitioning(
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
    ]


def test_matches_ruptures():
    signals = []

    for seed in range(5):
        rng = np.random.default_rng(
            100 + seed
        )

        signal = np.concatenate([
            rng.normal(
                0.0,
                0.5,
                100,
            ),
            rng.normal(
                4.0,
                0.5,
                100,
            ),
            rng.normal(
                -3.0,
                0.5,
                117,
            ),
        ])

        signals.append(signal)

    signals = np.stack(signals)

    settings = [
        {
            "min_size": 2,
            "jump": 1,
            "pen": 5,
        },
        {
            "min_size": 2,
            "jump": 5,
            "pen": 10,
        },
        {
            "min_size": 10,
            "jump": 5,
            "pen": 20,
        },
        {
            "min_size": 10,
            "jump": 10,
            "pen": 25,
        },
    ]

    for config in settings:

        cpu = cpu_reference(
            signals,
            min_size=config["min_size"],
            jump=config["jump"],
            pen=config["pen"],
        )

        gpu = (
            BatchOptimalPartitioning(
                model="l2",
                min_size=config["min_size"],
                jump=config["jump"],
            )
            .fit_predict(
                signals,
                pen=config["pen"],
            )
        )

        assert gpu == cpu


def test_multivariate_matches_ruptures():
    rng = np.random.default_rng(999)

    signals = []

    for _ in range(4):

        signal = np.vstack([
            rng.normal(
                loc=[0.0, 0.0, 1.0],
                scale=0.3,
                size=(100, 3),
            ),
            rng.normal(
                loc=[3.0, -2.0, 2.0],
                scale=0.3,
                size=(100, 3),
            ),
            rng.normal(
                loc=[-2.0, 4.0, -1.0],
                scale=0.3,
                size=(105, 3),
            ),
        ])

        signals.append(signal)

    signals = np.stack(signals)

    cpu = cpu_reference(
        signals,
        min_size=5,
        jump=5,
        pen=10,
    )

    gpu = (
        BatchOptimalPartitioning(
            model="l2",
            min_size=5,
            jump=5,
        )
        .fit_predict(
            signals,
            pen=10,
        )
    )

    assert gpu == cpu


def test_validation():
    model = BatchOptimalPartitioning()

    with pytest.raises(RuntimeError):
        model.predict(pen=10)

    with pytest.raises(ValueError):
        BatchOptimalPartitioning(
            min_size=0
        )

    with pytest.raises(ValueError):
        BatchOptimalPartitioning(
            jump=0
        )

    with pytest.raises(NotImplementedError):
        BatchOptimalPartitioning(
            model="rbf"
        )
