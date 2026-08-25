import numpy as np
import pytest
import ruptures as rpt

from curuptures import Pelt


def test_known_breakpoints():
    signal = np.concatenate([
        np.zeros(50),
        np.ones(50) * 5,
        np.ones(50) * -2,
    ])

    bkps = (
        Pelt(
            model="l2",
            min_size=2,
            jump=1,
        )
        .fit(signal)
        .predict(pen=10)
    )

    assert bkps == [50, 100, 150]


def test_matches_ruptures():
    rng = np.random.default_rng(42)

    signal = np.concatenate([
        rng.normal(0.0, 0.5, 100),
        rng.normal(4.0, 0.5, 100),
        rng.normal(-3.0, 0.5, 100),
        rng.normal(2.0, 0.5, 100),
    ])

    settings = [
        {
            "min_size": 2,
            "jump": 1,
            "pen": 5,
        },
        {
            "min_size": 2,
            "jump": 5,
            "pen": 5,
        },
        {
            "min_size": 10,
            "jump": 5,
            "pen": 10,
        },
        {
            "min_size": 20,
            "jump": 10,
            "pen": 20,
        },
    ]

    for config in settings:

        cpu = (
            rpt.Pelt(
                model="l2",
                min_size=config["min_size"],
                jump=config["jump"],
            )
            .fit(signal)
            .predict(
                pen=config["pen"]
            )
        )

        gpu = (
            Pelt(
                model="l2",
                min_size=config["min_size"],
                jump=config["jump"],
            )
            .fit(signal)
            .predict(
                pen=config["pen"]
            )
        )

        assert gpu == cpu


def test_multivariate_matches_ruptures():
    rng = np.random.default_rng(123)

    signal = np.vstack([
        rng.normal(
            loc=[0.0, 0.0, 0.0],
            scale=0.3,
            size=(100, 3),
        ),
        rng.normal(
            loc=[3.0, -2.0, 1.0],
            scale=0.3,
            size=(100, 3),
        ),
        rng.normal(
            loc=[-2.0, 4.0, -1.0],
            scale=0.3,
            size=(100, 3),
        ),
    ])

    cpu = (
        rpt.Pelt(
            model="l2",
            min_size=5,
            jump=5,
        )
        .fit(signal)
        .predict(pen=10)
    )

    gpu = (
        Pelt(
            model="l2",
            min_size=5,
            jump=5,
        )
        .fit(signal)
        .predict(pen=10)
    )

    assert gpu == cpu


def test_fit_predict():
    signal = np.concatenate([
        np.zeros(40),
        np.ones(40) * 5,
    ])

    model = Pelt(
        model="l2",
        min_size=2,
        jump=1,
    )

    result = model.fit_predict(
        signal,
        pen=10,
    )

    assert result == [40, 80]


def test_predict_before_fit():
    model = Pelt()

    with pytest.raises(RuntimeError):
        model.predict(pen=10)


def test_invalid_penalty():
    signal = np.arange(
        100,
        dtype=np.float64,
    )

    model = Pelt().fit(signal)

    with pytest.raises(ValueError):
        model.predict(pen=0)

    with pytest.raises(ValueError):
        model.predict(pen=-1)


def test_invalid_model():
    with pytest.raises(NotImplementedError):
        Pelt(model="rbf")


def test_invalid_parameters():
    with pytest.raises(ValueError):
        Pelt(min_size=0)

    with pytest.raises(ValueError):
        Pelt(jump=0)
