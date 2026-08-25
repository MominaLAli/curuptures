import numpy as np
import ruptures as rpt

from curuptures import BatchPelt


def cpu_batch_pelt(
    signals,
    min_size,
    jump,
    pen,
):
    """
    Reference results from ruptures.Pelt.
    """

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


def make_piecewise_signal(
    rng,
    n_samples,
    n_changes,
    noise=0.4,
):
    """
    Generate one noisy piecewise-constant signal.
    """

    if n_changes == 0:
        return rng.normal(
            loc=0.0,
            scale=noise,
            size=n_samples,
        )

    # Keep change points away from the boundaries.
    possible = np.arange(
        20,
        n_samples - 20,
    )

    change_points = np.sort(
        rng.choice(
            possible,
            size=n_changes,
            replace=False,
        )
    )

    boundaries = np.concatenate([
        [0],
        change_points,
        [n_samples],
    ])

    signal = np.empty(
        n_samples,
        dtype=np.float64,
    )

    # Random segment means.
    means = rng.uniform(
        -5.0,
        5.0,
        size=len(boundaries) - 1,
    )

    for i in range(
        len(boundaries) - 1
    ):
        start = boundaries[i]
        end = boundaries[i + 1]

        signal[start:end] = rng.normal(
            loc=means[i],
            scale=noise,
            size=end - start,
        )

    return signal


def test_randomized_batches_match_ruptures():
    """
    Many randomized univariate signals.
    """

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
            "min_size": 5,
            "jump": 5,
            "pen": 15,
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

    for seed in range(5):

        rng = np.random.default_rng(
            1000 + seed
        )

        # 317 deliberately is NOT divisible by
        # several jump values.
        n_samples = 317

        signals = []

        for series_index in range(4):

            n_changes = (
                series_index % 4
            )

            signal = make_piecewise_signal(
                rng=rng,
                n_samples=n_samples,
                n_changes=n_changes,
            )

            signals.append(signal)

        signals = np.stack(
            signals
        )

        for config in settings:

            cpu = cpu_batch_pelt(
                signals=signals,
                min_size=config["min_size"],
                jump=config["jump"],
                pen=config["pen"],
            )

            gpu = (
                BatchPelt(
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


def test_no_change_high_penalty():
    """
    High penalty should favor no interior change points.
    """

    rng = np.random.default_rng(2000)

    signals = rng.normal(
        loc=2.0,
        scale=0.5,
        size=(8, 333),
    )

    cpu = cpu_batch_pelt(
        signals=signals,
        min_size=2,
        jump=5,
        pen=1_000_000,
    )

    gpu = (
        BatchPelt(
            model="l2",
            min_size=2,
            jump=5,
        )
        .fit_predict(
            signals,
            pen=1_000_000,
        )
    )

    assert gpu == cpu

    for result in gpu:
        assert result == [333]


def test_many_change_points():
    """
    Test signals with frequent strong level changes.
    """

    rng = np.random.default_rng(3000)

    n_series = 6
    segment_length = 40

    levels = np.array([
        0.0,
        4.0,
        -3.0,
        5.0,
        -2.0,
        3.0,
        -4.0,
        2.0,
    ])

    signals = []

    for _ in range(n_series):

        pieces = []

        for level in levels:

            piece = rng.normal(
                loc=level,
                scale=0.25,
                size=segment_length,
            )

            pieces.append(piece)

        signals.append(
            np.concatenate(pieces)
        )

    signals = np.stack(signals)

    cpu = cpu_batch_pelt(
        signals=signals,
        min_size=5,
        jump=5,
        pen=5,
    )

    gpu = (
        BatchPelt(
            model="l2",
            min_size=5,
            jump=5,
        )
        .fit_predict(
            signals,
            pen=5,
        )
    )

    assert gpu == cpu


def test_randomized_multivariate_batches():
    """
    Randomized multivariate piecewise signals.
    """

    rng = np.random.default_rng(4000)

    n_series = 5
    n_samples = 305
    n_features = 3

    signals = []

    for series_index in range(n_series):

        first = rng.normal(
            loc=[
                0.0,
                1.0,
                -1.0,
            ],
            scale=0.4,
            size=(100, n_features),
        )

        second = rng.normal(
            loc=[
                3.0 + 0.1 * series_index,
                -2.0,
                2.0,
            ],
            scale=0.4,
            size=(100, n_features),
        )

        third = rng.normal(
            loc=[
                -3.0,
                4.0,
                0.5,
            ],
            scale=0.4,
            size=(
                n_samples - 200,
                n_features,
            ),
        )

        signals.append(
            np.vstack([
                first,
                second,
                third,
            ])
        )

    signals = np.stack(signals)

    settings = [
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
    ]

    for config in settings:

        cpu = cpu_batch_pelt(
            signals=signals,
            min_size=config["min_size"],
            jump=config["jump"],
            pen=config["pen"],
        )

        gpu = (
            BatchPelt(
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
