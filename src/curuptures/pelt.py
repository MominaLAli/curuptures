import cupy as cp
import numpy as np

from .costs import CostL2


class Pelt:
    """
    PELT change-point detection using GPU-accelerated L2 costs.

    Parameters
    ----------
    model : str, default="l2"
        Segment cost model. Currently only "l2" is supported.

    min_size : int, default=2
        Minimum allowed segment length.

    jump : int, default=5
        Candidate breakpoint spacing.
    """

    def __init__(
        self,
        model="l2",
        min_size=2,
        jump=5,
    ):
        if model != "l2":
            raise NotImplementedError(
                "cuRuptures currently supports only model='l2'"
            )

        if min_size < 1:
            raise ValueError("min_size must be >= 1")

        if jump < 1:
            raise ValueError("jump must be >= 1")

        self.model = model
        self.cost = CostL2()

        self.min_size = max(
            int(min_size),
            self.cost.min_size,
        )

        self.jump = int(jump)

        self.n_samples = None

    def fit(self, signal):
        """
        Fit the segment-cost model to a signal.
        """

        self.cost.fit(signal)

        self.n_samples = self.cost.n_samples

        return self

    def _check_is_fitted(self):
        if self.n_samples is None:
            raise RuntimeError(
                "Pelt must be fitted before predict()"
            )

    def _segment(self, pen):
        """
        Run PELT dynamic programming.

        Candidate segment costs are evaluated together on the GPU.
        """

        n = self.n_samples

        if n < self.min_size:
            raise ValueError(
                "signal is shorter than min_size"
            )

        # Best objective value for signal[0:t].
        best_cost = {
            0: 0.0
        }

        # previous[t] stores the breakpoint immediately
        # preceding t in the optimal segmentation.
        previous = {}

        # Candidate previous breakpoints that have not
        # been removed by the PELT pruning rule.
        admissible = []

        # Same candidate endpoint grid used by the
        # standard PELT interface.
        endpoints = [
            k
            for k in range(
                0,
                n,
                self.jump,
            )
            if k >= self.min_size
        ]

        # The final sample is always a breakpoint,
        # even when it is not on the jump grid.
        endpoints.append(n)

        for end in endpoints:

            # Introduce the newest candidate start that
            # can produce a segment satisfying min_size.
            new_start = (
                (end - self.min_size)
                // self.jump
            ) * self.jump

            admissible.append(new_start)

            # Keep only candidates for which a valid
            # optimal prefix solution already exists.
            valid_starts = [
                start
                for start in admissible
                if (
                    start in best_cost
                    and end - start >= self.min_size
                )
            ]

            if not valid_starts:
                continue

            # -------------------------------------------------
            # GPU WORK
            #
            # Compute C(start, end) for every active candidate
            # simultaneously.
            # -------------------------------------------------

            gpu_segment_costs = self.cost._error_many_unchecked(
                starts=valid_starts,
                end=end,
            )
            segment_costs = cp.asnumpy(
                gpu_segment_costs
            )

            # -------------------------------------------------
            # Dynamic programming update
            # -------------------------------------------------

            prefix_costs = np.array(
                [
                    best_cost[start]
                    for start in valid_starts
                ],
                dtype=np.float64,
            )

            totals = (
                prefix_costs
                + segment_costs
                + pen
            )

            best_index = int(
                np.argmin(totals)
            )

            best_value = float(
                totals[best_index]
            )

            best_start = valid_starts[
                best_index
            ]

            best_cost[end] = best_value
            previous[end] = best_start

            # -------------------------------------------------
            # PELT pruning
            # -------------------------------------------------

            threshold = (
                best_value
                + pen
            )

            admissible = [
                start
                for start, total
                in zip(
                    valid_starts,
                    totals,
                )
                if total <= threshold
            ]

        if n not in previous:
            raise RuntimeError(
                "No valid segmentation could be found"
            )

        # -------------------------------------------------
        # Reconstruct breakpoints
        # -------------------------------------------------

        breakpoints = []

        current = n

        while current > 0:

            breakpoints.append(
                current
            )

            current = previous[
                current
            ]

        breakpoints.reverse()

        return breakpoints

    def predict(self, pen):
        """
        Return optimal breakpoint locations.

        Parameters
        ----------
        pen : float
            Positive change-point penalty.
        """

        self._check_is_fitted()

        if pen <= 0:
            raise ValueError(
                "pen must be greater than 0"
            )

        return self._segment(
            float(pen)
        )

    def fit_predict(
        self,
        signal,
        pen,
    ):
        """
        Fit the signal and return breakpoints.
        """

        return (
            self.fit(signal)
            .predict(pen)
        )
