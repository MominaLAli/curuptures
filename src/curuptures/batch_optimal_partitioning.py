import cupy as cp

from .costs import BatchCostL2


class BatchOptimalPartitioning:
    """
    Batched GPU change-point detection with a PELT-compatible
    penalized segmentation objective.

    This initial implementation performs exact dynamic programming
    without PELT pruning. The purpose of this version is to establish
    correctness and exploit parallelism across independent time series.

    Parameters
    ----------
    model : str, default="l2"
        Currently only "l2" is supported.

    min_size : int, default=2
        Minimum segment length.

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
                "BatchOptimalPartitioning currently supports only model='l2'"
            )

        if min_size < 1:
            raise ValueError(
                "min_size must be >= 1"
            )

        if jump < 1:
            raise ValueError(
                "jump must be >= 1"
            )

        self.model = model
        self.cost = BatchCostL2()

        self.min_size = max(
            int(min_size),
            self.cost.min_size,
        )

        self.jump = int(jump)

        self.n_series = None
        self.n_samples = None

    def fit(self, signals):
        """
        Fit batched L2 prefix statistics on the GPU.

        Parameters
        ----------
        signals : array-like

            Shape:
                (n_series, n_samples)

            or:
                (n_series, n_samples, n_features)
        """

        self.cost.fit(signals)

        self.n_series = self.cost.n_series
        self.n_samples = self.cost.n_samples

        return self

    def _check_is_fitted(self):
        if self.n_samples is None:
            raise RuntimeError(
                "BatchOptimalPartitioning must be fitted before predict()"
            )

    def _segment(self, pen):
        """
        Exact batched dynamic programming.

        All series share the same candidate breakpoint grid,
        while objective values are maintained independently
        for each series.
        """

        n = self.n_samples
        batch = self.n_series

        if n < self.min_size:
            raise ValueError(
                "signal length is shorter than min_size"
            )

        endpoints = [
            k
            for k in range(
                0,
                n,
                self.jump,
            )
            if k >= self.min_size
        ]

        # Signal end is always included.
        endpoints.append(n)

        # --------------------------------------------------
        # GPU dynamic-programming state
        # --------------------------------------------------

        best_cost = cp.full(
            (batch, n + 1),
            cp.inf,
            dtype=cp.float64,
        )

        previous = cp.full(
            (batch, n + 1),
            -1,
            dtype=cp.int64,
        )

        best_cost[:, 0] = 0.0

        # Every potential start position on the jump grid.
        candidate_grid = cp.arange(
            0,
            n,
            self.jump,
            dtype=cp.int64,
        )

        rows = cp.arange(
            batch,
            dtype=cp.int64,
        )

        for end in endpoints:

            # Largest candidate start satisfying min_size.
            max_start = (
                (end - self.min_size)
                // self.jump
            ) * self.jump

            if max_start < 0:
                continue

            n_candidates = (
                max_start // self.jump
            ) + 1

            starts = candidate_grid[
                :n_candidates
            ]

            # --------------------------------------------------
            # Batched GPU segment costs
            #
            # Shape:
            # (n_series, n_candidates)
            # --------------------------------------------------

            segment_costs = self.cost._error_many_unchecked(
                starts=starts,
                end=end,
            )
            # Prefix states that were never feasible remain
            # infinity, so they cannot be selected.
            totals = (
                best_cost[:, starts]
                + segment_costs
                + pen
            )

            # Independent optimum for each time series.
            best_index = cp.argmin(
                totals,
                axis=1,
            )

            best_value = totals[
                rows,
                best_index,
            ]

            best_start = starts[
                best_index
            ]

            best_cost[:, end] = (
                best_value
            )

            previous[:, end] = (
                best_start
            )

        # --------------------------------------------------
        # One final GPU -> CPU copy for reconstruction
        # --------------------------------------------------

        previous_cpu = cp.asnumpy(
            previous
        )

        results = []

        for series_index in range(batch):

            if previous_cpu[
                series_index,
                n,
            ] < 0:
                raise RuntimeError(
                    "No valid segmentation could be found "
                    f"for series {series_index}"
                )

            breakpoints = []

            current = n

            while current > 0:

                breakpoints.append(
                    current
                )

                current = int(
                    previous_cpu[
                        series_index,
                        current,
                    ]
                )

            breakpoints.reverse()

            results.append(
                breakpoints
            )

        return results

    def predict(self, pen):
        """
        Return one breakpoint list per series.
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
        signals,
        pen,
    ):
        """
        Fit a batch and return breakpoint lists.
        """

        return (
            self.fit(signals)
            .predict(pen)
        )
