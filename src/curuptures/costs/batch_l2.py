import cupy as cp


class BatchCostL2:
    """
    Batched GPU-accelerated L2 segment cost.

    Input shapes
    ------------
    (n_series, n_samples)

    or

    (n_series, n_samples, n_features)
    """

    model = "l2"
    min_size = 1

    def __init__(self, dtype=cp.float64):
        self.dtype = dtype

        self.signal = None
        self.prefix_sum = None
        self.prefix_sq_sum = None

        self.n_series = None
        self.n_samples = None
        self.n_features = None

    def fit(self, signals):
        """
        Copy a batch of signals to GPU and build prefix statistics.
        """

        x = cp.asarray(
            signals,
            dtype=self.dtype,
        )

        # (batch, time) -> (batch, time, 1)
        if x.ndim == 2:
            x = x[:, :, None]

        if x.ndim != 3:
            raise ValueError(
                "signals must have shape "
                "(n_series, n_samples) or "
                "(n_series, n_samples, n_features)"
            )

        if x.shape[0] == 0:
            raise ValueError(
                "signals must contain at least one series"
            )

        if x.shape[1] == 0:
            raise ValueError(
                "signals must contain at least one sample"
            )

        self.signal = x

        self.n_series = x.shape[0]
        self.n_samples = x.shape[1]
        self.n_features = x.shape[2]

        zero = cp.zeros(
            (
                self.n_series,
                1,
                self.n_features,
            ),
            dtype=self.dtype,
        )

        self.prefix_sum = cp.concatenate(
            [
                zero,
                cp.cumsum(
                    x,
                    axis=1,
                ),
            ],
            axis=1,
        )

        self.prefix_sq_sum = cp.concatenate(
            [
                zero,
                cp.cumsum(
                    x * x,
                    axis=1,
                ),
            ],
            axis=1,
        )

        return self

    def _check_is_fitted(self):
        if self.signal is None:
            raise RuntimeError(
                "BatchCostL2 must be fitted "
                "before calculating costs"
            )

    def error_many(self, starts, end):
        """
        Compute candidate segment costs for every series.

        Parameters
        ----------
        starts : array-like
            Candidate segment starts.

        end : int
            Common segment endpoint.

        Returns
        -------
        cupy.ndarray
            Shape:
            (n_series, n_candidates)
        """

        self._check_is_fitted()

        starts = cp.asarray(
            starts,
            dtype=cp.int64,
        )

        if starts.ndim != 1:
            raise ValueError(
                "starts must be one-dimensional"
            )

        if starts.size == 0:
            return cp.empty(
                (
                    self.n_series,
                    0,
                ),
                dtype=self.dtype,
            )

        if end > self.n_samples:
            raise ValueError(
                f"end={end} exceeds signal length "
                f"{self.n_samples}"
            )

        if bool(
            cp.any(starts < 0).item()
        ):
            raise ValueError(
                "starts must be >= 0"
            )

        lengths = end - starts

        if bool(
            cp.any(
                lengths < self.min_size
            ).item()
        ):
            raise ValueError(
                "all segments must satisfy min_size"
            )

        # Shape:
        # (series, candidates, features)
        sums = (
            self.prefix_sum[:, end, :][:, None, :]
            - self.prefix_sum[:, starts, :]
        )

        sq_sums = (
            self.prefix_sq_sum[:, end, :][:, None, :]
            - self.prefix_sq_sum[:, starts, :]
        )

        lengths = lengths.astype(
            self.dtype
        )[None, :, None]

        cost_per_feature = (
            sq_sums
            - (sums * sums) / lengths
        )

        costs = cp.sum(
            cost_per_feature,
            axis=2,
        )

        return cp.maximum(
            costs,
            0.0,
        )
