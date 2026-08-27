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
        Copy a batch of signals to the GPU and build prefix statistics.
        """

        x = cp.asarray(
            signals,
            dtype=self.dtype,
        )

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

    def _compute_many_gpu(
        self,
        starts,
        end,
    ):
        """
        Core batched GPU calculation.

        Assumes starts and end have already been validated.

        Returns
        -------
        cupy.ndarray
            Shape (n_series, n_candidates)
        """

        lengths = end - starts

        sums = (
            self.prefix_sum[
                :,
                end,
                :
            ][:, None, :]
            - self.prefix_sum[
                :,
                starts,
                :
            ]
        )

        sq_sums = (
            self.prefix_sq_sum[
                :,
                end,
                :
            ][:, None, :]
            - self.prefix_sq_sum[
                :,
                starts,
                :
            ]
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

    def _compute_many_rows_gpu(
        self,
        rows,
        starts,
        end,
    ):
        """
        Compute segment costs for a subset of series.

        Parameters
        ----------
        rows : cupy.ndarray
            Series indices with shape
            (n_group_series,).

        starts : cupy.ndarray
            Candidate segment starts with shape
            (n_candidates,).

        end : int
            Segment endpoint.

        Returns
        -------
        cupy.ndarray
            Shape
            (n_group_series, n_candidates).
        """

        lengths = end - starts

        sums = (
            self.prefix_sum[
                rows,
                end,
                :
            ][:, None, :]
            - self.prefix_sum[
                rows[:, None],
                starts[None, :],
                :
            ]
        )

        sq_sums = (
            self.prefix_sq_sum[
                rows,
                end,
                :
            ][:, None, :]
            - self.prefix_sq_sum[
                rows[:, None],
                starts[None, :],
                :
            ]
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

    def _compute_many_row_slice_gpu(
        self,
        row_start,
        row_end,
        starts,
        end,
    ):
        """
        Compute segment costs for a contiguous range of series.

        The series are selected by the half-open interval

            [row_start, row_end)

        This avoids arbitrary row-index gathering.

        Parameters
        ----------
        row_start : int
            First series index in the group.

        row_end : int
            Exclusive final series index in the group.

        starts : cupy.ndarray
            Candidate segment starts.

        end : int
            Segment endpoint.

        Returns
        -------
        cupy.ndarray
            Shape
            (row_end - row_start, n_candidates).
        """

        lengths = end - starts

        group_prefix = self.prefix_sum[
            row_start:row_end,
            :,
            :
        ]

        group_prefix_sq = self.prefix_sq_sum[
            row_start:row_end,
            :,
            :
        ]

        sums = (
            group_prefix[
                :,
                end,
                :
            ][:, None, :]
            - group_prefix[
                :,
                starts,
                :
            ]
        )

        sq_sums = (
            group_prefix_sq[
                :,
                end,
                :
            ][:, None, :]
            - group_prefix_sq[
                :,
                starts,
                :
            ]
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

    def _error_many_unchecked(
        self,
        starts,
        end,
    ):
        """
        Internal fast path used by BatchPelt.

        No GPU-side validity checks are performed.
        """

        if not isinstance(
            starts,
            cp.ndarray,
        ):
            starts = cp.asarray(
                starts,
                dtype=cp.int64,
            )

        elif starts.dtype != cp.int64:
            starts = starts.astype(
                cp.int64,
                copy=False,
            )

        if starts.size == 0:
            return cp.empty(
                (
                    self.n_series,
                    0,
                ),
                dtype=self.dtype,
            )

        return self._compute_many_gpu(
            starts=starts,
            end=end,
        )

    def _error_many_rows_unchecked(
        self,
        rows,
        starts,
        end,
    ):
        """
        Internal fast path for grouped BatchPelt.

        Computes candidate costs only for the requested
        series rows.
        """

        if not isinstance(
            rows,
            cp.ndarray,
        ):
            rows = cp.asarray(
                rows,
                dtype=cp.int64,
            )

        elif rows.dtype != cp.int64:
            rows = rows.astype(
                cp.int64,
                copy=False,
            )

        if not isinstance(
            starts,
            cp.ndarray,
        ):
            starts = cp.asarray(
                starts,
                dtype=cp.int64,
            )

        elif starts.dtype != cp.int64:
            starts = starts.astype(
                cp.int64,
                copy=False,
            )

        if rows.size == 0:
            return cp.empty(
                (
                    0,
                    starts.size,
                ),
                dtype=self.dtype,
            )

        if starts.size == 0:
            return cp.empty(
                (
                    rows.size,
                    0,
                ),
                dtype=self.dtype,
            )

        return self._compute_many_rows_gpu(
            rows=rows,
            starts=starts,
            end=end,
        )

    def _error_many_row_slice_unchecked(
        self,
        row_start,
        row_end,
        starts,
        end,
    ):
        """
        Internal fast path for a contiguous group of series.

        This avoids arbitrary row-index gathering and is intended
        for grouped GPU execution where group members occupy
        contiguous rows.
        """

        if not isinstance(
            starts,
            cp.ndarray,
        ):
            starts = cp.asarray(
                starts,
                dtype=cp.int64,
            )

        elif starts.dtype != cp.int64:
            starts = starts.astype(
                cp.int64,
                copy=False,
            )

        n_rows = (
            row_end
            - row_start
        )

        if starts.size == 0:
            return cp.empty(
                (
                    n_rows,
                    0,
                ),
                dtype=self.dtype,
            )

        return self._compute_many_row_slice_gpu(
            row_start=row_start,
            row_end=row_end,
            starts=starts,
            end=end,
        )

    def error_many(
        self,
        starts,
        end,
    ):
        """
        Compute candidate segment costs for every series.

        This public method performs input validation.
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
            cp.any(
                starts < 0
            ).item()
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

        return self._compute_many_gpu(
            starts=starts,
            end=end,
        )
