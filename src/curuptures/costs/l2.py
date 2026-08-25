import cupy as cp
import numpy as np


class CostL2:
    """
    GPU-accelerated L2 segment cost.

    The cost of a segment [start:end) is the sum of squared
    deviations from the segment mean.

    Parameters
    ----------
    dtype : data-type, default cp.float64
        Data type used internally on the GPU.
    """

    model = "l2"
    min_size = 1

    def __init__(self, dtype=cp.float64):
        self.dtype = dtype

        self.signal = None
        self.prefix_sum = None
        self.prefix_sq_sum = None
        self.n_samples = None
        self.n_features = None

    def fit(self, signal):
        """
        Copy the signal to the GPU and precompute prefix statistics.

        Parameters
        ----------
        signal : numpy.ndarray or cupy.ndarray
            Shape:
                (n_samples,)
            or
                (n_samples, n_features)

        Returns
        -------
        self
        """

        x = cp.asarray(signal, dtype=self.dtype)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if x.ndim != 2:
            raise ValueError(
                "signal must have shape (n_samples,) "
                "or (n_samples, n_features)"
            )

        if x.shape[0] == 0:
            raise ValueError("signal must contain at least one sample")

        self.signal = x

        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]

        zero_row = cp.zeros(
            (1, self.n_features),
            dtype=self.dtype,
        )

        self.prefix_sum = cp.concatenate(
            [
                zero_row,
                cp.cumsum(x, axis=0),
            ],
            axis=0,
        )

        self.prefix_sq_sum = cp.concatenate(
            [
                zero_row,
                cp.cumsum(x * x, axis=0),
            ],
            axis=0,
        )

        return self

    def _check_is_fitted(self):
        if self.signal is None:
            raise RuntimeError(
                "CostL2 must be fitted before calculating costs"
            )

    def error_gpu(self, start, end):
        """
        Compute one segment cost and keep the result on the GPU.

        Returns
        -------
        cupy scalar
        """

        self._check_is_fitted()

        if start < 0:
            raise ValueError("start must be >= 0")

        if end > self.n_samples:
            raise ValueError(
                f"end={end} exceeds signal length {self.n_samples}"
            )

        if end <= start:
            raise ValueError("end must be greater than start")

        length = end - start

        if length < self.min_size:
            raise ValueError("segment is too short")

        segment_sum = (
            self.prefix_sum[end]
            - self.prefix_sum[start]
        )

        segment_sq_sum = (
            self.prefix_sq_sum[end]
            - self.prefix_sq_sum[start]
        )

        cost_per_feature = (
            segment_sq_sum
            - (segment_sum * segment_sum) / length
        )

        cost = cp.sum(cost_per_feature)

        # Numerical roundoff can occasionally create a tiny
        # negative value such as -1e-13.
        return cp.maximum(cost, 0.0)

    def error(self, start, end):
        """
        Compute one segment cost and return it as a Python float.

        This is convenient for compatibility/testing.
        """

        return float(
            self.error_gpu(start, end).item()
        )

    def error_many(self, starts, end):
        """
        Compute costs for many candidate start positions simultaneously.

        Parameters
        ----------
        starts : array-like
            Candidate segment start positions.

        end : int
            Common segment end position.

        Returns
        -------
        cupy.ndarray
            GPU array containing one cost per candidate start.
        """

        self._check_is_fitted()

        starts = cp.asarray(
            starts,
            dtype=cp.int64,
        )

        if starts.ndim != 1:
            raise ValueError("starts must be one-dimensional")

        if starts.size == 0:
            return cp.empty(
                0,
                dtype=self.dtype,
            )

        if end > self.n_samples:
            raise ValueError(
                f"end={end} exceeds signal length {self.n_samples}"
            )

        if bool(cp.any(starts < 0).item()):
            raise ValueError("starts must be >= 0")

        lengths = end - starts

        if bool(cp.any(lengths < self.min_size).item()):
            raise ValueError(
                "all segments must satisfy min_size"
            )

        sums = (
            self.prefix_sum[end]
            - self.prefix_sum[starts]
        )

        sq_sums = (
            self.prefix_sq_sum[end]
            - self.prefix_sq_sum[starts]
        )

        lengths = lengths.astype(
            self.dtype
        )[:, None]

        cost_per_feature = (
            sq_sums
            - (sums * sums) / lengths
        )

        costs = cp.sum(
            cost_per_feature,
            axis=1,
        )

        return cp.maximum(costs, 0.0)
