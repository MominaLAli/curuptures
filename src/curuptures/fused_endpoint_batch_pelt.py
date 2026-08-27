import cupy as cp

from .batch_pelt import BatchPelt
from .fused_endpoint_kernels import (
    fused_pelt_endpoint_l2,
)


class FusedEndpointBatchPelt(BatchPelt):
    """
    Experimental exact GPU PELT with one fused CUDA
    endpoint update.

    For each endpoint, one CUDA kernel performs:

    - candidate activation,
    - mask-aware L2 evaluation,
    - per-series argmin,
    - DP state update,
    - predecessor update,
    - exact PELT pruning.

    One CUDA block handles one independent series.
    """

    def __init__(
        self,
        model="l2",
        min_size=2,
        jump=5,
        threads=256,
    ):
        super().__init__(
            model=model,
            min_size=min_size,
            jump=jump,
        )

        self.threads = int(
            threads
        )

        if self.threads < 1:
            raise ValueError(
                "threads must be >= 1"
            )

        if (
            self.threads
            & (
                self.threads - 1
            )
        ):
            raise ValueError(
                "threads must be a power of two"
            )

    def _segment(
        self,
        pen,
    ):
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

        endpoints.append(
            n
        )

        # --------------------------------------------------
        # Dynamic-programming state
        # --------------------------------------------------

        best_cost = cp.full(
            (
                batch,
                n + 1,
            ),
            cp.inf,
            dtype=cp.float64,
        )

        previous = cp.full(
            (
                batch,
                n + 1,
            ),
            -1,
            dtype=cp.int64,
        )

        best_cost[
            :,
            0,
        ] = 0.0

        n_grid = (
            n
            + self.jump
            - 1
        ) // self.jump

        admissible = cp.zeros(
            (
                batch,
                n_grid,
            ),
            dtype=cp.bool_,
        )

        # Workspace reused by every endpoint.
        totals_workspace = cp.empty(
            (
                batch,
                n_grid,
            ),
            dtype=cp.float64,
        )

        # --------------------------------------------------
        # Exact PELT recurrence
        # --------------------------------------------------

        for end in endpoints:

            new_start = (
                (
                    end
                    - self.min_size
                )
                // self.jump
            ) * self.jump

            if new_start < 0:
                continue

            new_index = (
                new_start
                // self.jump
            )

            n_candidates = (
                new_index
                + 1
            )

            fused_pelt_endpoint_l2(
                prefix_sum=self.cost.prefix_sum,
                prefix_sq_sum=self.cost.prefix_sq_sum,
                best_cost=best_cost,
                previous=previous,
                admissible=admissible,
                totals_workspace=totals_workspace,
                n_candidates=n_candidates,
                end=end,
                jump=self.jump,
                new_index=new_index,
                pen=pen,
                threads=self.threads,
            )

        # --------------------------------------------------
        # One final device -> host transfer
        # --------------------------------------------------

        previous_cpu = cp.asnumpy(
            previous
        )

        results = []

        for series_index in range(
            batch
        ):

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
