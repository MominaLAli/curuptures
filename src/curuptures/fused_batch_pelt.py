import cupy as cp

from .batch_pelt import BatchPelt
from .fused_pelt_kernels import fused_masked_totals_l2


class FusedBatchPelt(BatchPelt):
    """
    Experimental exact GPU PELT using a fused masked-total kernel.

    The CUDA kernel evaluates the full series x candidate grid,
    but performs L2 segment-cost arithmetic only for candidates
    that remain admissible for each individual series.

    No global candidate-union compaction is required.
    """

    def __init__(
        self,
        model="l2",
        min_size=2,
        jump=5,
    ):
        super().__init__(
            model=model,
            min_size=min_size,
            jump=jump,
        )

    def _segment(self, pen):
        """
        Run exact PELT using the fused masked-total CUDA kernel.
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

        endpoints.append(n)

        # --------------------------------------------------
        # Dynamic-programming state
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

        candidate_grid = cp.arange(
            0,
            n,
            self.jump,
            dtype=cp.int64,
        )

        n_grid = candidate_grid.size

        admissible = cp.zeros(
            (batch, n_grid),
            dtype=cp.bool_,
        )

        rows = cp.arange(
            batch,
            dtype=cp.int64,
        )

        # Reusable workspace for fused totals.
        totals_workspace = cp.empty(
            (batch, n_grid),
            dtype=cp.float64,
        )

        # --------------------------------------------------
        # PELT recursion
        # --------------------------------------------------

        for end in endpoints:

            new_start = (
                (end - self.min_size)
                // self.jump
            ) * self.jump

            if new_start < 0:
                continue

            new_index = (
                new_start
                // self.jump
            )

            # Add/re-add newest candidate.
            admissible[
                :,
                new_index,
            ] = True

            n_candidates = (
                new_index + 1
            )

            # --------------------------------------------------
            # Fused candidate evaluation
            #
            # One CUDA kernel spans:
            #
            #     series x candidate-grid
            #
            # Inactive candidates exit before L2 arithmetic.
            # Candidates whose DP prefix cost is infinite also
            # exit immediately.
            # --------------------------------------------------

            totals = fused_masked_totals_l2(
                prefix_sum=self.cost.prefix_sum,
                prefix_sq_sum=self.cost.prefix_sq_sum,
                best_cost=best_cost,
                admissible=admissible,
                n_candidates=n_candidates,
                end=end,
                jump=self.jump,
                pen=pen,
                output=totals_workspace,
            )

            # --------------------------------------------------
            # Optimal DP state independently for every series
            # --------------------------------------------------

            best_index = cp.argmin(
                totals,
                axis=1,
            )

            best_value = totals[
                rows,
                best_index,
            ]

            best_start = (
                best_index
                * self.jump
            )

            best_cost[
                :,
                end,
            ] = best_value

            previous[
                :,
                end,
            ] = best_start

            # --------------------------------------------------
            # Exact PELT pruning
            #
            # Invalid-prefix candidates already have +inf totals,
            # so they automatically fail this comparison.
            # --------------------------------------------------

            active = admissible[
                :,
                :n_candidates,
            ]

            keep = (
                active
                & (
                    totals
                    <= (
                        best_value[:, None]
                        + pen
                    )
                )
            )

            # Clear processed region and restore survivors.
            admissible[
                :,
                :n_candidates,
            ] = keep

        # --------------------------------------------------
        # Final GPU -> CPU transfer
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
