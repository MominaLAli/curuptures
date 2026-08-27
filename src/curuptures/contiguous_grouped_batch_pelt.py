import cupy as cp

from .batch_pelt import BatchPelt


class ContiguousGroupedBatchPelt(BatchPelt):
    """
    Experimental exact PELT using fixed contiguous series groups.

    Unlike GroupedBatchPelt, this implementation does not sort
    or dynamically gather series rows at every endpoint.

    The batch is divided into contiguous row ranges once. Each
    group evaluates only the local union of candidates required
    by its own series.

    This class is intended as a performance ablation to measure
    the cost of dynamic grouping and arbitrary row indexing.
    """

    def __init__(
        self,
        model="l2",
        min_size=2,
        jump=5,
        n_groups=4,
    ):
        super().__init__(
            model=model,
            min_size=min_size,
            jump=jump,
        )

        if n_groups < 1:
            raise ValueError(
                "n_groups must be >= 1"
            )

        self.n_groups = int(n_groups)
        self.candidate_evaluations_ = None

    def _segment(self, pen):
        """
        Run exact PELT with fixed contiguous GPU groups.
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

        candidate_evaluations = 0

        n_groups = min(
            self.n_groups,
            batch,
        )

        # --------------------------------------------------
        # Fixed contiguous group boundaries
        # --------------------------------------------------

        groups = []

        for group_index in range(
            n_groups
        ):
            row_start = (
                group_index
                * batch
                // n_groups
            )

            row_end = (
                (group_index + 1)
                * batch
                // n_groups
            )

            if row_start < row_end:
                groups.append(
                    (
                        row_start,
                        row_end,
                    )
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

            admissible[
                :,
                new_index,
            ] = True

            n_candidates = (
                new_index + 1
            )

            all_starts = candidate_grid[
                :n_candidates
            ]

            active = admissible[
                :,
                :n_candidates
            ]

            finite_prefix = cp.isfinite(
                best_cost[
                    :,
                    all_starts,
                ]
            )

            active = (
                active
                & finite_prefix
            )

            # Clear this processed region. Each group will
            # restore only candidates surviving exact PELT
            # pruning.
            admissible[
                :,
                :n_candidates,
            ] = False

            # --------------------------------------------------
            # Process fixed contiguous groups
            # --------------------------------------------------

            for (
                row_start,
                row_end,
            ) in groups:

                group_active = active[
                    row_start:row_end,
                    :
                ]

                group_active_any = cp.any(
                    group_active,
                    axis=0,
                )

                selected_indices = cp.where(
                    group_active_any
                )[0]

                if selected_indices.size == 0:
                    continue

                starts = all_starts[
                    selected_indices
                ]

                active_selected = group_active[
                    :,
                    selected_indices
                ]

                # --------------------------------------------------
                # Contiguous-row segment cost evaluation
                # --------------------------------------------------

                segment_costs = (
                    self.cost._error_many_row_slice_unchecked(
                        row_start=row_start,
                        row_end=row_end,
                        starts=starts,
                        end=end,
                    )
                )

                group_size = (
                    row_end
                    - row_start
                )

                candidate_evaluations += (
                    group_size
                    * int(starts.size)
                )

                prefix_costs = best_cost[
                    row_start:row_end,
                    :
                ][
                    :,
                    starts
                ]

                totals = (
                    prefix_costs
                    + segment_costs
                    + pen
                )

                masked_totals = cp.where(
                    active_selected,
                    totals,
                    cp.inf,
                )

                # --------------------------------------------------
                # Exact dynamic-programming update
                # --------------------------------------------------

                best_index = cp.argmin(
                    masked_totals,
                    axis=1,
                )

                local_rows = cp.arange(
                    group_size,
                    dtype=cp.int64,
                )

                best_value = masked_totals[
                    local_rows,
                    best_index,
                ]

                best_start = starts[
                    best_index
                ]

                best_cost[
                    row_start:row_end,
                    end,
                ] = best_value

                previous[
                    row_start:row_end,
                    end,
                ] = best_start

                # --------------------------------------------------
                # Exact PELT pruning
                # --------------------------------------------------

                keep_selected = (
                    active_selected
                    & (
                        totals
                        <= (
                            best_value[:, None]
                            + pen
                        )
                    )
                )

                group_admissible = admissible[
                    row_start:row_end,
                    :n_candidates,
                ]

                group_admissible[
                    :,
                    selected_indices,
                ] = keep_selected

        # --------------------------------------------------
        # Final GPU -> CPU transfer
        # --------------------------------------------------

        previous_cpu = cp.asnumpy(
            previous
        )

        self.candidate_evaluations_ = int(
            candidate_evaluations
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
