import cupy as cp

from .batch_pelt import BatchPelt


class GroupedBatchPelt(BatchPelt):
    """
    Experimental dynamically grouped GPU PELT.

    Series are partitioned at every endpoint according to their
    current number of active PELT candidates. Each group evaluates
    only the union of candidates required by series in that group.

    The PELT objective, pruning rule, and dynamic-programming
    recurrence are unchanged.

    Parameters
    ----------
    model : str, default="l2"
        Currently only the L2 cost is supported.

    min_size : int, default=2
        Minimum allowed segment length.

    jump : int, default=5
        Candidate breakpoint spacing.

    n_groups : int, default=4
        Number of dynamic series groups.
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

        # Diagnostic counter:
        # actual number of candidate-series cost evaluations.
        self.candidate_evaluations_ = None

    def _segment(self, pen):
        """
        Run exact PELT using dynamically grouped GPU execution.
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

        # The signal end is always considered.
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

        # Count actual group × local-union evaluations.
        candidate_evaluations = cp.asarray(
            0,
            dtype=cp.int64,
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

            # If no series has a usable candidate, skip endpoint.
            active_any = cp.any(
                active,
                axis=0,
            )

            if not bool(
                cp.any(active_any).item()
            ):
                continue

            # --------------------------------------------------
            # Dynamic grouping
            #
            # Sort series by active candidate count.
            # Neighboring series then form groups.
            # --------------------------------------------------

            active_counts = cp.sum(
                active,
                axis=1,
                dtype=cp.int64,
            )

            sorted_rows = cp.argsort(
                active_counts
            )

            n_groups = min(
                self.n_groups,
                batch,
            )

            # We rebuild the admissible mask from the candidates
            # surviving pruning in each group.
            admissible[
                :,
                :n_candidates,
            ] = False

            for group_index in range(
                n_groups
            ):

                lo = (
                    group_index
                    * batch
                    // n_groups
                )

                hi = (
                    (group_index + 1)
                    * batch
                    // n_groups
                )

                group_rows = sorted_rows[
                    lo:hi
                ]

                if group_rows.size == 0:
                    continue

                group_active = active[
                    group_rows,
                    :
                ]

                # Local union for this group only.
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
                # Row-selective GPU segment costs
                #
                # Shape:
                # (n_group_series, n_group_candidates)
                # --------------------------------------------------

                segment_costs = (
                    self.cost._error_many_rows_unchecked(
                        rows=group_rows,
                        starts=starts,
                        end=end,
                    )
                )

                candidate_evaluations = (
                    candidate_evaluations
                    + (
                        group_rows.size
                        * starts.size
                    )
                )

                prefix_costs = best_cost[
                    group_rows[:, None],
                    starts[None, :],
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
                # Optimal DP state for this group
                # --------------------------------------------------

                local_rows = cp.arange(
                    group_rows.size,
                    dtype=cp.int64,
                )

                best_index = cp.argmin(
                    masked_totals,
                    axis=1,
                )

                best_value = masked_totals[
                    local_rows,
                    best_index,
                ]

                best_start = starts[
                    best_index
                ]

                best_cost[
                    group_rows,
                    end,
                ] = best_value

                previous[
                    group_rows,
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

                # Restore only surviving candidates for the
                # appropriate series in this group.
                admissible[
                    group_rows[:, None],
                    selected_indices[None, :],
                ] = keep_selected

        # --------------------------------------------------
        # Final GPU -> CPU transfer
        # --------------------------------------------------

        previous_cpu = cp.asnumpy(
            previous
        )

        self.candidate_evaluations_ = int(
            candidate_evaluations.item()
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
