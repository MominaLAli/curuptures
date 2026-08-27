import cupy as cp

from .costs import BatchCostL2


class BatchPelt:
    """
    Batched GPU implementation of PELT.

    Independent time series are processed in parallel on the GPU,
    while each series maintains its own PELT admissible candidate set.

    Parameters
    ----------
    model : str, default="l2"
        Currently only the L2 cost is supported.

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
                "BatchPelt currently supports only model='l2'"
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

        # Optional research diagnostics collected during PELT.
        self.candidate_stats_ = None

    def fit(self, signals):
        """
        Fit batched L2 prefix statistics on the GPU.
        """

        self.cost.fit(signals)

        self.n_series = self.cost.n_series
        self.n_samples = self.cost.n_samples

        return self

    def _check_is_fitted(self):
        if self.n_samples is None:
            raise RuntimeError(
                "BatchPelt must be fitted before predict()"
            )

    def _segment(self, pen, collect_stats=False):
        """
        Run batched PELT with independent pruning per series.

        Candidate segment costs are computed in parallel across
        both time series and candidate start positions.
        """

        n = self.n_samples
        batch = self.n_series

        # Reset optional research diagnostics.
        self.candidate_stats_ = None

        stats_endpoints = []
        stats_n_candidates = []
        stats_union_candidates = []
        stats_active_counts = []
        stats_survivor_counts = []

        # Diagnostic estimate of candidate-series work if
        # series are dynamically divided into smaller groups.
        stats_grouped_work = {
            2: [],
            4: [],
            8: [],
            16: [],
        }

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

        # The signal end is always a breakpoint candidate.
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

        # Candidate locations on the jump grid.
        candidate_grid = cp.arange(
            0,
            n,
            self.jump,
            dtype=cp.int64,
        )

        n_grid = candidate_grid.size

        # One admissibility mask per time series.
        #
        # admissible[i, j] == True means candidate_grid[j]
        # remains active for series i.
        admissible = cp.zeros(
            (batch, n_grid),
            dtype=cp.bool_,
        )

        rows = cp.arange(
            batch,
            dtype=cp.int64,
        )

        # --------------------------------------------------
        # PELT recursion
        # --------------------------------------------------

        for end in endpoints:

            # Same candidate-generation rule used by PELT.
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

            # Add/re-add the newest admissible candidate
            # independently for every series.
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

            # A candidate cannot be used until an optimal
            # prefix solution exists at that position.
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

            # Number of active PELT candidates for each series
            # before batch-wide candidate compaction.
            if collect_stats:
                active_counts = cp.sum(
                    active,
                    axis=1,
                    dtype=cp.int64,
                )

                # --------------------------------------------------
                # Hypothetical dynamically grouped execution.
                #
                # Series are sorted by current active-candidate
                # count, then neighboring series are assigned to
                # groups. Each group evaluates only the union of
                # candidates needed by its own members.
                # --------------------------------------------------

                sorted_rows = cp.argsort(
                    active_counts
                )

                for requested_groups in stats_grouped_work:

                    n_groups = min(
                        requested_groups,
                        batch,
                    )

                    grouped_work = cp.asarray(
                        0,
                        dtype=cp.int64,
                    )

                    for group_index in range(n_groups):

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

                        group_union = cp.count_nonzero(
                            cp.any(
                                group_active,
                                axis=0,
                            )
                        )

                        grouped_work = (
                            grouped_work
                            + (
                                group_rows.size
                                * group_union
                            )
                        )

                    stats_grouped_work[
                        requested_groups
                    ].append(
                        grouped_work
                    )

            # --------------------------------------------------
            # Compact candidates discarded by EVERY series.
            #
            # Each series still keeps its own independent mask.
            # --------------------------------------------------

            active_any = cp.any(
                active,
                axis=0,
            )

            selected_indices = cp.where(
                active_any
            )[0]

            if selected_indices.size == 0:

                if collect_stats:
                    stats_endpoints.append(
                        int(end)
                    )
                    stats_n_candidates.append(
                        int(n_candidates)
                    )
                    stats_union_candidates.append(
                        0
                    )
                    stats_active_counts.append(
                        active_counts.copy()
                    )
                    stats_survivor_counts.append(
                        cp.zeros_like(active_counts)
                    )

                continue

            starts = all_starts[
                selected_indices
            ]

            active_selected = active[
                :,
                selected_indices
            ]

            # --------------------------------------------------
            # Batched GPU segment costs
            #
            # Shape:
            # (n_series, n_selected_candidates)
            # --------------------------------------------------

            segment_costs = (
                self.cost._error_many_unchecked(
                    starts=starts,
                    end=end,
                )
            )

            totals = (
                best_cost[
                    :,
                    starts,
                ]
                + segment_costs
                + pen
            )

            # Candidates that are inactive for a particular
            # series must not participate in that series'
            # minimization.
            masked_totals = cp.where(
                active_selected,
                totals,
                cp.inf,
            )

            # --------------------------------------------------
            # Optimal DP state independently for every series
            # --------------------------------------------------

            best_index = cp.argmin(
                masked_totals,
                axis=1,
            )

            best_value = masked_totals[
                rows,
                best_index,
            ]

            best_start = starts[
                best_index
            ]

            best_cost[
                :,
                end,
            ] = best_value

            previous[
                :,
                end,
            ] = best_start

            # --------------------------------------------------
            # TRUE PELT PRUNING
            #
            # Keep candidate t for series i when:
            #
            # F_i(t) + C_i(t, end) + pen
            #     <=
            # F_i(end) + pen
            #
            # This mirrors the pruning test used by ruptures.
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

            if collect_stats:
                survivor_counts = cp.sum(
                    keep_selected,
                    axis=1,
                    dtype=cp.int64,
                )

                stats_endpoints.append(
                    int(end)
                )
                stats_n_candidates.append(
                    int(n_candidates)
                )
                stats_union_candidates.append(
                    int(selected_indices.size)
                )
                stats_active_counts.append(
                    active_counts.copy()
                )
                stats_survivor_counts.append(
                    survivor_counts.copy()
                )

            # Clear the processed candidate region.
            admissible[
                :,
                :n_candidates,
            ] = False

            # Restore only candidates that survived pruning.
            admissible[
                :,
                selected_indices,
            ] = keep_selected

        # --------------------------------------------------
        # One final GPU -> CPU transfer
        # --------------------------------------------------

        previous_cpu = cp.asnumpy(
            previous
        )

        if collect_stats and stats_active_counts:
            active_counts_gpu = cp.stack(
                stats_active_counts,
                axis=0,
            )

            survivor_counts_gpu = cp.stack(
                stats_survivor_counts,
                axis=0,
            )

            possible_gpu = cp.asarray(
                stats_n_candidates,
                dtype=cp.float64,
            )[:, None]

            union_gpu = cp.asarray(
                stats_union_candidates,
                dtype=cp.float64,
            )

            active_density_gpu = (
                active_counts_gpu
                / possible_gpu
            )

            survivor_density_gpu = (
                survivor_counts_gpu
                / possible_gpu
            )

            union_density_gpu = (
                union_gpu
                / possible_gpu[:, 0]
            )

            self.candidate_stats_ = {
                "endpoints": cp.asnumpy(
                    cp.asarray(stats_endpoints)
                ),
                "n_candidates": cp.asnumpy(
                    cp.asarray(stats_n_candidates)
                ),
                "union_candidates": cp.asnumpy(
                    cp.asarray(stats_union_candidates)
                ),
                "active_counts": cp.asnumpy(
                    active_counts_gpu
                ),
                "survivor_counts": cp.asnumpy(
                    survivor_counts_gpu
                ),
                "active_density": cp.asnumpy(
                    active_density_gpu
                ),
                "survivor_density": cp.asnumpy(
                    survivor_density_gpu
                ),
                "union_density": cp.asnumpy(
                    union_density_gpu
                ),
                "grouped_work": {
                    n_groups: cp.asnumpy(
                        cp.stack(values)
                    )
                    for n_groups, values
                    in stats_grouped_work.items()
                },
            }

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
