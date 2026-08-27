import cupy as cp


_FUSED_ENDPOINT_L2_SOURCE = r'''
extern "C" __global__
void fused_pelt_endpoint_l2(
    const double* prefix_sum,
    const double* prefix_sq_sum,
    double* best_cost,
    long long* previous,
    unsigned char* admissible,
    double* totals,
    const long long n_rows,
    const long long n_samples_plus_one,
    const long long n_grid,
    const long long n_features,
    const long long n_candidates,
    const long long end,
    const long long jump,
    const long long new_index,
    const double pen
)
{
    const long long row =
        (long long)blockIdx.x;

    if (row >= n_rows) {
        return;
    }

    const int tid =
        threadIdx.x;

    const int nthreads =
        blockDim.x;

    const double INF =
        __longlong_as_double(
            0x7ff0000000000000LL
        );

    const long long INVALID_INDEX =
        0x7fffffffffffffffLL;

    /*
     * One block owns one time series.
     *
     * Make the newest candidate admissible before
     * evaluating this endpoint.
     */
    if (tid == 0) {
        admissible[
            row * n_grid
            + new_index
        ] = 1;
    }

    __syncthreads();


    /*
     * Each thread evaluates a strided subset of
     * candidate starts.
     */
    double local_best =
        INF;

    long long local_best_index =
        INVALID_INDEX;

    const long long prefix_row_base =
        row
        * n_samples_plus_one
        * n_features;

    const long long best_row_base =
        row
        * n_samples_plus_one;

    const long long mask_row_base =
        row
        * n_grid;

    const long long totals_row_base =
        row
        * n_grid;


    for (
        long long candidate_index = tid;
        candidate_index < n_candidates;
        candidate_index += nthreads
    )
    {
        const long long start =
            candidate_index
            * jump;

        const long long mask_offset =
            mask_row_base
            + candidate_index;

        const long long output_offset =
            totals_row_base
            + candidate_index;

        double total =
            INF;

        if (
            admissible[
                mask_offset
            ]
        )
        {
            const double prefix_cost =
                best_cost[
                    best_row_base
                    + start
                ];

            if (
                prefix_cost
                != INF
            )
            {
                const double length =
                    (double)(
                        end
                        - start
                    );

                double segment_cost =
                    0.0;

                const long long end_base =
                    prefix_row_base
                    + end
                    * n_features;

                const long long start_base =
                    prefix_row_base
                    + start
                    * n_features;

                for (
                    long long feature = 0;
                    feature < n_features;
                    ++feature
                )
                {
                    const double sum =
                        prefix_sum[
                            end_base
                            + feature
                        ]
                        - prefix_sum[
                            start_base
                            + feature
                        ];

                    const double sq_sum =
                        prefix_sq_sum[
                            end_base
                            + feature
                        ]
                        - prefix_sq_sum[
                            start_base
                            + feature
                        ];

                    segment_cost += (
                        sq_sum
                        - (
                            sum
                            * sum
                        )
                        / length
                    );
                }

                if (
                    segment_cost
                    < 0.0
                )
                {
                    segment_cost =
                        0.0;
                }

                total = (
                    prefix_cost
                    + segment_cost
                    + pen
                );
            }
        }

        /*
         * Save each total so pruning can reuse it
         * after the block-level minimum reduction.
         */
        totals[
            output_offset
        ] = total;

        /*
         * Preserve argmin semantics:
         * on an exact tie, choose the earliest
         * candidate index.
         */
        if (
            total < local_best
            ||
            (
                total == local_best
                &&
                candidate_index
                < local_best_index
            )
        )
        {
            local_best =
                total;

            local_best_index =
                candidate_index;
        }
    }


    /*
     * Block-level reduction.
     */
    extern __shared__
    unsigned char shared_raw[];

    double* shared_values =
        (double*)shared_raw;

    long long* shared_indices =
        (long long*)(
            shared_values
            + nthreads
        );

    shared_values[
        tid
    ] = local_best;

    shared_indices[
        tid
    ] = local_best_index;

    __syncthreads();


    for (
        int offset =
            nthreads / 2;
        offset > 0;
        offset >>= 1
    )
    {
        if (
            tid
            < offset
        )
        {
            const double other_value =
                shared_values[
                    tid
                    + offset
                ];

            const long long other_index =
                shared_indices[
                    tid
                    + offset
                ];

            const double current_value =
                shared_values[
                    tid
                ];

            const long long current_index =
                shared_indices[
                    tid
                ];

            if (
                other_value
                < current_value
                ||
                (
                    other_value
                    == current_value
                    &&
                    other_index
                    < current_index
                )
            )
            {
                shared_values[
                    tid
                ] = other_value;

                shared_indices[
                    tid
                ] = other_index;
            }
        }

        __syncthreads();
    }


    const double best_value =
        shared_values[0];

    const long long best_index =
        shared_indices[0];


    /*
     * Match the existing BatchPelt behavior if no
     * usable prefix candidate exists at this endpoint.
     *
     * In that case we leave the admissibility state
     * unchanged and do not write a DP predecessor.
     */
    if (
        best_value
        == INF
    )
    {
        return;
    }


    if (
        tid == 0
    )
    {
        best_cost[
            best_row_base
            + end
        ] = best_value;

        previous[
            best_row_base
            + end
        ] = (
            best_index
            * jump
        );
    }


    /*
     * Exact PELT pruning.
     *
     * Each candidate is retained only when:
     *
     * total <= best_value + pen
     *
     * Inactive candidates and candidates with an
     * infinite prefix cost already have total = INF.
     */
    const double pruning_threshold =
        best_value
        + pen;


    for (
        long long candidate_index = tid;
        candidate_index < n_candidates;
        candidate_index += nthreads
    )
    {
        const double total =
            totals[
                totals_row_base
                + candidate_index
            ];

        admissible[
            mask_row_base
            + candidate_index
        ] = (
            total
            <= pruning_threshold
        );
    }
}
'''


fused_pelt_endpoint_l2_kernel = cp.RawKernel(
    _FUSED_ENDPOINT_L2_SOURCE,
    "fused_pelt_endpoint_l2",
)


def fused_pelt_endpoint_l2(
    prefix_sum,
    prefix_sq_sum,
    best_cost,
    previous,
    admissible,
    totals_workspace,
    n_candidates,
    end,
    jump,
    new_index,
    pen,
    threads=256,
):
    """
    Execute one complete exact PELT endpoint update.

    One CUDA block handles one independent time series.

    The kernel performs:

    1. candidate activation,
    2. masked L2 candidate evaluation,
    3. per-series argmin,
    4. DP state update,
    5. predecessor update,
    6. exact PELT pruning.

    Parameters
    ----------
    prefix_sum : cupy.ndarray
        Shape
        (n_series, n_samples + 1, n_features).

    prefix_sq_sum : cupy.ndarray
        Same shape as prefix_sum.

    best_cost : cupy.ndarray
        Shape
        (n_series, n_samples + 1).

    previous : cupy.ndarray
        Shape
        (n_series, n_samples + 1).

    admissible : cupy.ndarray
        Shape
        (n_series, n_grid).

    totals_workspace : cupy.ndarray
        Shape
        (n_series, n_grid).

    n_candidates : int
        Number of current candidate-grid positions.

    end : int
        Current endpoint.

    jump : int
        Candidate spacing.

    new_index : int
        Candidate-grid index newly introduced at
        this endpoint.

    pen : float
        PELT penalty.

    threads : int, default=256
        Threads per block.
    """

    n_rows = int(
        best_cost.shape[0]
    )

    n_samples_plus_one = int(
        best_cost.shape[1]
    )

    n_grid = int(
        admissible.shape[1]
    )

    n_features = int(
        prefix_sum.shape[2]
    )

    n_candidates = int(
        n_candidates
    )

    threads = int(
        threads
    )

    if threads < 1:
        raise ValueError(
            "threads must be >= 1"
        )

    if (
        threads
        & (
            threads - 1
        )
    ):
        raise ValueError(
            "threads must be a power of two"
        )

    shared_bytes = (
        threads
        * (
            8
            + 8
        )
    )

    fused_pelt_endpoint_l2_kernel(
        (n_rows,),
        (threads,),
        (
            prefix_sum,
            prefix_sq_sum,
            best_cost,
            previous,
            admissible,
            totals_workspace,
            n_rows,
            n_samples_plus_one,
            n_grid,
            n_features,
            n_candidates,
            int(end),
            int(jump),
            int(new_index),
            float(pen),
        ),
        shared_mem=shared_bytes,
    )
