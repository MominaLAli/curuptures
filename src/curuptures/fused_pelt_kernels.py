import cupy as cp


_FUSED_MASKED_TOTALS_SOURCE = r'''
extern "C" __global__
void fused_masked_totals_l2(
    const double* prefix_sum,
    const double* prefix_sq_sum,
    const double* best_cost,
    const unsigned char* admissible,
    double* totals,
    const long long totals_row_stride,
    const long long n_rows,
    const long long n_samples_plus_one,
    const long long n_grid,
    const long long n_features,
    const long long n_candidates,
    const long long end,
    const long long jump,
    const double pen
)
{
    const double INF =
        __longlong_as_double(
            0x7ff0000000000000LL
        );

    const long long idx =
        (long long)blockDim.x * blockIdx.x
        + threadIdx.x;

    const long long total_pairs =
        n_rows * n_candidates;

    if (idx >= total_pairs) {
        return;
    }

    const long long row =
        idx / n_candidates;

    const long long candidate_index =
        idx - row * n_candidates;

    const long long output_offset =
        row * totals_row_stride
        + candidate_index;

    const long long start =
        candidate_index * jump;

    const long long admissible_offset =
        row * n_grid + candidate_index;

    const long long best_offset =
        row * n_samples_plus_one + start;

    /*
     * Candidate is inactive for this particular
     * series: do no segment-cost arithmetic.
     */
    if (!admissible[admissible_offset]) {
        totals[output_offset] = INF;
        return;
    }

    const double prefix_cost =
        best_cost[best_offset];

    if (prefix_cost == INF) {
        totals[output_offset] = INF;
        return;
    }

    double segment_cost = 0.0;

    const long long prefix_row_base =
        row
        * n_samples_plus_one
        * n_features;

    const long long end_base =
        prefix_row_base
        + end * n_features;

    const long long start_base =
        prefix_row_base
        + start * n_features;

    const double length =
        (double)(end - start);

    for (
        long long feature = 0;
        feature < n_features;
        ++feature
    ) {
        const double sum =
            prefix_sum[
                end_base + feature
            ]
            - prefix_sum[
                start_base + feature
            ];

        const double sq_sum =
            prefix_sq_sum[
                end_base + feature
            ]
            - prefix_sq_sum[
                start_base + feature
            ];

        const double cost =
            sq_sum
            - (sum * sum) / length;

        segment_cost += cost;
    }

    /*
     * Match BatchCostL2 semantics exactly:
     * sum feature contributions first, then
     * clamp the total L2 segment cost.
     */
    if (segment_cost < 0.0) {
        segment_cost = 0.0;
    }

    totals[output_offset] =
        prefix_cost
        + segment_cost
        + pen;
}
'''


fused_masked_totals_l2_kernel = cp.RawKernel(
    _FUSED_MASKED_TOTALS_SOURCE,
    "fused_masked_totals_l2",
)


def fused_masked_totals_l2(
    prefix_sum,
    prefix_sq_sum,
    best_cost,
    admissible,
    n_candidates,
    end,
    jump,
    pen,
    output=None,
):
    """
    Compute masked PELT totals using one CUDA kernel.

    Heavy L2 arithmetic is performed only for candidate-series
    pairs that remain admissible.

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

    admissible : cupy.ndarray
        Boolean candidate mask with shape
        (n_series, n_grid).

    n_candidates : int
        Number of currently reachable candidate-grid positions.

    end : int
        Current segment endpoint.

    jump : int
        Candidate-grid spacing.

    pen : float
        PELT penalty.

    output : cupy.ndarray or None
        Optional reusable output workspace with shape at least
        (n_series, n_candidates).

    Returns
    -------
    cupy.ndarray
        Shape
        (n_series, n_candidates).
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

    if output is None:
        totals = cp.empty(
            (
                n_rows,
                n_candidates,
            ),
            dtype=cp.float64,
        )
    else:
        totals = output[
            :,
            :n_candidates,
        ]

    totals_row_stride = (
        totals.strides[0]
        // totals.itemsize
    )

    total_pairs = (
        n_rows
        * n_candidates
    )

    threads = 256

    blocks = (
        total_pairs
        + threads
        - 1
    ) // threads

    fused_masked_totals_l2_kernel(
        (blocks,),
        (threads,),
        (
            prefix_sum,
            prefix_sq_sum,
            best_cost,
            admissible,
            totals,
            int(totals_row_stride),
            n_rows,
            n_samples_plus_one,
            n_grid,
            n_features,
            n_candidates,
            int(end),
            int(jump),
            float(pen),
        ),
    )

    return totals
