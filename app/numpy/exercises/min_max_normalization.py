"""
Column-wise Min-Max Normalization and Z-Score Standardization
================================================================
Both operations are done without loops, using NumPy's axis-based
aggregation (axis=0 -> compute per column) combined with broadcasting.
"""

import numpy as np


"""
New concept: axis in NumPy aggregation functions

Functions like .min(), .max(), .mean(), .std() need to know which direction to collapse when computing a single statistic. For a 2D array:

axis=0 means "collapse down the rows, for each column" — i.e., compute one result per column.
axis=1 means "collapse across the columns, for each row" — i.e., compute one result per row.
"""
def min_max_normalize(data):
    """
    Scale each column of a 2D array to the 0-1 range.

    Formula per element: (value - column_min) / (column_max - column_min)
    """
    """
    Computes the minimum value per column, using axis=0 as explained above. If data has shape (4, 3), then col_min has shape (3,) — one minimum for each of the 3 columns. 
    For our example: [10, 200, 1] (the smallest value in each column).
    """
    col_min = data.min(axis=0)              # shape (n_cols,) -- min of each column
    col_max = data.max(axis=0)              # shape (n_cols,) -- max of each column

    # Broadcasting: (data - col_min) subtracts each column's min from every
    # value in that column. Dividing by (col_max - col_min) scales each
    # column into the 0-1 range, independently of other columns.
    min_max = (data - col_min) / (col_max - col_min)
    print("\n min_max: ", min_max)
    return min_max


def z_score_standardize(data):
    """
    Standardize each column of a 2D array to mean 0, std 1.

    Formula per element: (value - column_mean) / column_std
    """
    col_mean = data.mean(axis=0)             # shape (n_cols,) -- mean of each column
    col_std = data.std(axis=0)               # shape (n_cols,) -- std dev of each column

    # Same broadcasting pattern as min-max: subtract each column's mean,
    # then divide by that column's standard deviation.
    return (data - col_mean) / col_std


def min_max_normalize_alt(data):
    """Alternative: uses np.ptp() (peak-to-peak) to get the range directly."""
    return (data - data.min(axis=0)) / np.ptp(data, axis=0)


def run_demo():
    data = np.array([
        [10, 200, 1],
        [20, 400, 2],
        [30, 600, 3],
        [40, 800, 4],
    ])

    print("=" * 60)
    print("Original data")
    print("=" * 60)
    print(data)

    print("\n" + "=" * 60)
    print("Min-Max Normalization (each column scaled to 0-1)")
    print("=" * 60)
    normalized = min_max_normalize(data)
    print(normalized)
    print("\nVerification -- column mins (should be 0):", normalized.min(axis=0))
    print("Verification -- column maxs (should be 1):", normalized.max(axis=0))

    print("\n" + "=" * 60)
    print("Z-Score Standardization (each column: mean 0, std 1)")
    print("=" * 60)
    standardized = z_score_standardize(data)
    print(standardized)
    print("\nVerification -- column means (should be ~0):", standardized.mean(axis=0).round(10))
    print("Verification -- column stds (should be 1):", standardized.std(axis=0))

    """
    Since min-max normalization is defined to squeeze every column into exactly [0, 1], the smallest value in each column of the result should always be 0 and the largest 
    should always be 1, regardless of what the original data looked like.
    """
    print("\n" + "=" * 60)
    print("Confirming alternate min-max implementation (np.ptp) matches")
    print("=" * 60)
    normalized_alt = min_max_normalize_alt(data)
    print("Matches original implementation:", np.array_equal(normalized, normalized_alt))


if __name__ == "__main__":
    run_demo()