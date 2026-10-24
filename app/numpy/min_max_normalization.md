# Min-Max Normalization & Z-Score Standardization — Full Notes

## Quick concept recap before the code

**Min-max normalization** rescales values into the range [0, 1], per column, using:
new_value = (value - column_min) / (column_max - column_min)

**Z-score standardization** rescales values so the column has mean 0 and standard deviation 1, using:
new_value = (value - column_mean) / column_std

The key requirement here is **"column-wise"** — each column has its own min/max/mean/std, computed independently from the other columns. This is where a new concept comes in: the `axis` parameter.

## New concept: `axis` in NumPy aggregation functions

Functions like `.min()`, `.max()`, `.mean()`, `.std()` need to know **which direction** to collapse when computing a single statistic. For a 2D array:
- `axis=0` means "collapse **down** the rows, for each column" — i.e., compute one result *per column*.
- `axis=1` means "collapse **across** the columns, for each row" — i.e., compute one result *per row*.

Since we want one min/max/mean/std **per column**, we need `axis=0`.

## The code

```python
"""
Column-wise Min-Max Normalization and Z-Score Standardization
================================================================
Both operations are done without loops, using NumPy's axis-based
aggregation (axis=0 -> compute per column) combined with broadcasting.
"""

import numpy as np


def min_max_normalize(data):
    """
    Scale each column of a 2D array to the 0-1 range.

    Formula per element: (value - column_min) / (column_max - column_min)
    """
    col_min = data.min(axis=0)              # shape (n_cols,) -- min of each column
    col_max = data.max(axis=0)               # shape (n_cols,) -- max of each column

    # Broadcasting: (data - col_min) subtracts each column's min from every
    # value in that column. Dividing by (col_max - col_min) scales each
    # column into the 0-1 range, independently of other columns.
    return (data - col_min) / (col_max - col_min)


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

    print("\n" + "=" * 60)
    print("Confirming alternate min-max implementation (np.ptp) matches")
    print("=" * 60)
    normalized_alt = min_max_normalize_alt(data)
    print("Matches original implementation:", np.array_equal(normalized, normalized_alt))


if __name__ == "__main__":
    run_demo()
```

### Output when run

```
Original data:
[[ 10 200   1]
 [ 20 400   2]
 [ 30 600   3]
 [ 40 800   4]]

Min-Max Normalization (each column scaled to 0-1):
[[0.         0.         0.        ]
 [0.33333333 0.33333333 0.33333333]
 [0.66666667 0.66666667 0.66666667]
 [1.         1.         1.        ]]
Verification -- column mins (should be 0): [0. 0. 0.]
Verification -- column maxs (should be 1): [1. 1. 1.]

Z-Score Standardization (each column: mean 0, std 1):
[[-1.34164079 -1.34164079 -1.34164079]
 [-0.4472136  -0.4472136  -0.4472136 ]
 [ 0.4472136   0.4472136   0.4472136 ]
 [ 1.34164079  1.34164079  1.34164079]]
Verification -- column means (should be ~0): [0. 0. 0.]
Verification -- column stds (should be 1): [1. 1. 1.]

Confirming alternate min-max implementation (np.ptp) matches:
Matches original implementation: True
```

## Line-by-line explanation

```python
def min_max_normalize(data):
```
Takes in a 2D array — any shape, any number of columns.

```python
col_min = data.min(axis=0)
```
Computes the minimum value **per column**, using `axis=0`. If `data` has shape `(4, 3)`, then `col_min` has shape `(3,)` — one minimum for each of the 3 columns. For our example: `[10, 200, 1]` (the smallest value in each column).

```python
col_max = data.max(axis=0)
```
Same idea, but the maximum per column: `[40, 800, 4]`.

```python
return (data - col_min) / (col_max - col_min)
```
This is where broadcasting does the real work. `data` has shape `(4, 3)`, and `col_min` has shape `(3,)` — matching `data`'s **last** dimension directly, so it broadcasts cleanly without needing any reshape. `data - col_min` subtracts each column's minimum from every value in that column. `col_max - col_min` computes each column's range (also shape `(3,)`), and dividing applies that column's range to every value in that column — again via broadcasting. The result: every column now spans exactly 0 to 1, independently of the other columns.

```python
def z_score_standardize(data):
    col_mean = data.mean(axis=0)
    col_std = data.std(axis=0)
    return (data - col_mean) / col_std
```
Structurally identical logic — `data.mean(axis=0)` gives the average of each column, `data.std(axis=0)` gives the standard deviation of each column (both shape `(3,)`), and the same broadcasting subtraction/division centers each column around 0 with unit spread.

**Verification in the output**: after min-max normalization, printing `.min(axis=0)` and `.max(axis=0)` again on the *result* confirms every column is now exactly `[0, 0, 0]` to `[1, 1, 1]`. After z-score standardization, the column means are (numerically) `[0, 0, 0]` and the column stds are `[1, 1, 1]` — exactly what both formulas promise.

One subtlety: printing used `.round(10)` on the means before display. This is because floating-point arithmetic isn't perfectly exact — the true mean is `0`, but due to rounding during the subtraction/division, you often get something like `-1.38e-17` instead of a clean `0`. Rounding to 10 decimal places hides that tiny, harmless floating-point noise for display purposes.

## Tracing the actual numbers by hand (min-max)

Input data:
```
data = [[10, 200, 1],
        [20, 400, 2],
        [30, 600, 3],
        [40, 800, 4]]
```

```python
col_min = data.min(axis=0)   # [10, 200, 1]  -- smallest value in each column
col_max = data.max(axis=0)   # [40, 800, 4]  -- largest value in each column
```
Column range (`col_max - col_min`) = `[30, 600, 3]`.

**Row 0** — `[10, 200, 1]`:
- Column 0: `(10 - 10) / 30 = 0.0`
- Column 1: `(200 - 200) / 600 = 0.0`
- Column 2: `(1 - 1) / 3 = 0.0`
-> `[0.0, 0.0, 0.0]`

**Row 1** — `[20, 400, 2]`:
- Column 0: `(20 - 10) / 30 = 0.3333...`
- Column 1: `(400 - 200) / 600 = 0.3333...`
- Column 2: `(2 - 1) / 3 = 0.3333...`
-> `[0.333..., 0.333..., 0.333...]`

**Row 2** — `[30, 600, 3]`:
- Column 0: `(30 - 10) / 30 = 0.6667...`
- Column 1: `(600 - 200) / 600 = 0.6667...`
- Column 2: `(3 - 1) / 3 = 0.6667...`
-> `[0.666..., 0.666..., 0.666...]`

**Row 3** — `[40, 800, 4]`:
- Column 0: `(40 - 10) / 30 = 1.0`
- Column 1: `(800 - 200) / 600 = 1.0`
- Column 2: `(4 - 1) / 3 = 1.0`
-> `[1.0, 1.0, 1.0]`

**Why every column looks identical here (a data-specific coincidence, not a general rule)**: this isn't a property of min-max normalization in general — it's specific to how this example data was constructed. Each column increases in perfectly even, proportional steps (column 0 goes 10->20->30->40, column 1 goes 200->400->600->800, column 2 goes 1->2->3->4 — each is just evenly spaced values scaled by a different multiplier). Since min-max normalization only cares about *relative position within each column's own range*, and all three columns have that same even spacing, they all normalize identically. If the data had irregular gaps (say column 0 were `[10, 15, 30, 40]` instead), that column's normalized values would differ from the others.

## Alternative ways to accomplish this

**1. Using `np.ptp()` for the range in min-max normalization**
`ptp` stands for "peak to peak" — it directly computes `max - min` in one call:
```python
def min_max_normalize_v2(data):
    return (data - data.min(axis=0)) / np.ptp(data, axis=0)
```
Slightly more concise, though it recomputes min/max internally rather than reusing values you already calculated — a minor tradeoff between readability and redundant computation.

Note: `ndarray.ptp()` as a method (`data.ptp(axis=0)`) was deprecated and removed in recent NumPy versions — use the standalone function form `np.ptp(data, axis=0)` instead for version safety.

**2. `keepdims=True` instead of relying on automatic broadcasting**
```python
col_min = data.min(axis=0, keepdims=True)   # shape (1, 3) instead of (3,)
col_max = data.max(axis=0, keepdims=True)
```
`keepdims=True` keeps the collapsed axis in the shape as a `1` instead of dropping it entirely — so you get `(1, 3)` instead of `(3,)`. Functionally identical result after broadcasting, but some people find this clearer because the shape explicitly shows "this represents one row, three columns" rather than an ambiguous 1D array — a stylistic/readability choice, not a functional necessity here since `(3,)` already broadcasts correctly on its own.

**3. Using `sklearn.preprocessing`**
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

normalized = MinMaxScaler().fit_transform(data)
standardized = StandardScaler().fit_transform(data)
```
This is the standard, production-grade approach in real ML pipelines — it handles edge cases (like a column with zero variance) more gracefully, and critically, it lets you **fit on training data and separately transform new/test data using the same scaling parameters** (important in ML so you don't leak test-set statistics into training). For a from-scratch numpy exercise like this one, though, writing it manually is exactly the right way to build the underlying intuition sklearn is abstracting away.

**4. Handling the "zero range/zero std" edge case**
Worth being aware of: if a column has identical values in every row, `col_max - col_min` (or `col_std`) is `0`, causing a division-by-zero (`nan` or `inf` in the result). A more defensive version might do `np.where(col_std == 0, 1, col_std)` to avoid dividing by zero — substituting 1 as a safe fallback divisor for constant columns. Not needed for the clean example above, but worth knowing for real datasets.

---

# Why Do We Normalize/Standardize At All?

## Why do we even need to calculate this?

The core problem: real-world datasets usually have columns on wildly different scales. Imagine a dataset with:
- **age**: ranges from 0 to 100
- **income**: ranges from 20,000 to 500,000
- **number of children**: ranges from 0 to 5

If you feed this directly into many machine learning algorithms, the columns with naturally larger numeric ranges (income) will dominate calculations purely because of their scale — not because they're actually more important. A few concrete reasons this matters:

1. **Distance-based algorithms** (k-nearest neighbors, k-means clustering, SVMs): these compute distances between data points. If income ranges in the hundreds of thousands and age ranges in single/double digits, the distance calculation is essentially only "seeing" income — age's contribution gets numerically swamped, even if age is equally or more predictive.

2. **Gradient descent-based models** (linear/logistic regression, neural networks): these adjust weights iteratively based on gradients. Features on very different scales cause the loss surface to become elongated/skewed, making convergence slower and less stable — training can take far longer or get stuck.

3. **Regularization** (L1/L2 penalties): these penalize large weight values. If one feature's scale is naturally larger, its weight will naturally need to be smaller to have equivalent influence — regularization then unfairly penalizes it more than a feature that happens to sit on a smaller natural scale.

4. **Principal Component Analysis (PCA)** and similar variance-based techniques: these identify directions of maximum variance. A feature with a huge numeric range will dominate the variance calculation purely due to scale, not genuine signal.

## Min-max vs. z-score — when each is used

- **Min-max normalization** (scales to [0, 1]) is common when you want values bounded to a fixed, known range — useful for neural networks (especially with sigmoid/tanh activations that expect inputs in a bounded range), image pixel data (already naturally 0-255, often rescaled to 0-1), or when you specifically need interpretable "percentage of the way between min and max."

- **Z-score standardization** (mean 0, std 1) is more common in general statistical/ML contexts because it's less sensitive to outliers skewing the *range* (min-max normalization can get badly distorted by a single extreme outlier, since it directly uses min/max), and many algorithms (linear regression, logistic regression, PCA, SVMs) implicitly assume roughly standardized, zero-centered input.

## Is it mandatory that values land exactly between 0 and 1?

**Yes, for min-max normalization specifically — by mathematical construction, as long as the column actually has variation (min != max).**

Here's why it's guaranteed, not just a happy coincidence: every value in a column, by definition, sits somewhere between that column's own min and max (since min and max *are* the smallest and largest values present). The formula `(value - col_min) / (col_max - col_min)` is precisely designed so that:
- when `value = col_min`, the result is `(col_min - col_min) / range = 0 / range = 0`
- when `value = col_max`, the result is `(col_max - col_min) / range = range / range = 1`
- any value strictly between `col_min` and `col_max` produces a fraction strictly between 0 and 1

So it's not an approximation or a tendency — it's a mathematical guarantee that holds for *any* input data, as long as every value in the column is between its own min and max (which is trivially always true, since min and max are defined as the extremes of that very column).

**The one edge case where this breaks**: if a column has **zero variation** — every value identical (say, a column of all `7`s) — then `col_max - col_min = 0`, and you get a division by zero. In that case, the formula produces `nan` (0/0) or `inf`, not a value in [0,1] — because the "range" to normalize against genuinely doesn't exist. This is exactly the edge case flagged earlier as worth defensively handling (e.g., substituting a fallback divisor of `1` for constant columns) — a real gap in the basic implementation, worth knowing about even though it didn't come up in the example data.

**For z-score standardization, by contrast, there's no such guarantee** — the output is *not* bounded to any fixed range. Standardized values can be any real number (technically unbounded), though in practice they usually fall roughly between -3 and +3 for normally-distributed data (since standard deviation naturally bounds "typical" spread) — but a genuine outlier in the original data could easily produce a standardized value of -10 or +15. That's actually one of the practical differences between the two methods: min-max guarantees a hard bound, z-score does not.