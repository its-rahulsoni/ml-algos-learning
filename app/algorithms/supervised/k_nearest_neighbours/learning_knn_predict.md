# Understanding `knn_predict`

```python
def knn_predict(X_train, y_train, Q, k, leave_one_out=False):
    """Predict a label (0 or 1) for every row of Q.

    leave_one_out=True is for evaluating on the training data itself: Q must be
    X_train, and each point is stopped from voting for itself.
    """
    n_train = X_train.shape[0]
    max_k = n_train - 1 if leave_one_out else n_train
    if not (1 <= k <= max_k):
        raise ValueError(f"k must be between 1 and {max_k}, got {k}")

    sq_dists = squared_distances(Q, X_train)

    if leave_one_out:
        np.fill_diagonal(sq_dists, np.inf)

    order = np.argsort(sq_dists, axis=1, kind="stable")
    nearest = order[:, :k]
    nearest_labels = y_train[nearest]

    preds = np.empty(Q.shape[0], dtype=int)
    for i in range(Q.shape[0]):
        counts = np.bincount(nearest_labels[i])
        preds[i] = np.argmax(counts)

    return preds
```

**The concept, briefly**

`knn_predict` is the whole KNN algorithm in one function. For every query point it answers: which k training points are nearest, and what label do most of them have? That is Stage 1 (neighbors vote), Stage 3 (distance table), and Stage 4 (argsort, first k, labels, vote) put together. Nothing is "learned" here: the function just receives the stored data (`X_train`, `y_train`) and compares each query against it.

**The example I'll use**

```
Q (2 queries):          X_train (3 stored points):     y_train (labels):
[2, 0]  <- Q1           [0, 0]  <- A                   A = 0 (red)
[3, 3]  <- Q2           [3, 4]  <- B                   B = 1 (blue)
                        [1, 1]  <- C                   C = 0 (red)
```

I'll use k = 3, and mention k = 2 where it shows something useful.

**Step 1: check that k makes sense**

```python
n_train = X_train.shape[0]
max_k = n_train - 1 if leave_one_out else n_train
if not (1 <= k <= max_k):
    raise ValueError(...)
```

- `X_train.shape[0]` is the number of rows, so `n_train = 3`.
- You can't pick 4 neighbors from 3 stored points, so k must be between 1 and 3. If k = 0 or k = 4, the function stops with a clear message instead of failing strangely later.
- The `max_k` line handles `leave_one_out`, which I'll explain in Step 3. In that mode every point loses one candidate (itself), so the largest allowed k drops by 1.

**Step 2: build the distance table**

```python
sq_dists = squared_distances(Q, X_train)
```

This calls the function we explained earlier. For our example:

```
        A    B    C
Q1  [   4,  17,   2 ]
Q2  [  18,   1,   8 ]
```

Shape (n_query, n_train) = (2, 3). Row i is query i, column j is training point j. This is the only place distances are computed.

**Step 3: the `leave_one_out` option**

```python
if leave_one_out:
    np.fill_diagonal(sq_dists, np.inf)
```

This only matters when you ask the model to predict its own training points, for example to measure training accuracy. In that case `Q` is `X_train`, so each point sits in the stored data too. Here is the table for our 3 points against themselves:

```
        A    B    C
A  [    0,  25,   2 ]
B  [   25,   0,  13 ]
C  [    2,  13,   0 ]
```

The diagonal is 0 because each point is at distance 0 from itself. A point would always be its own nearest neighbor, and it would vote for its own label. That's cheating, and it is why k = 1 training accuracy always comes out at 100%. `np.fill_diagonal(..., np.inf)` sets the diagonal to infinity, so each point looks infinitely far from itself and is never picked as a neighbor. This is the "query point included among its own neighbors" pitfall from your list.

It also explains `max_k = n_train - 1`: each row now has one infinity in it, so only n_train − 1 real candidates are left. With k = n_train, an infinite-distance "neighbor" would be picked.

For normal predictions on new points (our example), this step is skipped.

**Step 4: sort each row with argsort**

```python
order = np.argsort(sq_dists, axis=1, kind="stable")
```

`argsort` returns positions ordered from nearest to farthest, not the distances. On our table:

- Q1 row [4, 17, 2]: smallest is 2 (position 2), then 4 (position 0), then 17 (position 1). So `[2, 0, 1]`.
- Q2 row [18, 1, 8]: smallest is 1 (position 1), then 8 (position 2), then 18 (position 0). So `[1, 2, 0]`.

```
order = [[2, 0, 1],
         [1, 2, 0]]       shape (2, 3)
```

Two details in that line:
- `axis=1` sorts within each row, so each query gets its own ranking. `axis=0` would sort down the columns, mixing different queries together, which makes no sense here.
- `kind="stable"` fixes ties. If a row were `[5, 5, 1]`, the two 5s are tied, and stable sorting keeps them in their original order, giving `[2, 0, 1]`. Without it, tied points could come out in either order. This keeps results repeatable, which matters when comparing against sklearn.

**Step 5: keep the first k positions**

```python
nearest = order[:, :k]
```

`[:, :k]` means all rows, first k columns. With k = 3 here it keeps everything. With k = 2 it keeps:

```
nearest = [[2, 0],       <- Q1's two nearest: C, then A
           [1, 2]]       <- Q2's two nearest: B, then C
```

`:k` takes positions 0 to k − 1, which is exactly k items, so there is no off-by-one. Shape is (n_query, k).

**Step 6: turn positions into labels**

```python
nearest_labels = y_train[nearest]
```

`nearest` holds positions of training points, and `y_train` is the label of each. Indexing with a table of positions replaces every position with its label and keeps the shape. For k = 3, `nearest` is `[[2, 0, 1], [1, 2, 0]]`:

```
nearest_labels = [[y[2], y[0], y[1]],     = [[0, 0, 1],      Q1: red, red, blue
                  [y[1], y[2], y[0]]]       [1, 0, 0]]      Q2: blue, red, red
```

This is where the earlier mix-up between a distance value and a point's position stays solved: distances found the neighbors, positions identified them, and the labels were fetched with those positions.

**Step 7: vote, query by query**

```python
preds = np.empty(Q.shape[0], dtype=int)
for i in range(Q.shape[0]):
    counts = np.bincount(nearest_labels[i])
    preds[i] = np.argmax(counts)
```

- `np.empty(Q.shape[0], dtype=int)` makes a blank list with room for one answer per query. Every slot is filled by the loop, so no leftover junk remains.
- The loop goes over queries (2 here), never over training points, so your no-loop rule holds.
- `np.bincount` counts how often each label appears. The position in the output is the label and the value is the count.
- `np.argmax` returns the position of the biggest count, which is the winning label.

For k = 3:
- Q1: labels `[0, 0, 1]`, so bincount is `[2, 1]` (label 0 twice, label 1 once), argmax is 0. Prediction: red.
- Q2: labels `[1, 0, 0]`, so bincount is `[2, 1]`, argmax is 0. Prediction: red.

For k = 2 on Q2, the neighbors are B and C, with labels `[1, 0]`, so bincount is `[1, 1]`. That's a tie, and `argmax` returns the first (smallest) label, 0. So the tie rule is "smallest label wins". It is deterministic but arbitrary, which is why odd k is preferred for two classes.

**Step 8: return the answers**

```python
return preds
```

`preds` has shape (n_query,): one label per query, in the same order as the rows of `Q`. For our example with k = 3 it is `[0, 0]`. That ordering is what lets the plotting code reshape the answers back onto the grid.

**All steps mapped to the algorithm**

| Step | Code | KNN idea | Shape after |
|---|---|---|---|
| 1 | `max_k`, `ValueError` | k must be possible | none |
| 2 | `squared_distances` | measure distance to every stored point | (n_query, n_train) |
| 3 | `fill_diagonal(..., inf)` | stop a point voting for itself | same |
| 4 | `argsort(axis=1)` | rank neighbors nearest to farthest | (n_query, n_train) |
| 5 | `order[:, :k]` | keep the k nearest | (n_query, k) |
| 6 | `y_train[nearest]` | look up their labels | (n_query, k) |
| 7 | `bincount` + `argmax` | majority vote | (n_query,) |

**Things to watch for**
- Labels must be non-negative whole numbers (0, 1, 2, ...), because `bincount` uses them as positions. String labels like "red" would fail.
- With very large data, `argsort` sorts every row fully, which is more work than needed. `np.argpartition` finds the k smallest without a full sort, at the cost of no ordering among them and no stable tie rule.
- The cost of one prediction is dominated by Step 2: one query is compared with every stored point. That is the "lazy" price we discussed in Stage 1.

**One question to check your understanding:** with the Q2 row `[18, 1, 8]`, labels `y = [0, 1, 0]`, and k = 1, what does the function return, and which steps produce that answer?