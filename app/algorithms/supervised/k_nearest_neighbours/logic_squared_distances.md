# Understanding `squared_distances`

```python
def squared_distances(Q, X_train):
    """Return a table of squared Euclidean distances, shape (n_query, n_train).

    Row i, column j holds the squared distance from query i to training point j.
    We skip the square root because it never changes the ranking (25 < 36 means 5 < 6).
    """
    diff = Q[:, None, :] - X_train[None, :, :]
    return np.sum(diff ** 2, axis=2)
```

**The logic, briefly**

For KNN we need to know, for every query point, how far it is from every training point. By hand, for one query and one training point, that takes three moves:

1. Subtract the coordinates (feature by feature).
2. Square each difference.
3. Add the squared differences together.

That gives one squared distance. We skip the square root because it never changes the ranking: if 25 < 36, then 5 < 6.

The function does the same three moves for all pairs at once, with no loop, and returns a table: row i is query i, column j is training point j, and each cell is the squared distance between them.

**The example I'll use**

```
Q (2 queries):          X_train (3 stored points):
[2, 0]  <- Q1           [0, 0]  <- A
[3, 3]  <- Q2           [3, 4]  <- B
                        [1, 1]  <- C
```

Shapes: Q is (2, 2) and X_train is (3, 2).

**Line 1: `diff = Q[:, None, :] - X_train[None, :, :]`**

This is move 1 (subtract), for every pair at once. It happens in three small pieces.

Piece 1: `Q[:, None, :]` wraps each query in its own bracket, so Q goes from (2, 2) to (2, 1, 2):

```
[[[2, 0]],
 [[3, 3]]]
```

Piece 2: `X_train[None, :, :]` wraps all stored points in one outer bracket, so X_train goes from (3, 2) to (1, 3, 2):

```
[[[0, 0],
  [3, 4],
  [1, 1]]]
```

Piece 3: the subtraction. Numpy lines the shapes up from the right and stretches any dimension of size 1 by copying:

- Q is (2, 1, 2) and X is (1, 3, 2).
- Last dimension: 2 and 2, they match.
- Middle: 1 and 3, so each query is copied 3 times, once per stored point.
- First: 2 and 1, so the group of stored points is copied 2 times, once per query.
- Result: (2, 3, 2).

Read that shape as "2 queries, 3 stored points, 2 feature differences". The contents are every query minus every stored point:

```
diff[0]  (Q1 minus each point):      diff[1]  (Q2 minus each point):
  A: (2,0)-(0,0) = ( 2,  0)           A: (3,3)-(0,0) = (3,  3)
  B: (2,0)-(3,4) = (-1, -4)           B: (3,3)-(3,4) = (0, -1)
  C: (2,0)-(1,1) = ( 1, -1)           C: (3,3)-(1,1) = (2,  2)
```

Look at `diff[0]`, row B: that is the single number pair for "Q1 versus B". It is exactly the by-hand subtraction from move 1.

**Line 2: `np.sum(diff ** 2, axis=2)`**

This is moves 2 and 3, in two parts.

Part 1: `diff ** 2` squares every number, which is move 2. The shape stays (2, 3, 2):

```
[0]: (4, 0)   (1, 16)  (1, 1)
[1]: (9, 9)   (0, 1)   (4, 4)
```

Part 2: `np.sum(..., axis=2)` adds the two features within each pair, which is move 3. The `axis` you name is the one that disappears. Dimension 2 is the feature dimension, so the 2 features collapse into 1 number:

```
Q1:  4+0 = 4     1+16 = 17    1+1 = 2
Q2:  9+9 = 18    0+1  = 1     4+4 = 8
```

Final shape: (2, 3), which is (n_query, n_train).

```
        A    B    C
Q1  [   4,  17,   2 ]
Q2  [  18,   1,   8 ]
```

This is the same table we built by hand in Stage 3. For example, `Q1` to `B` is 17, because (2−3)² + (0−4)² = 1 + 16.

**Logic and code side by side**

| By hand | In the code |
|---|---|
| Subtract coordinates for one pair | `Q[:, None, :] - X_train[None, :, :]` does it for all pairs; shape (n_query, n_train, 2) |
| Square each difference | `diff ** 2`; shape unchanged |
| Add the features together | `np.sum(..., axis=2)`; shape (n_query, n_train) |
| Repeat for every query and stored point | Broadcasting: the size-1 dimensions get stretched, so no loop |

**Three things worth knowing**

1. The function isn't limited to 2 features. The "2" in the comments is just our example. Because `axis=2` is the last dimension, it works for any number of features, and `axis=-1` ("the last one") would make that explicit.
2. The wrong axis fails quietly. `axis=1` would add across stored points instead of across features. The shape would still look plausible, and the numbers would be meaningless.
3. Memory: `diff` holds n_query × n_train × n_features numbers. That's fine for a 150-point model and a few thousand queries, and it is why the plotting code predicts the grid in chunks.

**Quick check:** if Q has 10 rows, X_train has 50 rows, and there are 3 features, what shapes do `diff` and the final result have? Tell me your answer and the reasoning.