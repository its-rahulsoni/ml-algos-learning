"""
PROJECT 3: K-NEAREST NEIGHBORS FROM SCRATCH
===========================================

STEPS IMPLEMENTED IN THIS FILE (in the same order as the code below)
--------------------------------------------------------------------
 STEP 1  Get data: 2 features, 2 classes (labels 0 = red, 1 = blue).
 STEP 2  Split into training data and test data.
 STEP 3  Scale the features (min and max come from TRAINING data only).
 STEP 4  Distance table with broadcasting: every query vs every training
         point, with NO Python loop over training points. Shape (n_query, n_train).
 STEP 5  Sort each row with argsort (gives POSITIONS, not distances)
         and keep the first k positions.
 STEP 6  Look up the labels of those k neighbors and take a majority vote
         (np.bincount counts the votes, np.argmax picks the winner).
 STEP 7  Predict for many queries at once (this is knn_predict).
 STEP 8  Decision boundary: meshgrid -> ravel -> predict -> reshape -> contourf.
 STEP 9  Draw 3 panels side by side for k = 1, 5, 20 (plt.subplots(1, 3)).
 STEP 10 YOU look at the plots and write one sentence per plot
         (the code prints a reminder; the sentences must be yours).
 STEP 11 Compare against scikit-learn's KNeighborsClassifier.
 STEP 12 Test scenarios (run first, so you know the code is correct).

HOW TO RUN
----------
    python knn_from_scratch.py

To use your Project 2 data, edit load_data() (STEP 1) so it returns
X with shape (n_samples, 2) and y with shape (n_samples,) holding 0/1.

KEY SHAPES TO REMEMBER
----------------------
    X_train        (n_train, 2)      stored points (the "model")
    y_train        (n_train,)        their labels
    Q              (n_query, 2)      points we want predictions for
    sq_dists       (n_query, n_train)  the distance table
    order          (n_query, n_train)  positions sorted nearest -> farthest
    nearest        (n_query, k)        positions of the k nearest
    nearest_labels (n_query, k)        labels of the k nearest
    preds          (n_query,)          final answer for each query
"""

# numpy does all the array math (broadcasting, argsort, bincount).
import numpy as np
# matplotlib draws the plots.
import matplotlib.pyplot as plt
# Counter and math are used ONLY by the slow reference implementation in the tests.
from collections import Counter
import math


# =====================================================================
# STEP 1: GET DATA
# =====================================================================
def load_data(n_per_class=100, seed=0):
    """Make a 2-feature, 2-class dataset."""
    # A random number generator with a fixed seed, so you get the same data every run.
    rng = np.random.default_rng(seed)
    # Class 0: 100 points scattered around the location (0, 0). Shape (100, 2).
    X0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n_per_class, 2))
    # Class 1: 100 points scattered around (2, 2). The two clouds overlap, which makes k matter.
    X1 = rng.normal(loc=[2.0, 2.0], scale=1.0, size=(n_per_class, 2))
    # Stack the two clouds on top of each other into one array. Shape (200, 2).
    X = np.vstack([X0, X1])
    # Labels: 100 zeros followed by 100 ones. Shape (200,).
    y = np.array([0] * n_per_class + [1] * n_per_class)
    # Hand both back to the caller.
    return X, y


# =====================================================================
# STEP 2: TRAIN / TEST SPLIT
# =====================================================================
def split_data(X, y, test_fraction=0.25, seed=1):
    """Shuffle the rows, then cut them into a training part and a test part."""
    # Random number generator with a fixed seed so the split is repeatable.
    rng = np.random.default_rng(seed)

    """
    len(X) is the number of rows, so for example - 8.
    rng.permutation(8) takes the numbers 0 to 7 and returns them in random order. For example:
    shuffled = [5, 2, 7, 0, 3, 6, 1, 4]
    """
    shuffled = rng.permutation(len(X))

    """
    How many rows go to the test set.
    len(X) * test_fraction is 8 * 0.25 = 2.0, the number of rows we want in the test set.
    int(...) turns it into a whole number, because you can't take 2.5 rows. int() drops the decimals and doesn't round, so with 10 rows, 10 * 0.25 = 2.5 becomes 2.
    """
    n_test = int(len(X) * test_fraction)

    """
    The first n_test shuffled row numbers become the test rows.
    shuffled[:2] means “the first 2 items”, positions 0 and 1.
    That gives [5, 2].
    So rows 5 and 2 are our test rows. Because the list was shuffled, they are a random pick and not just the first 2 rows of the file.
    """
    test_idx = shuffled[:n_test]

    # All the remaining row numbers become the training rows.
    train_idx = shuffled[n_test:]

    # Use the row numbers to pull out the actual data.
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ==========================================================================
# STEP 3: FEATURE SCALING (min-max)
#
# This is done to bring all the features into a one same scale of 0 to 1.
# ==========================================================================
def fit_minmax(X_train):
    """Find the min and max of each feature column, using TRAINING data only."""
    # Smallest value in each column. Shape (2,).
    mins = X_train.min(axis=0)
    # Largest value in each column. Shape (2,).
    maxs = X_train.max(axis=0)
    # Hand both back; we reuse these same numbers for test data and grid points.
    return mins, maxs


"""
Note: Read "logic_min_max.md" file for more details on min_max concepts.

The concept behind min/max:

Each feature is measured in its own units, such as years for age and rupees for salary. Min-max scaling replaces every 
value with how far along its own range it sits:

the feature's smallest value becomes 0
its largest value becomes 1
a value halfway between them becomes 0.5

Formula: (value - min) / (max - min)

After that, every feature lives on the same 0 to 1 ruler, so none of them can dominate the distance just because 
its numbers are bigger.
"""
def apply_minmax(X, mins, maxs):
    """Rescale each feature to roughly the range 0..1 using the given mins and maxs."""
    # The width of each feature's range.
    span = maxs - mins
    # If a feature never changes, its span is 0; use 1 instead to avoid dividing by zero.
    span = np.where(span == 0, 1.0, span)
    # (value - min) / (max - min): (n, 2) minus (2,) broadcasts across every row.
    return (X - mins) / span


# =====================================================================
# STEP 4: THE DISTANCE TABLE (broadcasting, no loop over training points)
# Note: Read "logic_squared_distances.md" file for more details on this section.
# =====================================================================
def squared_distances(Q, X_train):
    """Return a table of squared Euclidean distances, shape (n_query, n_train).

    Row i, column j holds the squared distance from query i to training point j.
    We skip the square root because it never changes the ranking (25 < 36 means 5 < 6).
    """
    # Q[:, None, :] has shape (n_query, 1, 2): each query gets its own little block.
    # X_train[None, :, :] has shape (1, n_train, 2): all training points in one block.
    # Subtracting stretches the size-1 dimensions, so every query meets every
    # training point. Result shape: (n_query, n_train, 2) = difference per feature.
    diff = Q[:, None, :] - X_train[None, :, :]
    # Square every difference, then add the 2 features together.
    # axis=2 is the feature dimension, so it disappears: (n_query, n_train, 2) -> (n_query, n_train).
    return np.sum(diff ** 2, axis=2)


# =====================================================================
# STEPS 5, 6, 7: SORT, TAKE k NEAREST, VOTE, PREDICT MANY QUERIES
# =====================================================================
def knn_predict(X_train, y_train, Q, k, leave_one_out=False):
    """Predict a label (0 or 1) for every row of Q.

    leave_one_out=True is for evaluating on the training data itself: Q must be
    X_train, and each point is stopped from voting for itself.
    """
    # Number of stored training points.
    n_train = X_train.shape[0]
    # With leave_one_out each point loses one candidate (itself), so k can be at most n_train - 1.
    max_k = n_train - 1 if leave_one_out else n_train
    # Reject silly values of k early, with a clear message.
    if not (1 <= k <= max_k):
        raise ValueError(f"k must be between 1 and {max_k}, got {k}")

    # STEP 4: the distance table, shape (n_query, n_train).
    sq_dists = squared_distances(Q, X_train)

    # Only used when Q is X_train: a point's distance to itself is 0, so it would
    # always be its own nearest neighbor. Setting that diagonal to infinity removes it.
    if leave_one_out:
        np.fill_diagonal(sq_dists, np.inf)

    # STEP 5: argsort each row (axis=1). It returns POSITIONS of the training points
    # ordered from nearest to farthest, not the distances themselves.
    # kind="stable" means equal distances keep their original order (a fixed tie rule).
    order = np.argsort(sq_dists, axis=1, kind="stable")

    # Keep the first k columns of every row: the positions of the k nearest points.
    # ":k" means positions 0 .. k-1, which is exactly k items (no off-by-one).
    nearest = order[:, :k]

    # STEP 6: use the positions to fetch labels. Shape (n_query, k).
    nearest_labels = y_train[nearest]

    # Space for one answer per query.
    preds = np.empty(Q.shape[0], dtype=int)
    # Loop over the QUERIES (allowed). We never loop over training points.
    for i in range(Q.shape[0]):
        # Count how many times each label appears among query i's k neighbors.
        # Example: labels [0, 0, 1] -> counts [2, 1] (label 0 twice, label 1 once).
        counts = np.bincount(nearest_labels[i])
        # The position of the biggest count IS the winning label.
        # On a tie, argmax returns the smallest label (our fixed tie-break rule).
        preds[i] = np.argmax(counts)

    # One predicted label per query.
    return preds


def accuracy(y_true, y_pred):
    """Fraction of predictions that match the true labels."""
    # (y_true == y_pred) is a True/False array; its mean is the fraction of Trues.
    return float(np.mean(y_true == y_pred))


# =====================================================================
# STEP 8: DECISION BOUNDARY HELPERS
# Just NumPy and no MatPLotLib ....
# =====================================================================

# n = number of grid positions per axis, so the grid has n * n spots in total.
# A bigger n gives a smoother boundary but needs more distance calculations.
# pad = extra margin around the data (0.05 suits features scaled to 0..1).
# NOTE: The details on the GRID concept and the method params are present int he GOogle Doc file - "ML Algo - KNN" ....
def make_grid(X, n=200, pad=0.05):
    """Build a grid of spots covering the data, returned as xx, yy tables and a list Q."""

    # The grid must cover the data, plus a small margin so edge points are not cut off.
    # X[:, 0] is the first feature of every point; we find its smallest and largest value.
    # pad pushes both ends outward, so points at the edge don't sit on the plot border
    # and the colour is visible all around them.
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    # Same for the second feature.
    # Same as above but for the second feature (column 1).
    # Together the two ranges define the rectangle that the grid has to cover.
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    # linspace gives n evenly spaced numbers from start to stop, both ends included.
    # step = (stop - start) / (n - 1), because n points have n - 1 gaps between them.
    # Example with n = 3:
    #   xs: step = (1.1 - (-0.1)) / 2 = 0.6  ->  -0.1, -0.1 + 0.6 = 0.5, 0.5 + 0.6 = 1.1
    #       xs = [-0.1, 0.5, 1.1]
    #   ys: step = (0.9 - 0.1) / 2 = 0.4     ->  0.1, 0.1 + 0.4 = 0.5, 0.5 + 0.4 = 0.9
    #       ys = [0.1, 0.5, 0.9]
    # The middle value 0.5 is not from X. It is just the halfway point of each range.
    xs = np.linspace(x_min, x_max, n)
    ys = np.linspace(y_min, y_max, n)


    # meshgrid returns two tables of shape (len(ys), len(xs)):
    # xx holds the x of every spot, yy holds the y of every spot.
    """
    Example: For above xs and ys (calcualted in last step) - 
    xx = [[-0.1, 0.5, 1.1],           yy = [[0.1, 0.1, 0.1],
         [-0.1, 0.5, 1.1],                 [0.5, 0.5, 0.5],
         [-0.1, 0.5, 1.1]]                 [0.9, 0.9, 0.9]]
    """
    xx, yy = np.meshgrid(xs, ys)


    # ravel() unrolls each table row by row into a flat list; np.c_ pairs them
    # as two columns. Q has shape (n*n, 2): one row per spot, ready for knn_predict.
    """
    knn_predict wants a list with one point per row, not two tables.

    xx.ravel() reads the table row by row into a flat list: [-0.1, 0.5, 1.1, -0.1, 0.5, 1.1, -0.1, 0.5, 1.1]
    yy.ravel() does the same: [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9]
    np.c_[a, b] places the two lists side by side as two columns.

    Q = [[-0.1, 0.1],
        [ 0.5, 0.1],
        [ 1.1, 0.1],
        [-0.1, 0.5],
        [ 0.5, 0.5],
        [ 1.1, 0.5],
        [-0.1, 0.9],
        [ 0.5, 0.9],
        [ 1.1, 0.9]]         shape (9, 2) = (n*n, 2)
    """
    Q = np.c_[xx.ravel(), yy.ravel()]

    # Give back the tables (for drawing) and the list of spots (for predicting).
    # We return both layouts of the same spots: xx and yy (map layout, for matplotlib)
    # and Q (list layout, for knn_predict). The order of Q matches xx.ravel(), so the
    # predictions can later be reshaped back onto the map with preds.reshape(xx.shape).
    return xx, yy, Q


def predict_in_chunks(X_train, y_train, Q, k, chunk=5000):
    """Predict for many queries without building one giant (n_query, n_train, 2) array."""
    # Collect the answer of each chunk here.
    pieces = []
    # Walk through the queries 'chunk' rows at a time (a loop over queries, not training points).
    for start in range(0, Q.shape[0], chunk):
        # Predict for this slice of queries only.
        pieces.append(knn_predict(X_train, y_train, Q[start:start + chunk], k))
    # Glue the pieces back into one flat array, in the original order.
    return np.concatenate(pieces)


# =====================================================================
# STEP 9: THREE PANELS FOR k = 1, 5, 20
# =====================================================================
"""
The whole idea fits in four sentences:

1. Lay a fine net of points over the area where your data lives (make_grid).
2. Ask KNN “which class?” for every point in the net (predict_in_chunks).
3. Color each point by the answer, red or blue (reshape + contourf).
4. Drop your real data points on top (scatter).
"""
def plot_boundaries(X_train, y_train, X_test, y_test, ks=(1, 5, 20), n_grid=200,
                    save_as="knn_boundaries.png"):
    # Method's JOB 1: PICTURE ....
                    
    """Draw one decision-boundary panel per value of k and print train/test accuracy."""
    # The grid only depends on the data, not on k, so build it once.
    xx, yy, Q = make_grid(X_train, n=n_grid)

    # 1 row, len(ks) panels. sharex/sharey force identical axes so panels compare fairly.
    """
    # plt.subplots creates the empty canvas: fig is the whole page, axs are the frames on it.
    # Here: 1 row and len(ks) columns, so one frame per value of k, side by side.
    # sharex/sharey force the same axis ranges on every frame, so the panels compare fairly.
    # Nothing is drawn yet; later lines draw into each frame with ax.contourf, ax.scatter, etc.
    """
    """
    fig is the whole canvas, and there is exactly one.
    Each individual plot inside it is called an axes (matplotlib's official word). I called these “frames”.

    So plt.subplots(2, 3) gives 1 figure containing 6 axes.

    fig (the whole board)
    +--------+--------+--------+
    | axes   | axes   | axes   |   <- row 0
    | [0][0] | [0][1] | [0][2] |
    +--------+--------+--------+
    | axes   | axes   | axes   |   <- row 1
    | [1][0] | [1][1] | [1][2] |
    +--------+--------+--------+

    “Axes” is a confusing name, since it sounds like just the x and y lines. In matplotlib it means “one complete plot area: its axis lines, its tick numbers, and everything drawn in it”.

    1 -> A single Row ....
    len(ks) -> now of Columns in each Row ....
    """
    fig, axs = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4.8), sharex=True, sharey=True)

    # Loop over the values of k (not over training points).
    # zip -> pairs each panel with a k: (panel 1, k = 1), (panel 2, k = 5), (panel 3, k = 20). Everything below runs once per panel.
    for ax, k in zip(axs, ks):
        # Predict a label for every grid spot. Shape (n_grid * n_grid,).
        # KNN Algo implementation comes into the picture here ....
        preds = predict_in_chunks(X_train, y_train, Q, k)

        # Fold the flat answers back into the map layout. Using xx.shape avoids scrambling.
        """
        preds is a flat list of 40,000 values, but drawing needs a 200 * 200 table. xx.shape is (200, 200), so reshape folds the list back into the same layout as the grid. 
        Row by row it undoes the ravel() from make_grid.
        """
        Z = preds.reshape(xx.shape)

        # Paint the regions: label 0 area light red, label 1 area light blue.
        """
        contourf fills areas of a map by value. Z only contains 0s and 1s, so levels cuts it into two bands:
        from -0.5 to 0.5 is label 0, painted light red
        from 0.5 to 1.5 is label 1, painted light blue

        The -0.5, 0.5 and 1.5 are just cut points placed between the whole numbers. The border between the red and blue areas is the decision boundary.
        """
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#f4b6b6", "#b6cff4"])

        # Draw the training points on top, so you can see why the boundary sits where it does.
        """
        Draw the training points on top:

        X_train[:, 0] is all the x values and X_train[:, 1] is all the y values (the column slicing we covered earlier).
        np.where(y_train == 0, "red", "blue") builds a color list: red where the label is 0, blue otherwise.
        edgecolor="black" gives each dot an outline and s=22 sets its size.
        """
        ax.scatter(X_train[:, 0], X_train[:, 1],
                   c=np.where(y_train == 0, "red", "blue"), edgecolor="black", s=22)



        # Method's JOB 2: SCORES (just for the title) ....

        # Training accuracy: each point may not vote for itself (leave-one-out).
        train_acc = accuracy(y_train, knn_predict(X_train, y_train, X_train, min(k, len(X_train) - 1),
                                                 leave_one_out=True))

        # Test accuracy: accuracy on points the model has never stored.
        test_acc = accuracy(y_test, knn_predict(X_train, y_train, X_test, k))

        # A clear title for each panel.
        ax.set_title(f"k = {k}\nleave-one-out acc {train_acc:.2f} | test acc {test_acc:.2f}")

        # Axis labels so the plot is readable.
        ax.set_xlabel("feature 1 (scaled)")

    # Only the first panel needs the y label (the axes are shared).
    axs[0].set_ylabel("feature 2 (scaled)")
    # Stop labels from overlapping.
    plt.tight_layout()
    # Save the picture to a file.
    plt.savefig(save_as, dpi=120)
    # Show the window (skipped silently on machines with no display).
    plt.show()


# =====================================================================
# STEP 11: COMPARE WITH SCIKIT-LEARN
# =====================================================================
def compare_with_sklearn(X_train, y_train, X_test, ks=(1, 5, 20), n_grid=100):
    """Fit sklearn's KNN with the same k and measure how often it agrees with ours."""
    # sklearn may not be installed, so import inside a try block.
    try:
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError:
        # Tell the user and stop; the rest of the project still works without it.
        print("scikit-learn not installed, skipping comparison (pip install scikit-learn).")
        return

    # Build the same kind of grid we use for plotting (coarser, to be quick).
    xx, yy, Q = make_grid(X_train, n=n_grid)

    # Compare for each k.
    for k in ks:
        # algorithm="brute" makes sklearn compute all distances, like ours.
        model = KNeighborsClassifier(n_neighbors=k, algorithm="brute")

        # "Training" = storing the data, same as ours.
        model.fit(X_train, y_train)
        
        # Our predictions and sklearn's predictions on the test points.
        ours_test = knn_predict(X_train, y_train, X_test, k)
        theirs_test = model.predict(X_test)

        # Our predictions and sklearn's predictions on every grid spot.
        ours_grid = predict_in_chunks(X_train, y_train, Q, k)
        theirs_grid = model.predict(Q)

        # How many predictions differ in each case.
        test_diff = int(np.sum(ours_test != theirs_test))
        grid_diff = int(np.sum(ours_grid != theirs_grid))
        
        # Print a one-line report. Zeros mean a perfect match.
        print(f"  k={k:>2}: test mismatches = {test_diff}/{len(X_test)}, "
              f"grid mismatches = {grid_diff}/{len(Q)}")


# =====================================================================
# STEP 12: TEST SCENARIOS
# =====================================================================
# Keeps a running count of passed and failed checks.
_results = {"pass": 0, "fail": 0}


def check(name, condition):
    """Record and print one PASS/FAIL line."""
    # Count the result.
    _results["pass" if condition else "fail"] += 1
    # Show it to the user.
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")


def slow_reference_predict(X_train, y_train, Q, k):
    """A deliberately slow, loop-based KNN used ONLY to double-check the fast version."""
    # One prediction per query.
    out = []
    # Loop over every query point.
    for q in Q:
        # Pair each training point's distance with its position (loop over training points: fine here).
        pairs = [(math.dist(q, x), j) for j, x in enumerate(X_train)]
        # Sort by distance, then by position, mirroring the stable sort of the fast version.
        pairs.sort()
        # Labels of the k nearest.
        labels = [int(y_train[j]) for _, j in pairs[:k]]
        # Count votes.
        votes = Counter(labels)
        # Highest count wins; ties go to the smallest label (same rule as argmax).
        best = max(votes.values())
        out.append(min(lbl for lbl, c in votes.items() if c == best))
    # Same type as the fast version returns.
    return np.array(out)


def run_tests():
    """Run all test scenarios and report how many passed."""
    print("\n=== TEST SCENARIOS ===")

    # ---- Test 1: the lesson's first example (distances 2, 5, 1, 3; labels red, blue, red, blue)
    print("\nTest 1: the 4-point example from the lesson (query at the origin)")
    # Four stored points on the x-axis at exactly those distances from the query (0, 0).
    X1 = np.array([[2.0, 0.0], [5.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    # Labels red, blue, red, blue = 0, 1, 0, 1.
    y1 = np.array([0, 1, 0, 1])
    # A single query point at the origin, as a (1, 2) array.
    q1 = np.array([[0.0, 0.0]])
    # k=1: the nearest point is at distance 1.0 (red).
    check("k=1 -> red (0)", knn_predict(X1, y1, q1, 1)[0] == 0)
    # k=2: neighbors at 1.0 (red) and 2.0 (red) -> red.
    check("k=2 -> red (0)", knn_predict(X1, y1, q1, 2)[0] == 0)
    # k=3: neighbors at 1.0, 2.0, 3.0 = red, red, blue -> red wins 2 to 1.
    check("k=3 -> red (0)", knn_predict(X1, y1, q1, 3)[0] == 0)
    # k=4: 2 red vs 2 blue is a tie; our rule gives the smallest label, 0.
    check("k=4 tie -> smallest label (0)", knn_predict(X1, y1, q1, 4)[0] == 0)

    # ---- Test 2: the distance table from the lesson
    print("\nTest 2: distance table matches the hand calculation")
    # Two queries (2 rows) from the lesson.
    Q2 = np.array([[2.0, 0.0], [3.0, 3.0]])
    # Three stored points A, B, C from the lesson.
    X2 = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
    # What we calculated by hand: Q1 -> [4, 17, 2], Q2 -> [18, 1, 8].
    expected = np.array([[4.0, 17.0, 2.0], [18.0, 1.0, 8.0]])
    # Compute the table with broadcasting.
    table = squared_distances(Q2, X2)
    # The table must have one row per query and one column per training point.
    check("shape is (n_query, n_train) = (2, 3)", table.shape == (2, 3))
    # And the numbers must match the hand calculation.
    check("values equal the hand-calculated table", np.allclose(table, expected))

    # ---- Test 3: argsort / first k / labels / vote on the lesson table
    print("\nTest 3: argsort and voting steps on the lesson table")
    # Labels of A, B, C = red, blue, red.
    y3 = np.array([0, 1, 0])
    # argsort of each row, as we did by hand: Q1 -> [2, 0, 1], Q2 -> [1, 2, 0].
    order = np.argsort(table, axis=1, kind="stable")
    check("argsort gives [[2,0,1],[1,2,0]]", np.array_equal(order, np.array([[2, 0, 1], [1, 2, 0]])))
    # Labels of the 3 nearest: Q1 -> [0, 0, 1], Q2 -> [1, 0, 0].
    check("labels of nearest 3 are [[0,0,1],[1,0,0]]",
          np.array_equal(y3[order[:, :3]], np.array([[0, 0, 1], [1, 0, 0]])))
    # Both queries should be red with k=3.
    check("k=3 predictions are [0, 0]", np.array_equal(knn_predict(X2, y3, Q2, 3), np.array([0, 0])))

    # ---- Test 4: grid mechanics (meshgrid -> ravel -> reshape round trip)
    print("\nTest 4: meshgrid, ravel and reshape line up")
    # 4 x-values and 3 y-values: 12 spots; xx should have shape (3, 4) (rows follow y).
    xs, ys = np.arange(4), np.arange(3)
    xx, yy = np.meshgrid(xs, ys)
    check("xx has shape (3, 4)", xx.shape == (3, 4))
    # Flatten into 12 query rows, 2 columns.
    Qg = np.c_[xx.ravel(), yy.ravel()]
    check("Q has shape (12, 2)", Qg.shape == (12, 2))
    # Pretend answer number i is just i, fold it back into the grid.
    Z = np.arange(12).reshape(xx.shape)
    # Cell (i, j) of Z must belong to the spot (xx[i, j], yy[i, j]) = row i*4 + j of Q.
    ok = all(Z[i, j] == i * 4 + j and np.array_equal(Qg[i * 4 + j], [xx[i, j], yy[i, j]])
             for i in range(3) for j in range(4))
    check("every Z cell matches its spot in Q", ok)

    # ---- Test 5: scaling changes which neighbor is nearest
    print("\nTest 5: feature scaling (age 20-60, salary 20,000-100,000)")
    # Point A is 30 years older but has almost the same salary; point B has a similar age but +10,000 salary.
    Xs = np.array([[60.0, 50500.0], [31.0, 60000.0]])
    # Label A = 0, label B = 1.
    ys_ = np.array([0, 1])
    # Query: age 30, salary 50,000.
    qs = np.array([[30.0, 50000.0]])
    # Without scaling, salary swamps age, so A (label 0) looks nearest.
    check("unscaled: nearest is A (label 0)", knn_predict(Xs, ys_, qs, 1)[0] == 0)
    # Known ranges of the two features.
    mins, maxs = np.array([20.0, 20000.0]), np.array([60.0, 100000.0])
    # After scaling, both features count fairly, so B (label 1) becomes nearest.
    check("scaled: nearest is B (label 1)",
          knn_predict(apply_minmax(Xs, mins, maxs), ys_, apply_minmax(qs, mins, maxs), 1)[0] == 1)

    # ---- Test 6: fast version equals the slow loop-based reference
    print("\nTest 6: vectorized KNN equals a slow loop-based reference")
    # Random data with a fixed seed.
    rng = np.random.default_rng(42)
    Xr = rng.normal(size=(60, 2))
    yr = rng.integers(0, 2, size=60)
    Qr = rng.normal(size=(25, 2))
    # Try several values of k.
    for k in (1, 3, 7, 15):
        check(f"k={k}: fast == slow", np.array_equal(knn_predict(Xr, yr, Qr, k),
                                                     slow_reference_predict(Xr, yr, Qr, k)))

    # ---- Test 7: k = N predicts the majority class everywhere
    print("\nTest 7: k = N (use every point) predicts the majority class")
    # 7 zeros and 3 ones: majority class is 0.
    Xm = rng.normal(size=(10, 2))
    ym = np.array([0] * 7 + [1] * 3)
    # Predictions for 5 random queries should all be 0.
    check("all predictions are the majority label",
          np.all(knn_predict(Xm, ym, rng.normal(size=(5, 2)), k=10) == 0))

    # ---- Test 8: k=1 training accuracy and the self-neighbor trap
    print("\nTest 8: k=1 training accuracy and leave-one-out")
    # Overlapping clouds so the classes are not perfectly separable.
    Xo, yo = load_data(n_per_class=60, seed=3)
    # Each point is its own nearest neighbor (distance 0), so accuracy is 100%.
    check("k=1 on training data, self allowed -> accuracy 1.0",
          accuracy(yo, knn_predict(Xo, yo, Xo, 1)) == 1.0)
    # When a point may not vote for itself, accuracy drops below 100%.
    check("k=1 on training data, leave-one-out -> accuracy < 1.0",
          accuracy(yo, knn_predict(Xo, yo, Xo, 1, leave_one_out=True)) < 1.0)

    # ---- Test 9: invalid k is rejected
    print("\nTest 9: bad values of k raise an error")
    # k=0 makes no sense.
    try:
        knn_predict(Xm, ym, Xm[:1], 0)
        check("k=0 raises ValueError", False)
    except ValueError:
        check("k=0 raises ValueError", True)
    # k larger than the dataset makes no sense either.
    try:
        knn_predict(Xm, ym, Xm[:1], 11)
        check("k=11 (> n_train) raises ValueError", False)
    except ValueError:
        check("k=11 (> n_train) raises ValueError", True)

    # ---- Test 10: a clearly separated dataset is classified perfectly
    print("\nTest 10: two far-apart clusters are classified perfectly")
    # Cluster 0 near (0, 0) and cluster 1 near (10, 10); they cannot be confused.
    Xc = np.vstack([rng.normal(0, 0.5, size=(30, 2)), rng.normal(10, 0.5, size=(30, 2))])
    yc = np.array([0] * 30 + [1] * 30)
    # Query points sitting at the two cluster centers.
    check("queries at the centers get the right label",
          np.array_equal(knn_predict(Xc, yc, np.array([[0.0, 0.0], [10.0, 10.0]]), 5), np.array([0, 1])))

    # Final tally.
    print(f"\nTests passed: {_results['pass']}   failed: {_results['fail']}")
    return _results["fail"] == 0


# =====================================================================
# MAIN: RUN EVERYTHING IN ORDER
# =====================================================================
if __name__ == "__main__":
    # STEP 12 first: if the tests fail, nothing below can be trusted.
    all_ok = run_tests()
    # Stop with a clear message if anything failed.
    if not all_ok:
        raise SystemExit("Some tests failed. Fix them before looking at the plots.")

    # STEP 1: get the data.
    X, y = load_data()

    # STEP 2: split into train and test parts.
    X_train, X_test, y_train, y_test = split_data(X, y)

    # STEP 3: learn min/max from TRAINING data only, then scale both parts with the same numbers.
    mins, maxs = fit_minmax(X_train)
    X_train_s = apply_minmax(X_train, mins, maxs)
    X_test_s = apply_minmax(X_test, mins, maxs)

    # Quick look at the sizes, so you can check shapes against the notes at the top.
    print("\n=== DATA ===")
    print(f"  X_train {X_train_s.shape}, y_train {y_train.shape}, X_test {X_test_s.shape}")

    # STEPS 8 and 9: boundaries for k = 1, 5, 20 side by side.
    # The KNN algo execution happens inside this plotting method call ....
    print("\n=== DECISION BOUNDARIES (also saved to knn_boundaries.png) ===")
    plot_boundaries(X_train_s, y_train, X_test_s, y_test, ks=(1, 5, 20))

    # STEP 11: compare with scikit-learn.
    print("\n=== COMPARISON WITH SCIKIT-LEARN (0 mismatches = identical) ===")
    compare_with_sklearn(X_train_s, y_train, X_test_s)

    # STEP 10: this part is YOUR job. Do not skip it.
    print("\n=== YOUR TURN (Step 10) ===")
    print("Before reading anything else, look at the three panels and write ONE concrete")
    print("sentence per plot about the boundary: how smooth or jagged is it, are there small")
    print("islands around single points, and what happens to the overall shape?")
    print("Then tell me what you saw, and we continue with the bias-variance questions.")