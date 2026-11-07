"""
logistic_regression_basics.py
Project 2 (Logistic Regression from scratch) - building blocks so far:
    1. make_data()       -> two overlapping Gaussian blobs
    2. sigmoid(z)        -> squash any number into (0, 1)
    3. predict_proba()   -> forecast P(y = 1 | x) for every row of X
    4. main()            -> tests all of the above

Run with:  python logistic_regression_basics.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
def make_data(n_per_class=100, seed=42):
    """Build a 2-feature binary dataset: two overlapping Gaussian blobs.

    Returns
    -------
    X : array of shape (2 * n_per_class, 2)
        Each row is one data point, each column is one feature.
    y : array of shape (2 * n_per_class,)
        Flat list of labels: n_per_class zeros, then n_per_class ones.
    """
    # Fix the random seed so you get the same data every time you run this.
    np.random.seed(seed)

    # Class 0 blob: points centered around (0, 0).
    # np.random.randn(n, 2) draws a table of random numbers with n rows and
    # 2 columns from the standard normal bell curve (mean 0, spread 1).
    # Adding [0, 0] shifts every row by 0 in each column (no movement).
    class0 = np.random.randn(n_per_class, 2) + np.array([0, 0])

    # Class 1 blob: same idea, but the + [2, 2] moves every point 2 units
    # right and 2 units up (NumPy broadcasting). The centers are close enough
    # that the two blobs overlap a bit.
    class1 = np.random.randn(n_per_class, 2) + np.array([2, 2])

    # Merge the two blobs into ONE table: each row is one data point, each
    # column is one feature. Class 0 rows come first, class 1 rows after.
    X = np.vstack([class0, class1])

    # Build matching labels in the SAME order as X: n zeros, then n ones.
    # y is a flat list (one label per row of X), so we join end to end with
    # concatenate instead of stacking like a table.
    # Row i of X must correspond to label y[i].
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

    return X, y


# ----------------------------------------------------------------------
# CORE FUNCTIONS
# ----------------------------------------------------------------------
def sigmoid(z):
    """Squash any real number (or array of numbers) into the range (0, 1).

    Formula: sigma(z) = 1 / (1 + e^(-z))

    np.exp works on a whole array at once, so if z has 200 numbers we get
    200 outputs with no loop needed.
    """
    return 1 / (1 + np.exp(-z))


def predict_proba(X, w, b):
    """Predicted probability that y = 1, for every row of X.

    Shapes
    ------
    X : (n, 2)   n rows (points), 2 columns (features)
    w : (2,)     one weight per feature
    b : float    a single bias number
    returns y_hat : (n,)   one probability per point, same shape as y
    """
    # Step 1: raw score z = w.x + b for EVERY row at once.
    # X @ w is a matrix-times-vector product. For each row it computes
    # x1*w1 + x2*w2, giving n numbers, so z has shape (n,).
    # Adding b adds the same number to all n scores (broadcasting).
    z = X @ w + b

    # Step 2: squash each score into a probability between 0 and 1.
    return sigmoid(z)

def log_loss(y, y_hat):
    # y:     true labels, shape (n,), each 0 or 1
    # y_hat: predicted probabilities, shape (n,), each between 0 and 1

    # Step 1: clip y_hat so it can never be exactly 0 or 1.
    # Without this, log(0) = -infinity and the loss becomes inf or nan.
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)

    # Step 2: per-row loss using the switch formula.
    # For y = 1 only the first term survives: -log(y_hat)
    # For y = 0 only the second term survives: -log(1 - y_hat)
    per_row = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    """
    It works on all elements of y and y-hat, because NumPy applies every operation element by element on whole arrays at once. 
    The loop still happens, but it runs inside NumPy (in fast compiled code) instead of in your Python code. This is called vectorization.
    """

    # Step 3: average over all n rows to get ONE number.
    return np.mean(per_row)

# Read the "compute_gradients_basics.md" file for detailed explanations ....
def compute_gradients(X, y, y_hat):
    # X:     shape (n, 2)  the data (n rows, 2 features)
    # y:     shape (n,)    true labels
    # y_hat: shape (n,)    predicted probabilities
    # Returns: dw with shape (2,) and db as a single number.

    n = X.shape[0]            # number of rows (200 in our data)

    # Step 1: the "blame report": one error per row.
    # Positive = model predicted too high, negative = too low.
    error = y_hat - y         # shape (n,)

    # Step 2: gradient for w.
    # X.T has shape (2, n). Multiplying by error (n,) gives shape (2,):
    # for feature j, it computes  sum over rows of  x[i, j] * error[i].
    # That is exactly the Σ xᵢ(ŷᵢ − yᵢ) from our formula, for both
    # features at once. Dividing by n turns the sum into an average.
    dw = (X.T @ error) / n    # shape (2,)

    # Step 3: gradient for b.
    # b moves z by 1 for every row (not by x), so we just average the errors.
    db = np.sum(error) / n    # single number

    return dw, db


"""
Dry run: your 3-row example, 1 feature, α = 0.1, starting at w = 0, b = 0

x = [1.4, -0.85, -2.2], y = [1, 0, 1]

                            Iteration 1	            Iteration 2
w, b at start	            0, 0	                0.0008, 0.0167
z = w·x + b	                [0, 0, 0]	            [0.018, 0.016, 0.015]
ŷ = σ(z)	                [0.5, 0.5, 0.5]	        [0.504, 0.504, 0.504]
error = ŷ - y	            [-0.5, 0.5, -0.5]	    [-0.496, 0.504, -0.496]
x * error	                [-0.70, -0.425, +1.10]	[-0.694, -0.428, +1.092]
dw = sum ÷ 3	            -0.025 ÷ 3 = -0.0083	-0.0303 ÷ 3 = -0.0101
db = mean(error)	        -0.5 ÷ 3 = -0.1667	    -0.1626
loss (recorded)	            0.6931	                0.6904
new w = w - 0.1·dw	        0.0008	                0.0018
new b = b - 0.1·db	        0.0167	                0.0329

Notice two things. The loss at the start is 0.6931 = -log(0.5), the “coin flip” value, and every row contributes the same amount. 
And the steps are small: w moved by about 0.001 and b by about 0.017. The step size is α * slope, so tiny slopes plus α = 0.1 give tiny steps.
"""
def train(X, y, lr, n_iters):
    """
    Gradient descent for logistic regression.
    X       : (n, 2) features
    y       : (n,)   labels, 0 or 1
    lr      : learning rate α (step size)
    n_iters : how many times to repeat the loop
    returns : final w (2,), final b (float), losses (list of n_iters floats)
    """
    # Starting position of the knobs. np.zeros(2) gives array([0., 0.]),
    # one weight per feature. With w = 0 and b = 0, z = 0 for every row,
    # so every y_hat = 0.5 (a coin flip). Zeros are fine here because
    # the loss has a single valley; they are just where we start.
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []

    for i in range(n_iters):
        # 1. Predict: current knobs -> probabilities, shape (n,)
        y_hat = predict_proba(X, w, b)

        # 2. Measure: record the loss BEFORE updating, so losses[0] is
        #    the loss at the starting knobs. This list is the loss curve.
        losses.append(log_loss(y, y_hat))

        # 3. Slope: how the loss changes if each knob is nudged up.
        dw, db = compute_gradients(X, y, y_hat)

        # 4. Nudge: move opposite to the slope, scaled by lr.
        #    w and dw are both arrays of shape (2,), so this is element-wise:
        #    w = [1.0, 2.0], dw = [0.5, -0.5], lr = 0.1
        #    -> w - lr*dw = [1.0 - 0.05, 2.0 + 0.05] = [0.95, 2.05]
        w = w - lr * dw
        b = b - lr * db

    return w, b, losses


def plot_loss_curve(losses, lr):
    # plt.plot(losses) with one list uses the list positions
    # (0, 1, 2, ...) as the x values automatically,
    # so x = iteration number and y = loss at that iteration.
    plt.plot(losses)
    plt.xlabel("iteration")
    plt.ylabel("log-loss")
    plt.title(f"Loss curve, lr = {lr}")
    plt.show()


def plot_boundary(X, y, w, b, title=""):
    # Boolean mask: (y == 0) is an array of True/False, shape (200,).
    # X[y == 0] keeps only the rows where it is True, so we get
    # the class 0 points, shape (100, 2). Same idea for class 1.
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", label="class 0", alpha=0.6)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", label="class 1", alpha=0.6)

    # np.linspace(a, c, 100) gives 100 evenly spaced numbers from a to c.
    # Example: np.linspace(0, 1, 5) -> [0, 0.25, 0.5, 0.75, 1].
    # These are the x1 positions where we compute the line.
    x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

    # The formula from question 2: x2 = -(w1/w2)*x1 - b/w2.
    # x1_vals is an array, so this computes all 100 x2 values at once.
    x2_vals = -(w[0] / w[1]) * x1_vals - b / w[1]

    plt.plot(x1_vals, x2_vals, "k-", label="decision boundary")

    # Without this, a steep line can stretch the y-axis and squash the data.
    plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title(title)   
    plt.show()

# Usage, with the w and b from your training run:
# plot_boundary(X, y, w, b)


# ----------------------------------------------------------------------
# MAIN: test everything
# ----------------------------------------------------------------------
def main():
    # --- Test 1: data shapes -------------------------------------------
    X, y = make_data()
    print("Test 1: data shapes")
    print("  X.shape =", X.shape, "(expected (200, 2))")
    print("  y.shape =", y.shape, "(expected (200,))")
    print("  class counts: zeros =", int((y == 0).sum()),
          ", ones =", int((y == 1).sum()))
    print()



    # --- Test 2: sigmoid at known points -------------------------------
    # From Exercise A: sigma(0) = 0.5, sigma(2) ~ 0.881, sigma(-2) ~ 0.119,
    # and sigma(-z) = 1 - sigma(z).
    print("Test 2: sigmoid at known points")
    print("  sigmoid(0)  =", round(float(sigmoid(0)), 4), "(expected 0.5)")
    print("  sigmoid(2)  =", round(float(sigmoid(2)), 4), "(expected ~0.8808)")
    print("  sigmoid(-2) =", round(float(sigmoid(-2)), 4), "(expected ~0.1192)")
    z = np.array([-8.0, -2.0, 0.0, 2.0, 8.0])
    print("  works on arrays:", np.round(sigmoid(z), 4))

    # np.allclose(a, b) compares the two arrays element by element and returns True only if every pair is equal within a tiny tolerance.
    print("  symmetry sigmoid(-z) == 1 - sigmoid(z):",
          np.allclose(sigmoid(-z), 1 - sigmoid(z)))
    print()



    # --- Test 3: predict_proba on the tiny hand example ----------------
    # X = [[1, 2], [3, 0]], w = [0.5, -1], b = 0
    #   row 1: z = 1*0.5 + 2*(-1) = -1.5 -> sigmoid ~ 0.18
    #   row 2: z = 3*0.5 + 0*(-1) =  1.5 -> sigmoid ~ 0.82
    print("Test 3: predict_proba on the tiny hand example")
    X_small = np.array([[1.0, 2.0], [3.0, 0.0]])
    w_small = np.array([0.5, -1.0])
    print("  y_hat =", np.round(predict_proba(X_small, w_small, 0.0), 4),
          "(expected ~[0.1824 0.8176])")
    print()



    # --- Test 4: predict_proba on the real data with zero weights ------
    # With w = 0 and b = 0, z = 0 for every point, so every y_hat = 0.5.
    # (The model has no opinion yet.)
    print("Test 4: zero weights on the real data")
    """
    Why Test 4 uses zeros. predict_proba doesn’t find w and b, it only uses whatever w and b it’s given. 
    Zeros are the simplest starting point for training (the “before any learning” state), so Test 4 checks that the 
    function behaves correctly there: z = 0 for every row, so every ŷ = 0.5. It’s a useful test because we know the 
    exact answer in advance.
    """
    w = np.zeros(2)
    b = 0.0
    y_hat = predict_proba(X, w, b)
    print("  y_hat[:5] =", y_hat[:5])
    print("  y_hat.shape =", y_hat.shape, "(expected (200,))")
    print("  all values equal 0.5:", bool(np.allclose(y_hat, 0.5)))
    print()



    # --- Test 5: random w and b ----------------------------------------
    # w and b will normally be LEARNED by gradient descent (coming soon).
    # Here we just feed in arbitrary values to check that predict_proba
    # works for any w and b, not only zeros.
    print("Test 5: random w and b")
    rng = np.random.default_rng(0)
    w_rand = rng.normal(size=2)          # two random weights
    b_rand = float(rng.normal())         # one random bias
    y_hat_rand = predict_proba(X, w_rand, b_rand)
    print("  w =", np.round(w_rand, 3), ", b =", round(b_rand, 3))
    print("  shape ok:", y_hat_rand.shape == (200,))
    print("  all values strictly between 0 and 1:",
          bool(np.all((y_hat_rand > 0) & (y_hat_rand < 1))))
    # Recompute row 0 by hand, one number at a time, and compare.
    z0 = X[0, 0] * w_rand[0] + X[0, 1] * w_rand[1] + b_rand
    manual_0 = 1 / (1 + np.exp(-z0))
    print("  row 0 by hand vs function:", round(float(manual_0), 4),
          "vs", round(float(y_hat_rand[0]), 4),
          "| match:", bool(np.isclose(manual_0, y_hat_rand[0])))
    print()



    # --- Test 6: hand-picked w and b that roughly separate the blobs ----
    # Blob centers are (0, 0) and (2, 2). The line x1 + x2 = 2 sits between
    # them, which is w = [1, 1], b = -2 (z = x1 + x2 - 2).
    # A point is called class 1 if y_hat >= 0.5.
    # Gradient descent should end up somewhere near values like these.
    print("Test 6: hand-picked w and b vs zero weights")

    """
    This is a loop over a list with two items. Each item is a bundle of three things: a label for printing, a w, and a b. 
    First time round: name = “zero weights”, w_t = [0, 0], b_t = 0. Second time: name = “w=[1,1], b=-2”, w_t = [1, 1], b_t = -2. 
    It's a shortcut for running the same code twice with different settings.
    """
    for name, w_t, b_t in [("zero weights   ", np.zeros(2), 0.0),
                           ("w=[1,1], b=-2  ", np.array([1.0, 1.0]), -2.0)]:
        y_hat_t = predict_proba(X, w_t, b_t)
        accuracy = np.mean((y_hat_t >= 0.5) == (y == 1))
        print(f"  {name} -> accuracy = {accuracy:.2f}")
    print()



    # --- Test to Check log-loss ---------------------------------------------
    y_test = np.array([1, 0, 1])
    y_hat_test = np.array([0.8, 0.3, 0.1])
    print("log_loss: ", log_loss(y_test, y_hat_test))   # expect about 0.9608



    # --- Test to calculate Gradients ---------------------------------------------
    # The 3-row example data (1 feature, so X has shape (3, 1))
    X_t = np.array([[1.4], [-0.85], [-2.2]])
    y_t = np.array([1.0, 0.0, 1.0])

    # Current knob settings: w has one weight (one feature), b is a number
    w_t = np.array([1.0])
    b_t = 0.0

    # Step 1: our own code computes the predictions (no hand-typed values)
    y_hat_t = predict_proba(X_t, w_t, b_t)

    # Step 2: feed those predictions into the gradient function
    dw, db = compute_gradients(X_t, y_t, y_hat_t)

    print("y_hat =", np.round(y_hat_t, 4))
    print("dw =", dw, " db =", db)
    # expect dw ≈ [0.482], db ≈ -0.267, the same numbers as our hand calculation



    # --- Plot the data -------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", label="class 0", alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:red", label="class 1", alpha=0.7)
    plt.xlabel("feature 1 (x1)")
    plt.ylabel("feature 2 (x2)")
    plt.title("Two overlapping Gaussian blobs")
    plt.legend()
    plt.show()

    # --- Plot the Loss -------------------------------------------------
    w, b, losses = train(X, y, lr=0.1, n_iters=1000)
    print(losses[0], losses[100], losses[500], losses[-1])
    print(w, b)
    plot_loss_curve(losses, lr=0.1)


    # --- Experiments -------------------------------------------------
    # Each entry is (w, b, title). Edit or add your own.
    experiments = [
        (np.array([1.762, 1.576]), -3.352, "learned by training"),
        (np.array([1.0, 1.0]),     -2.0,   "hand-picked w=[1,1], b=-2"),
        (np.array([1.0, 1.0]),     -1.0,   "same w, b raised to -1"),
        (np.array([1.0, 1.0]),     -3.0,   "same w, b lowered to -3"),
        (np.array([2.0, 1.0]),     -3.0,   "w1 bigger: steeper line"),
        (np.array([1.0, 2.0]),     -3.0,   "w2 bigger: flatter line"),
        (np.array([-1.0, -1.0]),    2.0,   "signs flipped: same line as hand-picked"),
    ]

    for w, b, title in experiments:
        plot_boundary(X, y, w, b, title)   # each call opens one figure; close it to see the next



if __name__ == "__main__":
    main()

"""
Explanation of Comment of Test #06:

Step A: what the model computes. The model's raw score is z = w₁·x₁ + w₂·x₂ + b. Put in w = [1, 1] and b = -2:

z = 1·x₁ + 1·x₂ - 2 = x₁ + x₂ - 2

So this model just adds the two features and subtracts 2.

Step B: when does it say “class 1”? We call a point class 1 when ŷ ≥ 0.5. From Exercise A, σ(0) = 0.5 and the sigmoid goes 
up as z goes up, so ŷ ≥ 0.5 exactly when z ≥ 0. That means:

z ≥ 0 → x₁ + x₂ - 2 ≥ 0 → x₁ + x₂ ≥ 2

In everyday terms: think of two exam marks. Add them up. If the total is 2 or more, you're class 1 (red). 
If it's less than 2, you're class 0 (blue).

Step C: why 2? The blue blob is centered at (0, 0), so its total is 0 + 0 = 0. The red blob is centered at (2, 2), 
so its total is 2 + 2 = 4. The number 2 sits exactly halfway between 0 and 4, so it's a fair pass mark.

Step D: what is “the line x₁ + x₂ = 2”? It is the set of all points whose total is exactly 2, such as (2, 0), (1, 1), and (0, 2).
Draw them on the scatter plot and they form a diagonal fence between the blobs. Points on one side are class 1, 
and the other side is class 0. (We'll plot this fence properly in step 5 of Stage 2.2. For now it's just “the pass mark”.)

Check with three points:

Point	                z = x₁ + x₂ - 2	    ŷ = σ(z)	Called
(0, 0) blue center	        -2	            0.12	    class 0 (correct)
(2, 2) red center	        +2	            0.88	    class 1 (correct)
(1, 1) in the middle	     0	            0.50	    borderline

"""