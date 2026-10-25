import os
import numpy as np
import matplotlib.pyplot as plt


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


# ---------------------------------------------------------------
# Core functions: general code, true for ANY dataset (no hand-worked numbers inside)
# ---------------------------------------------------------------

def build_design_matrix(x):
    """Ones column first, then x. Column order must match theta = [b, w]."""
    col_ones = np.ones(len(x))
    return np.column_stack((col_ones, x))


def check_shapes(x, X, y):
    """X must be (n, 2) and y must be a column (n, 1). n comes from the data."""
    n = len(x)
    assert X.shape == (n, 2), f"X should be ({n}, 2), got {X.shape}"
    assert y.shape == (n, 1), f"y should be ({n}, 1), got {y.shape}"


def compute_XtX_Xty(X, y):
    """The two known pieces of the normal equation: XᵀX and Xᵀy."""
    XtX = X.T @ X
    Xty = X.T @ y
    assert XtX.shape == (2, 2), f"XtX should be (2, 2), got {XtX.shape}"
    assert Xty.shape == (2, 1), f"Xty should be (2, 1), got {Xty.shape}"
    return XtX, Xty


def solve_theta(XtX, Xty):
    """Solve XᵀXθ = Xᵀy for theta = [b, w]ᵀ."""
    # TODO (your words): why solve() is used instead of the inverse
    theta = np.linalg.solve(XtX, Xty)
    assert theta.shape == (2, 1), f"theta should be (2, 1), got {theta.shape}"
    assert np.allclose(XtX @ theta, Xty), f"theta does not satisfy the equation:\n{theta}"
    return theta


def predict(X, theta):
    """ŷ = X @ θ"""
    return X @ theta


def compute_residuals(y_hat, y):
    """residual = ŷ - y"""
    return y_hat - y

def compute_loss(residuals):
    """L = average of squared residuals (MSE)."""
    return np.mean(residuals ** 2)

def fit_closed_form(x, y):
    """Whole fit in one call: x, y in, theta out. Reuse this on any dataset."""
    X = build_design_matrix(x)
    check_shapes(x, X, y)
    XtX, Xty = compute_XtX_Xty(X, y)
    return solve_theta(XtX, Xty)


# ---------------------------------------------------------------
# Demo steps for the three points (1,2), (2,2), (3,5):
# prints and hand-worked asserts live here, one function per step
# ---------------------------------------------------------------

def create_arrays():
    """
    Step #01: We are creating required arrays ....
    """
    x = np.array([1, 2, 3])

    X = build_design_matrix(x)

    print("X: \n", X)

    y = np.array([
        [2],
        [2],
        [5]], dtype=float)

    print("y: \n", y)

    # Note: This is the java equivalent of printing multiple values in a print statement ....
    print("Shapes of each matrix. x: {}, y: {}, X: {}".format(x.shape, y.shape, X.shape))

    # Assertions ....
    check_shapes(x, X, y)

    return x, X, y


def do_matrix_multiplications(X, y):
    """
    Step #02: We are performing matrix multiplications ....
    """
    matrix_mul_a, matrix_mul_b = compute_XtX_Xty(X, y)
    print("matrix_mul_a: \n", matrix_mul_a, " ShapeA: ", matrix_mul_a.shape)
    print("matrix_mul_b: \n", matrix_mul_b, " ShapeB: ", matrix_mul_b.shape)

    # Assertions ....  (shape asserts now live inside compute_XtX_Xty)
    assert np.allclose(matrix_mul_a, [[3, 6], [6, 14]]), f"XtX values wrong:\n{matrix_mul_a}"
    assert np.allclose(matrix_mul_b, [[9], [21]]), f"Xty values wrong:\n{matrix_mul_b}"

    return matrix_mul_a, matrix_mul_b


def calculate_theta(matrix_mul_a, matrix_mul_b):
    """
    Step #03: We are calculating 'θ' from the Equation: XᵀXθ = Xᵀy ....
    """
    theta = solve_theta(matrix_mul_a, matrix_mul_b)
    print("theta: \n", theta, " theta shape: ", theta.shape)

    assert np.allclose(theta, [[0], [1.5]]), f"theta wrong:\n{theta}"

    return theta


def predict_and_check(X, y, theta, matrix_mul_a, matrix_mul_b):
    """
    X @ θ gives ŷ. How ??

    Explanation:
    θ = [b, w]ᵀ, so b is first and w is second. Row i of X is [1, xᵢ], so:

    first entry of the row (the 1) multiplies the first entry of θ (b)
    second entry of the row (xᵢ) multiplies the second entry of θ (w)

    Row times column means multiply the pairs and add:

    [1, xᵢ] · [b, w]ᵀ = 1·b + xᵢ·w = b + w·xᵢ

    With your three points, written out symbolically

    X = [[1,1],[1,2],[1,3]], θ = [b, w]ᵀ

    row 1: 1·b + 1·w = b + w
    row 2: 1·b + 2·w = b + 2w
    row 3: 1·b + 3·w = b + 3w

    So X @ θ = [b + w, b + 2w, b + 3w]ᵀ. That is exactly the model ŷ = b + w·x applied to x = 1, 2, 3, stacked as a column. 
    One matrix multiplication is the model applied to all three points at once.
    """
    x_thetha = predict(X, theta)
    print("x_thetha: \n", x_thetha, " x_thetha shape: ", x_thetha.shape)

    """
    # residual = ŷ - y  (prediction minus actual, one value per point)
    # It is the gap between the line and each real point.
    # Positive: the line is above the point. Negative: below.
    # For (1,2),(2,2),(3,5): ŷ = [1.5, 3, 4.5], y = [2, 2, 5]
    #   residuals = [-0.5, 1, -0.5]
    # Loss L = average of squared residuals = (0.25 + 1 + 0.25) / 3 = 0.5
    # At the best line, Xᵀ @ residuals = 0, meaning:
    #   Σ residual = 0  and  Σ x·residual = 0
    """
    residuals = compute_residuals(x_thetha, y)
    print("residuals: \n", residuals, " residuals shape: ", residuals.shape)

    check = X.T @ residuals
    print("check: \n", check, " shape: ", check.shape)

    assert np.allclose(check, 0)
    assert np.allclose(matrix_mul_a @ theta, matrix_mul_b)
    assert theta.shape == (2, 1)

def matches_numpy(x, y, theta):
    """
    Independent check: does my theta agree with np.polyfit on the same data?

    np.polyfit(x, y, 1) fits a straight line and returns [w, b] (slope first).
    [::-1] flips it to [b, w] so it lines up with my theta = [b, w].
    y.flatten() turns the (n, 1) column into a flat list, which polyfit expects.
    """
    numpy_theta = np.polyfit(x, y.flatten(), 1)[::-1]
    return np.allclose(theta.flatten(), numpy_theta)

def demo_exact_line():
    """
    Step #04: Exact line (0,5), (1,8), (2,11). No noise, so we know the answer: b = 5, w = 3.
 
    By hand:
      n = 3, Σx = 3, Σx² = 5, Σy = 24, Σxy = 0·5 + 1·8 + 2·11 = 30
      3b + 3w = 24
      3b + 5w = 30
      subtract: 2w = 6, so w = 3, and then 3b + 9 = 24, so b = 5
    Every point sits exactly on the line, so every residual should be 0 and the loss 0.

    -------------------------------------------------------------------------------
    1. How the two equations come from the five sums ?

    Answer: The five sums go into two fixed templates. These are the “setting both slopes to zero” equations from your notes:

    n·b + Σx·w = Σy
    Σx·b + Σx²·w = Σxy
    """
    x = np.array([0, 1, 2])
    y = np.array([[5], [8], [11]], dtype=float)
 
    # One call does the whole fit
    theta = fit_closed_form(x, y)
    print("exact line theta: \n", theta, " theta shape: ", theta.shape)
    assert np.allclose(theta, [[5], [3]]), f"theta wrong:\n{theta}"
 
    # Check against the original points
    X = build_design_matrix(x)
    residuals = compute_residuals(predict(X, theta), y)
    print("exact line residuals: \n", residuals)

    """
    2. Why every residual is 0 on this data ?

    A residual is ŷ - y, the prediction minus the actual value. Put b = 5 and w = 3 into ŷ = 5 + 3·x for each of your points:

    x = 0: ŷ = 5 + 0 = 5, and y = 5, so the residual is 0
    x = 1: ŷ = 5 + 3 = 8, and y = 8, so the residual is 0
    x = 2: ŷ = 5 + 6 = 11, and y = 11, so the residual is 0

    The loss is the average of the squared residuals, so it is (0 + 0 + 0) / 3 = 0.

    This is true because of the data, not because of the solver. I chose these three points by taking y = 5 + 3x at x = 0, 1, 2, 
    so one line passes through all of them exactly.

    -------------------------------------------------------------------------------
    The contrast with our first data:

    For (1,2), (2,2), (3,5), no straight line passes through all three points. The best line (ŷ = 1.5x) still misses by [-0.5, 1, -0.5], 
    and the loss is 0.5. The loss can never go below 0, because squares are never negative. So on the exact-line data, 0 is the lowest the 
    loss can possibly be.

    -------------------------------------------------------------------------------
    Where our numbers sit on that scale:

    Case	                                        Loss
    Exact-line data, best line ŷ = 5 + 3x	        0
    Your three points, best line ŷ = 1.5x	        0.5
    Your three points, wrong line ŷ = x	            5/3 ≈ 1.67

    Smaller is better, and 0 is the lowest value any loss can take.
    """
    loss = compute_loss(residuals)
    print("exact line loss: ", loss)

    assert np.allclose(residuals, 0)
    assert np.isclose(loss, 0)


def generate_noisy_data(n, noise_std, seed):
    """
    Make n points that come from the hidden line y = 3x + 5,
    with random noise added to each y.

    n          : how many points to make (for example 50)
    noise_std  : the standard deviation of the noise (for example 1)
    seed       : a fixed starting number, so the same seed gives the same data

    Returns x with shape (n,) and y with shape (n, 1).
    """
    # 1. Create the random number generator from the seed.
    rng = np.random.default_rng(seed)

    """
    2. n x values spread evenly between 0 and 10. Shape (n,).
    Read it as: “give me n numbers, each picked anywhere between 0 and 10, with every spot equally likely.” 
    Think of a spinner whose dial runs from 0 to 10, spun n times.

    With n = 3, you might get:
    x = [2.0, 7.5, 4.1]
    """
    x = rng.uniform(0, 10, n)
    #print("Inside generate_noisy_data, x: ", x)

    """
    3. n noise values, average 0, spread given by noise_std. Shape (n,).

    Read it as: “give me n random nudges. Most are small, a few are big, and they average out to 0.”

    The three inputs mean:

    0 is the average of the nudges. Using 0 means the noise pushes points up as often as down, so it does not shift the whole line upward or 
    downward. If you used 2 here, every y would drift up by about 2 and the true b would effectively become 7, which would spoil our test.
    noise_std is the spread. With noise_std = 1, the typical nudge is about 1 in size. About two thirds of the values fall between −1 and +1, 
    and almost all fall between −2 and +2.

    n is how many nudges, one per point.

    With n = 3 and noise_std = 1, you might get:

    noise = [+0.3, -0.8, +1.2]

    Some are positive and some negative, and none is huge. This is the “normal” (bell curve) shape:
    """
    noise = rng.normal(0, noise_std, n)
  #  print("Inside generate_noisy_data, noise: ", noise)

    # 4. The hidden line plus the noise. Shape (n,).
    y = 3 * x + 5 + noise

    # 5. Turn y into a column of shape (n, 1), as the rest of the code expects.
    y = y.reshape(n, 1)
   # print("Inside generate_noisy_data, y: ", y)

    # Checks, written in terms of n
    assert x.shape == (n,), f"x should be ({n},), got {x.shape}"
    assert y.shape == (n, 1), f"y should be ({n}, 1), got {y.shape}"

    return x, y


def plot_scatter(x, y):
    plt.scatter(x, y.flatten())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Noisy data from y = 3x + 5")
    plt.show()


def plot_fit(x, y, theta):
    b = theta[0, 0]
    w = theta[1, 0]

    """
    np.linspace(0, 10, 100) makes 100 evenly spaced x values from 0 to 10. We need it because a line is drawn by computing y at many x values 
    and joining them. It replaces picking x values by hand.
    """
    x_line = np.linspace(0, 10, 100)
    fitted_line = b + w * x_line
    true_line = 5 + 3 * x_line

    plt.scatter(x, y.flatten(), label="data")

    # plt.plot(x_line, y_line) draws a line through the points. label="..." names it, and linestyle="--" makes it dashed.
    plt.plot(x_line, fitted_line, color="red", label=f"fitted: b={b:.2f}, w={w:.2f}")
    plt.plot(x_line, true_line, color="black", linestyle="--", label="true: b=5, w=3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Fitted line vs true line")

    # plt.legend() shows the labels so you can tell the lines apart.
    plt.legend()
    plt.show()


def demo_noisy_fit(n=50, noise_std=1, seed=0):
    x, y = generate_noisy_data(n, noise_std, seed)
    theta = fit_closed_form(x, y)

    b = theta[0, 0]
    w = theta[1, 0]
    print(f"fitted b = {b:.4f}   (true b = 5)")
    print(f"fitted w = {w:.4f}   (true w = 3)")

    # Question 1: is my code correct?
    print("matches numpy:", matches_numpy(x, y, theta))

    # Question 2: did the fit recover the hidden line?
    print("b within 1 of 5:  ", abs(b - 5) <= 1)
    print("w within 0.2 of 3:", abs(w - 3) <= 0.2)

    # Drawing a graph of this noisy data ....
    plot_fit(x, y, theta)


def demo_seeds(n, noise_std):
    print(f"--- n = {n}, noise = {noise_std} ---")
    bs = []
    ws = []
    for seed in range(5):
        x, y = generate_noisy_data(n, noise_std, seed)
        theta = fit_closed_form(x, y)
        bs.append(theta[0, 0])
        ws.append(theta[1, 0])
        print(f"seed {seed}: b = {theta[0, 0]:.3f}, w = {theta[1, 0]:.3f}")
    print(f"b ranged from {min(bs):.3f} to {max(bs):.3f}")
    print(f"w ranged from {min(ws):.3f} to {max(ws):.3f}")


# ---------------------------------------------------------------
# main: the table of contents of the whole program
# ---------------------------------------------------------------

# Use this main method when we want to study the work we have done for Linear Regression using one set of random values using only '1' seed value ....
"""
def main():
    clear_screen()

    print("**** Ml Algo: Linear Regression with Equation: XᵀXθ = Xᵀy ****")

    x, X, y = create_arrays()
    matrix_mul_a, matrix_mul_b = do_matrix_multiplications(X, y)
    theta = calculate_theta(matrix_mul_a, matrix_mul_b)
    predict_and_check(X, y, theta, matrix_mul_a, matrix_mul_b)

    # One-call version must agree with the step-by-step version
    assert np.allclose(fit_closed_form(x, y), theta)

    # Verify theta using Numpy's built-in function ....
    theta_from_numpy = np.polyfit(x, y.flatten(), 1)
    assert np.allclose(theta.flatten(), np.polyfit(x, y.flatten(), 1)[::-1])

    demo_exact_line()

    # Using data with noise ....
    x, y = generate_noisy_data(n=50, noise_std=0, seed=1)
    print("x shape:", x.shape, " y shape:", y.shape)
    print("first 3 x:", x[:3])
    print("first 3 y:", y[:3].flatten())

    # Generating a graph on XY axis for the x and y values ....
    plot_scatter(x, y)

    # Noisy data demo ....
    demo_noisy_fit()
"""

# Use this main when we want to test our code with different set of random numbers using different seed values ....
def main():
    clear_screen()
    demo_seeds(n=50, noise_std=1)
    demo_seeds(n=50, noise_std=3)
    demo_seeds(n=10, noise_std=1)

"""
OUTPUT for this demo_seeds():

Setting	b range	w range
n = 50, noise 1 (baseline)	4.757 to 5.363, width 0.61	2.904 to 3.052, width 0.15
n = 50, noise 3	4.272 to 6.088, width 1.82	2.711 to 3.157, width 0.45
n = 10, noise 1	4.031 to 6.203, width 2.17	2.819 to 3.094, width 0.28

---------------------------------------------------------
What the numbers say:

More noise (noise 3): both widths grew by about 3 times (0.61 to 1.82 for b, 0.15 to 0.45 for w). Noise went up 3 times and the wobble went up 3 times. So the answer to the prediction question is 
“about 3 times wider”, not 10 times and not the same. The wobble grows in proportion to the noise.

Fewer points (n = 10): the wobble also grew, b by about 3.5 times (0.61 to 2.17) and w by about 2 times (0.15 to 0.28). Only 10 points means less cancelling of noise, so one unlucky point pulls the 
line harder.

Summary: 
“With baseline settings b moved by about 0.6 and w by about 0.15 across seeds, and they moved more when the noise was larger or the points fewer, because there is less cancelling of noise.”
"""

# Runs main() only when this file is executed directly, not when it is imported
if __name__ == "__main__":
    main()