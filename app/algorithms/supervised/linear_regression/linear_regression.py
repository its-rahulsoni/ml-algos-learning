import os
import numpy as np


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

# ---------------------------------------------------------------
# main: the table of contents of the whole program
# ---------------------------------------------------------------

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

def noise_data_testing():
    

# Runs main() only when this file is executed directly, not when it is imported
if __name__ == "__main__":
    main()