# ---------------------------------------------------------------
# gradient_descent.py
# Gradient descent for linear regression: model ŷ = b + w·x
# ---------------------------------------------------------------
import os

# NumPy: arrays and matrix maths
import numpy as np

# matplotlib: for the loss-curve plots
import matplotlib.pyplot as plt

# Reuse functions we already wrote and tested, instead of rewriting them.
# (Importing is safe because that file has the if __name__ == "__main__" guard.)
# If generate_noisy_data still prints x, noise and y, comment those prints out.
from linear_regression_noise_data import (
    build_design_matrix,   # x -> X (ones column + x column)
    generate_noisy_data,   # makes noisy data from y = 3x + 5
    fit_closed_form,       # normal-equation answer, used as the reference
    predict,               # ŷ = X @ θ
    compute_residuals,     # ŷ - y
    compute_loss,          # average of squared residuals
)

"""
What gradient descent is

The normal equation jumps straight to the bottom of the bowl. Gradient descent walks down it:

Start with a guess for θ = [b, w].
Compute the gradient, which tells you which way is uphill.
Take a small step the opposite way.
Repeat.

The update rule is θ_new = θ - α · gradient. Here α (the learning rate) is the step size. You choose it, and it is the one new knob in this stage. 
The gradient is the one you already know: (2/n) Xᵀ(Xθ - y).

# Gradient (2/n) Xᵀ(Xθ - y): where it comes from
#
# 1. Loss: L = (1/n) · Σ (ŷᵢ - yᵢ)², with ŷᵢ = b + w·xᵢ
# 2. Gradient = the slopes of L with respect to b and w
# 3. Slope of a square = 2 · (inside) · (how fast inside changes)
# 4. Slope for b: inside changes by 1 per unit of b
#       dL/db = (2/n) · Σ residualᵢ
# 5. Slope for w: inside changes by xᵢ per unit of w
#       dL/dw = (2/n) · Σ xᵢ · residualᵢ
# 6. These two sums are the two entries of Xᵀ @ residuals
#       (row of ones adds residuals, row of x adds x-weighted residuals)
# 7. Xθ - y = ŷ - y = residuals
# 8. So gradient = (2/n) · Xᵀ @ (Xθ - y), shape (2, 1)
# 9. Gradient = 0  gives  XᵀXθ = Xᵀy  (the normal equation)
# 10. Check: θ = [0, 0] on (1,2), (2,2), (3,5)
#       residuals = [-2, -2, -5], gradient = (2/3)·[-9, -21] = [-6, -14]
"""

# ---------------------------------------------------------------
# Core functions (general, true for any dataset)
# ---------------------------------------------------------------

# Gradient = the two slopes of the loss: how the loss changes if b or w is nudged.
# Formula: (2/n) · Xᵀ(Xθ − y). The 2 comes from squaring, the 1/n from averaging.
def compute_gradient(X, y, theta):
    # n = number of points; used for the 1/n of the average
    n = len(y)

    # Xθ gives the predictions ŷ, so Xθ − y is the residual of every point
    residuals = X @ theta - y

    # Xᵀ @ residuals gives Σ residual and Σ x·residual; scale by 2/n
    gradient = (2 / n) * (X.T @ residuals)
    """
    dL/db = (2/n) · Σ residualᵢ
    dL/dw = (2/n) · Σ xᵢ · residualᵢ

    What the gradient is --> Put those two derivatives in one column:
    gradient = [ dL/db ]  =  [ (2/n) · Σ residual     ]
               [ dL/dw ]     [ (2/n) · Σ x · residual ]
    """

    # One slope for b and one for w, stacked as a column
    assert gradient.shape == (2, 1), f"gradient should be (2, 1), got {gradient.shape}"
    return gradient


# Random starting guess for θ = [b, w]. Gradient descent needs somewhere to start.
def make_initial_theta(seed):
    # Same seed gives the same starting guess, so runs are repeatable
    rng = np.random.default_rng(seed)

    # Two small random numbers in the (2, 1) shape: one for b, one for w
    return rng.normal(0, 1, (2, 1))


# The loop: repeatedly step downhill using θ_new = θ − α · gradient
# lr is α (the learning rate = step size). n_iters is how many steps to take.
def gradient_descent(X, y, lr, n_iters, theta_init):
    # Copy so the caller's starting θ is not changed by our updates
    theta = theta_init.copy()

    # Will store the loss at every step, so we can plot how it falls
    loss_history = []

    # Repeat the same update n_iters times
    for i in range(n_iters):
        # Predictions with the current θ
        y_hat = predict(X, theta)

        # Gap between prediction and actual value, per point
        residuals = compute_residuals(y_hat, y)

        # Record the current loss (should shrink over time)
        loss_history.append(compute_loss(residuals))

        # Slopes of the loss at the current θ
        gradient = compute_gradient(X, y, theta)

        # The update rule: move opposite to the gradient, scaled by lr
        theta = theta - lr * gradient

    # Final θ and the loss at each step
    return theta, loss_history


# Draw several loss curves side by side in ONE window (one plt.show only)
def plot_loss_curves(histories, titles):
    # One panel per curve, in a single row
    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4))

    # If there is only one panel, make it a list so the loop below still works
    if len(histories) == 1:
        axes = [axes]

    # Draw each history on its own panel
    for ax, history, title in zip(axes, histories, titles):
        ax.plot(history)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.set_title(title)

        # Log scale on y so a huge blow-up and a tiny loss both stay visible
        ax.set_yscale("log")

    # Stop titles and labels overlapping
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# Demos (hand-worked numbers live here)
# ---------------------------------------------------------------

# Steps A, B, C, D on the three points (1,2), (2,2), (3,5)
def demo_three_points():
    # The three points, built exactly as before
    x = np.array([1, 2, 3])
    y = np.array([[2], [2], [5]], dtype=float)
    X = build_design_matrix(x)

    # Start at θ = [0, 0], where the paper gradient is [−6, −14]
    theta0 = np.zeros((2, 1))
    grad0 = compute_gradient(X, y, theta0)
    print("gradient at [0, 0]:\n", grad0)

    # Check against the hand calculation
    assert np.allclose(grad0, [[-6], [-14]]), f"gradient wrong:\n{grad0}"

    # Starting loss, should be 11 (paper: (4 + 4 + 25) / 3)
    start_loss = compute_loss(compute_residuals(predict(X, theta0), y))
    print("loss at [0, 0]:", start_loss)

    # ONE manual step with α = 0.1. Do this on paper FIRST, then compare.
    theta1 = theta0 - 0.1 * grad0
    new_loss = compute_loss(compute_residuals(predict(X, theta1), y))
    print("theta after one step:\n", theta1)
    print("loss after one step:", new_loss)

    # At the closed-form answer [0, 1.5] the gradient should be about 0
    theta_best = np.array([[0.0], [1.5]])
    assert np.allclose(compute_gradient(X, y, theta_best), 0), "gradient not 0 at best θ"

    # Full run: random start, lr 0.1, 1000 steps
    theta_gd, history = gradient_descent(
        X, y, lr=0.1, n_iters=1000, theta_init=make_initial_theta(seed=0)
    )
    print("gradient descent theta:\n", theta_gd)

    # Gradient descent gets close, not exact, so allow a small tolerance
    assert np.allclose(theta_gd, theta_best, atol=1e-3), f"not close to [0, 1.5]:\n{theta_gd}"

    return history


# Step F: noisy data, compare gradient descent with the closed form
def demo_noisy():
    # 50 noisy points from y = 3x + 5, seed 0
    x, y = generate_noisy_data(n=50, noise_std=1, seed=0)
    X = build_design_matrix(x)

    # The normal-equation answer is our reference
    theta_closed = fit_closed_form(x, y)

    # x runs 0 to 10 here, so the gradient is bigger: use a smaller lr (0.01)
    # and more steps, because the walk is slower
    theta_gd, history = gradient_descent(
        X, y, lr=0.01, n_iters=5000, theta_init=make_initial_theta(seed=0)
    )

    # Print both answers side by side
    print("closed form  b, w:", theta_closed.flatten())
    print("grad descent b, w:", theta_gd.flatten())

    # They should agree closely
    assert np.allclose(theta_gd, theta_closed, atol=1e-2), "gradient descent far from closed form"

    return X, y, history


# Step G: learning rate too high and too low, on the same noisy data
def demo_bad_learning_rates(X, y):
    # Same starting guess for every run, so only lr differs
    init = make_initial_theta(seed=0)

    # Good lr: loss falls steadily and flattens
    _, hist_good = gradient_descent(X, y, lr=0.01, n_iters=200, theta_init=init)

    # Too high: steps overshoot the bottom and the loss explodes
    _, hist_high = gradient_descent(X, y, lr=0.05, n_iters=50, theta_init=init)

    # Too low: steps are tiny and the loss crawls
    _, hist_low = gradient_descent(X, y, lr=0.0001, n_iters=200, theta_init=init)

    # Final loss of each run, to compare in numbers
    print("final loss, lr 0.01  :", hist_good[-1])
    print("final loss, lr 0.05  :", hist_high[-1])
    print("final loss, lr 0.0001:", hist_low[-1])

    # Show the three curves side by side
    plot_loss_curves(
        [hist_good, hist_high, hist_low],
        ["lr = 0.01 (good)", "lr = 0.05 (too high)", "lr = 0.0001 (too low)"],
    )

    """
    Final Output:
    lr	        Final loss	                Meaning
    0.01	    1.618 (after 200 steps)	    good, still falling
    0.05	    1.09 × 10⁴⁵	                too high, overshoots, loss explodes
    0.0001	    32.89	                    too low, crawls
    """


# ---------------------------------------------------------------
# main: the table of contents
# ---------------------------------------------------------------

def main():
    # Steps A to D: three points
    print("--- Three points ---")
    demo_three_points()

    # Step F: noisy data vs closed form
    print("--- Noisy data ---")
    X, y, history = demo_noisy()

    # Step E: loss curve for the noisy run
    plot_loss_curves([history], ["Loss curve, lr = 0.01"])

    # Step G: bad learning rates
    print("--- Learning rates ---")
    demo_bad_learning_rates(X, y)


# Run main() only when this file is executed directly
if __name__ == "__main__":
    main()

"""
A little note about Learning Rate:

Too high: “A learning rate that is too large makes each step overshoot the bottom, so the loss grows instead of shrinking and ends up astronomically large.”
Too low: “A learning rate that is too small moves in the right direction but so slowly that the loss barely drops in a reasonable number of steps.”`
"""