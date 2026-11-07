import numpy as np
import matplotlib.pyplot as plt

# This is the implementation of the sigmoid function ....
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# np.linspace(start, stop, num) creates an array of `num` evenly spaced
# numbers from `start` to `stop`, with both endpoints included.
# Here: 200 points from -10 to 10, so the spacing between neighbours is
# 20 / 199 ≈ 0.1005. The result is a 1D NumPy array of shape (200,):
# [-10.0, -9.899, -9.799, ..., 9.899, 10.0]
# Why 200 points? A curve is drawn by joining dots with short straight
# segments. More points means the segments are shorter and the curve looks
# smooth. With only 5 points, you would see a jagged zigzag.
# Why -10 to 10? 
# Sigmoid is already almost 0 at z = -10 and almost 1 at z = 10, (Explanation for this point is below)
# so this range shows the whole "S" shape: both flat tails and the steep middle.
z = np.linspace(-10, 10, 200)

# Calls sigmoid on the whole array at once. This works without a loop because
# NumPy functions are vectorised: np.exp(-z) is applied to every element,
# giving an array of 200 values. The 1 + ... and 1 / ... then also act
# element by element (this is called broadcasting a scalar over an array).
# So s[i] = 1 / (1 + e^(-z[i])), and s has the same shape as z: (200,).
# s[i] is the model's "probability" if the linear output were z[i].
# Because z and s have matching lengths, plt.plot(z, s) can pair them up
# as 200 (x, y) points: (z[0], s[0]), (z[1], s[1]), and so on.
s = sigmoid(z)

plt.figure(figsize=(6, 4))
plt.plot(z, s, label="σ(z)")
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
plt.axvline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("z")
plt.ylabel("σ(z)")
plt.title("Sigmoid")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Check against our hand calculations
print("σ(0)  =", sigmoid(0))
print("σ(2)  =", sigmoid(2))
print("σ(-2) =", sigmoid(-2))
print("σ(2) + σ(-2) =", sigmoid(2) + sigmoid(-2))

"""
WHY SIGMOID IS ALMOST 0 AT z = -10 AND ALMOST 1 AT z = +10

Sigmoid: σ(z) = 1 / (1 + e^(-z))

Key idea:
    The term e^(-z) shrinks exponentially. Every time z increases by 1,
    e^(-z) gets divided by about 2.7 (the value of e). Exponential shrinking
    gets tiny very quickly.

Table of values:

    z     e^(-z)          σ(z) = 1 / (1 + e^(-z))
    ----  --------------  ------------------------
    0     1               0.5
    2     0.135           0.881
    5     0.0067          0.9933
    10    0.0000454       0.9999546
    20    0.000000002     0.999999998

Positive side (z large and positive):
    Past z = 5, e^(-z) is already almost 0, so the denominator is
    1 + (almost nothing), and σ ≈ 1 / 1 = 1.
    Going from z = 10 to z = 20 changes σ by only about 0.00005. On a plot,
    a change that small is invisible, so the curve looks like a flat line at 1.

Negative side (z large and negative):
    The same thing happens in reverse. For z = -10, e^(-z) = e^10 ≈ 22,026,
    so σ = 1 / 22,027 ≈ 0.0000454, which is almost 0.

Important detail:
    Mathematically, σ(z) never exactly reaches 0 or 1. It only gets so close
    that the difference is invisible. For very large |z|, the computer rounds
    the result to exactly 1.0 or 0.0.

Why this matters for the project:
    A "flat" curve means the slope is almost 0.
    The slope of sigmoid is σ(z) * (1 - σ(z)).
        At z = 0:   slope = 0.5 * 0.5 = 0.25   (steepest point)
        At z = -8:  σ ≈ 0.0003, so slope = ?   (to be computed)
    A slope near 0 means a weak gradient signal, which affects how well
    gradient descent can correct a model that is confidently wrong.
"""