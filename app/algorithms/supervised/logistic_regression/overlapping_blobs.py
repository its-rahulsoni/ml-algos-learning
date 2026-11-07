import numpy as np
import matplotlib.pyplot as plt

# Fix the random seed so you get the same data every time you run this.
np.random.seed(42)

n_per_class = 100  # number of points in each blob

# Class 0 blob: 100 points centered around (0, 0).
# np.random.randn(100, 2) gives a table of random numbers with
# 100 rows and 2 columns (mean 0, spread 1). Adding [0, 0] shifts the center.
class0 = np.random.randn(n_per_class, 2) + np.array([0, 0])

# Class 1 blob: same idea, but centered around (2, 2).
# The centers are close enough that the blobs overlap.
class1 = np.random.randn(n_per_class, 2) + np.array([2, 2])

# Stack the two blobs on top of each other into one big table of points.
# Merge the two blobs into ONE table: each row is one data point,
# each column is one feature. Class 0 rows come first, class 1 rows after.
X = np.vstack([class0, class1])

# Build the labels: 100 zeros (class 0) followed by 100 ones (class 1).
# Build matching labels in the SAME order as X: 100 zeros, then 100 ones.
# y is a flat list (one label per row of X), so we join end to end
# with concatenate instead of stacking like a table.
# Row i of X must correspond to label y[i].
y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

# Scatter plot, colored by class.
plt.figure(figsize=(6, 5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", label="class 0", alpha=0.7)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:red", label="class 1", alpha=0.7)
plt.xlabel("feature 1 (x₁)")
plt.ylabel("feature 2 (x₂)")
plt.title("Two overlapping Gaussian blobs")
plt.legend()
plt.show()