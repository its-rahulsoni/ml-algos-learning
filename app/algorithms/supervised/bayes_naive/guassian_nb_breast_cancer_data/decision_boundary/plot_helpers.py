import numpy as np
import matplotlib.pyplot as plt

def plot_boundary(model, X, y, ax, title, step=0.1):
    """Draw the decision regions of a fitted 2-feature model, with data on top."""
    classes = np.unique(y)                                  # e.g. ['benign','malignant']

    # 1. Meshgrid: a rectangular grid of points covering the data (with a margin)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))     # both have shape (rows, cols)

    # 2. Flatten the grid into an (N, 2) table: one row per grid point
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 3. Predict on every grid point
    pred = model.predict(grid)

    # 4. Words -> numbers (0/1) so contourf can colour them, then reshape to grid shape
    pred_num = np.searchsorted(classes, pred)
    zz = pred_num.reshape(xx.shape)

    # 5. Filled regions, then the real points scattered on top
    ax.contourf(xx, yy, zz, levels=np.arange(len(classes) + 1) - 0.5,
                cmap="coolwarm", alpha=0.3)
    for k, c in enumerate(classes):
        ax.scatter(X[y == c, 0], X[y == c, 1], s=12, edgecolor="k",
                   linewidth=0.3, label=c)
    ax.set_title(title)
    ax.set_xlabel("mean radius")
    ax.set_ylabel("mean texture")
    ax.legend()
