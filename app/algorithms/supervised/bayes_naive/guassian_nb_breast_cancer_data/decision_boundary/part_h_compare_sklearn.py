import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from my_gaussian_nb import MyGaussianNB
from plot_helpers import plot_boundary

here = os.path.dirname(os.path.abspath(__file__))

# Same data and same split as Parts F and G
data = load_breast_cancer()
cols = [list(data.feature_names).index("mean radius"),
        list(data.feature_names).index("mean texture")]
X = data.data[:, cols]
y = data.target_names[data.target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# Fit both models on the same training rows
mine = MyGaussianNB().fit(X_train, y_train)
theirs = GaussianNB().fit(X_train, y_train)

pred_mine = mine.predict(X_test)
pred_theirs = theirs.predict(X_test)

print("my accuracy      :", round(np.mean(pred_mine == y_test), 4))
print("sklearn accuracy :", round(np.mean(pred_theirs == y_test), 4))
print("predictions agree:", np.mean(pred_mine == pred_theirs))

# Compare the learned numbers too (sklearn calls variances var_)
print("means match      :", np.allclose(mine.means_, theirs.theta_))
print("variances match  :", np.allclose(mine.vars_, theirs.var_))
print("priors match     :", np.allclose(mine.priors_, theirs.class_prior_))

# Boundaries side by side
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
plot_boundary(mine, X_train, y_train, axes[0], "My GaussianNB")
plot_boundary(theirs, X_train, y_train, axes[1], "scikit-learn GaussianNB")
path = os.path.join(here, "part_h_compare.png")
fig.savefig(path, dpi=120, bbox_inches="tight")
print("saved to:", path)
plt.show()
