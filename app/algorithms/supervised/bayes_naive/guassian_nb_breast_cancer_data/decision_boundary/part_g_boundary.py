import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from my_gaussian_nb import MyGaussianNB
from plot_helpers import plot_boundary

# Folder where THIS script lives (not the folder you ran the command from)
here = os.path.dirname(os.path.abspath(__file__))

data = load_breast_cancer()
cols = [list(data.feature_names).index("mean radius"),
        list(data.feature_names).index("mean texture")]
X = data.data[:, cols]
y = data.target_names[data.target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

model = MyGaussianNB().fit(X_train, y_train)
print("test accuracy:", round(np.mean(model.predict(X_test) == y_test), 4))

fig, ax = plt.subplots(figsize=(7, 5))
plot_boundary(model, X_train, y_train, ax, "MyGaussianNB boundary (training data)")

# Save next to the script
save_path = os.path.join(here, "part_g_boundary.png")
fig.savefig(save_path, dpi=120, bbox_inches="tight")
print("saved to:", save_path)

# Show the plot in a window (close the window to end the program)
plt.show()