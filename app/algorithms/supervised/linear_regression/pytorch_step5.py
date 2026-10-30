# pytorch_step5.py
# PyTorch training on housing.csv, compared with the numpy version.
# Save this file in the SAME folder as linear_regression_house.py.

# numpy for arrays, torch for autograd and the optimizer
import numpy as np
import torch
# reuse the functions that already exist in linear_regression_house.py
from linear_regression_house import (make_housing_csv, load_data, split_train_test,
                                     fit_standardizer, apply_standardizer, add_ones,
                                     gradient_descent, compute_rmse, compute_r2)

# rebuild the same csv (same seed, same houses)
make_housing_csv("housing.csv")
# read it back
X, y = load_data("housing.csv")
# same 80/20 split as before
X_train, X_test, y_train, y_test = split_train_test(X, y)
# train-only averages and s
means, stds = fit_standardizer(X_train)
# standardized design matrices (ones column included)
Xz_train = add_ones(apply_standardizer(X_train, means, stds))
Xz_test = add_ones(apply_standardizer(X_test, means, stds))

# numpy arrays -> tensors; float64 keeps the numbers identical in precision to numpy
Xt = torch.tensor(Xz_train, dtype=torch.float64)
yt = torch.tensor(y_train, dtype=torch.float64)
# four weights at 0, float64, tracked by autograd
theta = torch.zeros((4, 1), dtype=torch.float64, requires_grad=True)
# plain gradient descent with lr 0.1, same as the numpy run
optimizer = torch.optim.SGD([theta], lr=0.1)

# loss before each step
history = []
# 500 steps, same as the numpy run
for _ in range(500):
    # clear old gradient
    optimizer.zero_grad()
    # loss on all 160 training houses
    loss = ((Xt @ theta - yt) ** 2).mean()
    # autograd gradient
    loss.backward()
    # update theta
    optimizer.step()
    # record the loss
    history.append(loss.item())

# tensor -> numpy array (detach drops tracking, .numpy() converts)
theta_torch = theta.detach().numpy()
# the numpy gradient descent from your file, same data, same lr, same steps
theta_numpy, hist_numpy = gradient_descent(Xz_train, y_train, lr=0.1, n_iters=500)

# print both answers
print("torch theta:", theta_torch.ravel().round(4))
print("numpy theta:", theta_numpy.ravel().round(4))
# largest difference between the two answers
print("max difference:", np.abs(theta_torch - theta_numpy).max())
# test scores using your existing functions
print("test RMSE", round(compute_rmse(Xz_test, y_test, theta_torch), 3),
      "test R2", round(compute_r2(Xz_test, y_test, theta_torch), 4))