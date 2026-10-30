# numpy for the standardizing, torch for the gradient
import numpy as np
import torch

# the 4 houses: age, bedrooms, plot area
X_raw = np.array([[10, 2, 800], [40, 3, 1200], [20, 3, 1600], [30, 4, 2000]], dtype=float)
# train averages and s per column, exactly as fit_standardizer does
means = X_raw.mean(axis=0)
stds = X_raw.std(axis=0)
# z-scores for every column
Z = (X_raw - means) / stds
# put the ones column in front and turn the whole table into a tensor, shape (4, 4)
X = torch.tensor(np.hstack([np.ones((4, 1)), Z]), dtype=torch.float32)
# prices as a column, shape (4, 1)
y = torch.tensor([[30.0], [45.0], [55.0], [70.0]])
# all four weights start at 0, tracked, shape (4, 1)
theta = torch.zeros((4, 1), requires_grad=True)

# prediction X @ theta, residuals, squared, averaged: the same loss as before
loss = ((X @ theta - y) ** 2).mean()
# let autograd find the gradient
loss.backward()
# the gradient according to PyTorch
print("autograd:", theta.grad.flatten())

# your hand formula (2/n) Xt (X theta - y), using theta without tracking
grad_hand = (2 / 4) * X.T @ (X @ theta.detach() - y)
# the gradient according to your formula
print("by hand: ", grad_hand.flatten())