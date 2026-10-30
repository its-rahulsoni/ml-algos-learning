# pytorch_step4.py
# Same 4-house training as step 3, but torch.optim.SGD does the update and the zeroing.

# numpy for standardizing, torch for gradients and the optimizer
import numpy as np
import torch

# the 4 houses: age, bedrooms, plot area
X_raw = np.array([[10, 2, 800], [40, 3, 1200], [20, 3, 1600], [30, 4, 2000]], dtype=float)
# average and s of each column
means = X_raw.mean(axis=0)
stds = X_raw.std(axis=0)
# z-scores for every column
Z = (X_raw - means) / stds
# ones column in front, as a tensor of shape (4, 4)
X = torch.tensor(np.hstack([np.ones((4, 1)), Z]), dtype=torch.float32)
# prices as a column, shape (4, 1)
y = torch.tensor([[30.0], [45.0], [55.0], [70.0]])

# four weights at 0, tracked by autograd
theta = torch.zeros((4, 1), requires_grad=True)
# the optimizer: told WHICH tensors to update ([theta]) and the learning rate
optimizer = torch.optim.SGD([theta], lr=0.1)

# the loss before each step is stored here
loss_history = []
# repeat the step 500 times
for _ in range(500):
    # clear the old gradient (replaces theta.grad.zero_())
    optimizer.zero_grad()
    # the same loss as before
    loss = ((X @ theta - y) ** 2).mean()
    # autograd computes the gradient into theta.grad
    loss.backward()
    # update theta (replaces the no_grad block and theta -= lr * theta.grad)
    optimizer.step()
    # store the loss as a plain number
    loss_history.append(loss.item())

# loss at the start, after 1, 2 and 10 steps
print("loss history:", [round(loss_history[i], 2) for i in [0, 1, 2, 10]])
# final theta
print("theta after 500 steps:", theta.detach().flatten())