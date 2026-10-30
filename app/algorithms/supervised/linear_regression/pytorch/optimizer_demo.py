import torch
import torch.optim as optim

b = torch.tensor(0.0, requires_grad=True)
optimizer = optim.SGD([b], lr=0.1)

loss = (7 - (6 + b)) ** 2
loss.backward()

print(f"before step: b = {b.item():.4f}, b.grad = {b.grad.item():.4f}")   # b = 0.0000, b.grad = -2.0000

optimizer.step()

print(f"after  step: b = {b.item():.4f}, b.grad = {b.grad.item():.4f}")   # b = 0.2000, b.grad = -2.0000