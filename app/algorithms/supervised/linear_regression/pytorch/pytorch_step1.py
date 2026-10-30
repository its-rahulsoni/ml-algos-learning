# bring in PyTorch
import torch

"""
What is a Tensor ?
Answer: In PyTorch, a tensor is a specialized data structure used to store and manipulate numbers.
Think of it as the ultimate multi-dimensional grid for data. If you are familiar with NumPy, a tensor is almost identical 
to a NumPy array, but with two superpowers: it can run on a GPU to accelerate calculations, and it tracks mathematical 
operations automatically to help models learn.
"""

# one weight b, starting at 0, with autograd tracking switched on
"""
torch.tensor(0.0, requires_grad=True) makes a tensor, which is PyTorch’s version of a numpy array. Here it holds the single number 0.0. 
The requires_grad=True part is the important one. It tells PyTorch “I will want the slope of the loss with respect to this number, so keep a record of 
everything that happens to it.” The 0.0 has a decimal point because gradients only exist for decimal (float) numbers.
"""
b = torch.tensor(0.0, requires_grad=True)

# the four true prices as a tensor (like a numpy array)
y = torch.tensor([30.0, 45.0, 55.0, 70.0])
# loss = average of (prediction - truth) squared, here the prediction is just b
loss = ((b - y) ** 2).mean()
# walk backward through the recorded steps and compute the slope
loss.backward()
# print the loss and the slope PyTorch found for b
print(loss)
print(b.grad)