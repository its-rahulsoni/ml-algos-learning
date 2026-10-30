import torch
import torch.optim as optim

# 1. Setup the problem
x = torch.tensor(3.0)
y_true = torch.tensor(7.0)
b = torch.tensor(0.0, requires_grad=True)

# 2. Setup the optimizer (Learning rate 0.1)
optimizer = optim.SGD([b], lr=0.1)

print("🚀 Starting the training loop...")
print("-" * 50)

# 3. Run the loop 20 times
for epoch in range(1, 51):
    # Step 1: Forward Pass (Guess)
    y_pred = 2 * x + b
    
    # Step 2: Compute Loss (How wrong are we?)
    loss = (y_true - y_pred) ** 2
    
    # Step 3: Backward Pass (Wipe old gradients, calculate new ones)
    optimizer.zero_grad()
    loss.backward()
    
    # Step 4: Optimize (Turn the knob)
    """
    When you called optimizer.step(), it performed a simple mathematical formula behind the scenes to update the weight:
    New Weight = Old Weight - (Learning Rate * Gradient)
    
    Plugging in our numbers:
    New b = 0.0 - (0.1 * -2.0) = 0.2

    Read: optimizer_pytorch.md for more details on 'Optimizer'.
    """
    optimizer.step()
    
    # Print progress every 2 steps so it's easy to read
    if epoch % 2 == 0 or epoch == 1:
        print(f"Step {epoch:02d} | Prediction: {y_pred.item():.4f} | Loss: {loss.item():.4f} | Knob 'b': {b.item():.4f}")

print("-" * 50)
print(f"🎉 Training complete! Final value of b: {b.item():.2f}")
