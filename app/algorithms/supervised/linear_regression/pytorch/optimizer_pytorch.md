# What `optimizer.zero_grad()` does behind the scenes

In one line: it clears the `.grad` field of every parameter the optimizer manages, so the next `loss.backward()` starts from a clean slate. To see why that matters, you need to know where gradients are stored and how `backward()` writes into them.

## 1. Where the gradient lives

Every tensor has a `.grad` attribute. For your `b` (created with `requires_grad=True`):

```python
b = torch.tensor(0.0, requires_grad=True)
print(b.grad)   # None. Nothing has been computed yet
```

`b.grad` is just a slot attached to the tensor object. It starts as `None`, and `loss.backward()` is what fills it.

## 2. How `backward()` writes into `.grad`

During the forward pass, PyTorch builds a graph: b → y_pred → loss. At the very end of each path, where a leaf tensor like `b` sits, PyTorch attaches a small internal node called **AccumulateGrad**. When `backward()` finishes the chain rule and arrives at `b`, that node runs this logic:

```
if b.grad is None:
    b.grad = incoming_gradient          # first time: just store it
else:
    b.grad = b.grad + incoming_gradient # otherwise: ADD to what is there
```

It says accumulate, not overwrite. This is deliberate:

- If a parameter is used in several places in the graph (shared weights), its gradient contributions from each place must be summed. This uses the same mechanism.
- If you want a large effective batch but can't fit it in memory, you can call `backward()` on several small batches and let the gradients pile up, then call `step()` once. This is called gradient accumulation.

The cost of this design is that PyTorch never clears gradients on its own. That is your job, and `zero_grad()` is how you do it.

## 3. What `zero_grad()` does, step by step

The optimizer stores your parameters in `optimizer.param_groups`, a list of dictionaries:

```python
optimizer.param_groups
# [{'params': [tensor(0., requires_grad=True)], 'lr': 0.1, 'momentum': 0, ...}]
```

This is the real reason you pass `[b]` into `optim.SGD(...)`. The optimizer needs to know which tensors it owns, so that `zero_grad()` can clear them and `step()` can update them.

Simplified, `zero_grad()` is basically this:

```python
def zero_grad(self, set_to_none=True):
    for group in self.param_groups:          # loop over parameter groups
        for p in group['params']:            # loop over each parameter (here just b)
            if p.grad is not None:
                if set_to_none:
                    p.grad = None            # default: throw the gradient away
                else:
                    p.grad.detach_()         # cut it from any graph
                    p.grad.zero_()           # fill the existing tensor with 0.0
```

There are two modes.

**`set_to_none=True` (the default since PyTorch 2.0).** The `.grad` tensor is discarded and the slot goes back to `None`. On the next `backward()`, AccumulateGrad sees `None` and simply stores the new gradient, with no addition at all. Benefits:

- It saves memory, because the old gradient tensor can be freed.
- It is faster, because no kernel has to write zeros into memory.
- The first write is a plain assignment instead of an add.

**`set_to_none=False`.** The old gradient tensor is kept and filled with zeros in place. The next `backward()` adds onto those zeros, which gives the same result.

One small gotcha: with the default mode, if you print `b.grad` right after `zero_grad()`, you get `None`, not `0.0`. Also, optimizers skip parameters whose `.grad` is `None`, which is normally what you want.

## 4. What `zero_grad()` does NOT touch

- It does not touch the computation graph. The graph is rebuilt from scratch on every forward pass, and it is freed automatically after `backward()` (unless you pass `retain_graph=True`).
- It does not touch the parameter values. `b` keeps its value, and only `b.grad` is cleared.
- It does not touch tensors the optimizer doesn't own. If you had other tensors with `requires_grad=True` that were not passed to the optimizer, their gradients would keep accumulating.

## 5. What happens if you forget it

Let me trace your exact problem without `zero_grad()`. The correct gradient at each step is 2(b − 1), but the gradients pile up:

| Step | New gradient | Accumulated b.grad | b after step() |
|------|--------------|--------------------|----------------|
| 1 | −2.00 | −2.00 | 0.200 |
| 2 | −1.60 | −3.60 | 0.560 |
| 3 | −0.88 | −4.48 | 1.008 |
| 4 | +0.016 | −4.464 | 1.454 |
| 5 | +0.91 | −3.555 | 1.810 |

At step 3, b has already reached the target (1.0), but the stale gradient from steps 1 and 2 is still stored in `b.grad`, so `step()` keeps pushing b upward. It overshoots to about 2, swings back, and oscillates around the target instead of settling. With `zero_grad()`, each step uses only the fresh gradient, so b converges smoothly to 1.0.

## 6. See it yourself

Run this small experiment to watch the accumulation happen:

```python
import torch

b = torch.tensor(0.0, requires_grad=True)

for i in range(3):
    loss = (7 - (6 + b)) ** 2     # same loss as yours, with b fixed at 0.0
    loss.backward()               # no zero_grad() on purpose
    print(f"after backward #{i+1}: b.grad = {b.grad.item()}")

# Prints -2.0, then -4.0, then -6.0 (each backward adds another -2.0)

b.grad = None                     # this is essentially what zero_grad() does
print(b.grad)                     # None
```

The gradient grows by −2 each time because the same gradient is added again on every call, since `b` never changes in this experiment.

## 7. Where it fits in the loop

Technically, `zero_grad()` can go anywhere as long as it runs after `step()` has consumed the old gradient and before the next `backward()`. Placing it just before `backward()`, as in your code, is a common and safe choice. The pattern is:

1. Clear the old gradients (`zero_grad`)
2. Compute fresh gradients (`backward`)
3. Use them to update the parameters (`step`)


--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

# What `optimizer.step()` does behind the scenes

In one line: it reads the gradient stored in each parameter's `.grad` and uses it to update the parameter's value in place. This is the only moment in the training loop where your parameter actually changes.

Where the other calls fit:

| Call | Job |
|------|-----|
| `loss.backward()` | Computes the gradient and stores it in `b.grad` |
| `optimizer.step()` | Reads `b.grad` and changes the value of `b` |
| `optimizer.zero_grad()` | Clears `b.grad` so the next backward starts fresh |

## 1. What it does for your code

Your optimizer is `optim.SGD([b], lr=0.1)` with no momentum and no weight decay. For this plain case, `step()` is just:

```
b = b − lr × b.grad
```

Here is the first iteration with real numbers:

- `b = 0.0`, and `loss.backward()` has stored `b.grad = −2.0`
- `step()` computes `b = 0.0 − 0.1 × (−2.0) = 0.2`

The minus sign matters. The gradient points in the direction where the loss goes UP, so subtracting it moves b in the direction where the loss goes DOWN. That is why it is called gradient descent. Here the gradient is negative (−2.0), so subtracting it increases b, which is exactly what we want, because the target needs b = 1.

## 2. What the code looks like inside

Simplified, `SGD.step()` is basically this:

```python
@torch.no_grad()                                  # (1) do not record these operations in the graph
def step(self, closure=None):
    loss = None
    if closure is not None:                       # (2) optional: re-run the model to get a loss
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:               # (3) loop over parameter groups
        lr = group['lr']
        for p in group['params']:                 # (4) loop over each parameter (here just b)
            if p.grad is None:                    # (5) skip parameters with no gradient
                continue

            grad = p.grad
            if group['weight_decay'] != 0:        # (6) optional: L2 penalty
                grad = grad + group['weight_decay'] * p

            if group['momentum'] != 0:            # (7) optional: momentum
                buf = self.state[p].get('momentum_buffer')
                if buf is None:
                    buf = grad.clone()            #     first step: buffer starts as the gradient
                else:
                    buf = group['momentum'] * buf + grad
                self.state[p]['momentum_buffer'] = buf
                grad = buf

            p.add_(grad, alpha=-lr)               # (8) the actual update: p = p − lr × grad, IN PLACE

    return loss
```

The real implementation has extra options (dampening, Nesterov, maximize, and faster "foreach" and "fused" code paths), but the structure is the same. For your simple optimizer, only steps (3), (4), (5) and (8) do any work.

## 3. The parts that are worth understanding

**(1) `@torch.no_grad()`.** The update itself is a math operation on a tensor that has `requires_grad=True`. Without `no_grad`, PyTorch would record the update as part of the computation graph, so the graph would grow every iteration and autograd would try to differentiate through your optimizer. The update is bookkeeping, not part of the model, so it is hidden from autograd.

**(5) Skipping `None` gradients.** If `p.grad` is `None`, the parameter is left untouched. This is why `zero_grad()` with `set_to_none=True` is safe: a parameter with no fresh gradient simply doesn't move. It is also why calling `step()` before any `backward()` does nothing.

**(8) In-place update with `add_`.** The trailing underscore in PyTorch means "modify this tensor in place". The tensor object `b` stays the same Python object with the same memory, and only its stored number changes. This is why the optimizer can change your variable `b` even though it never returns anything. It holds a reference to the same tensor you created, not a copy.

## 4. What `step()` does NOT do

- It does not compute gradients. That is `loss.backward()`, and `step()` only reads the result.
- It does not clear gradients. After `step()`, `b.grad` still holds the old value, and `zero_grad()` is what clears it.
- It does not recompute the loss or the prediction. The `y_pred` and `loss` you print were computed before `step()`, using the old b. This is why the printed b looks one step ahead of the printed prediction.

## 5. See it yourself

```python
import torch
import torch.optim as optim

b = torch.tensor(0.0, requires_grad=True)
optimizer = optim.SGD([b], lr=0.1)

loss = (7 - (6 + b)) ** 2
loss.backward()

print(f"before step: b = {b.item():.4f}, b.grad = {b.grad.item():.4f}")   # b = 0.0000, b.grad = -2.0000

optimizer.step()

print(f"after  step: b = {b.item():.4f}, b.grad = {b.grad.item():.4f}")   # b = 0.2000, b.grad = -2.0000
```

Notice that `b` changed but `b.grad` is unchanged. That confirms `step()` consumes the gradient without clearing it.

### Doing the same update by hand

You can reproduce `step()` yourself, which shows there is no magic:

```python
with torch.no_grad():          # required, see below
    b -= 0.1 * b.grad          # same as b = b − lr × b.grad
```

If you remove the `with torch.no_grad():` line, PyTorch raises this error:

```
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
```

That error is PyTorch protecting you from modifying a tracked tensor behind autograd's back. The optimizer avoids it by wrapping its work in `no_grad`.

## 6. The learning rate controls the step size

For your problem, one update gives (b − 1) → (b − 1) × (1 − 2 × lr), since the gradient is 2(b − 1). So the distance to the target is multiplied by (1 − 2 × lr) each step:

| lr | Factor per step | What happens |
|----|-----------------|--------------|
| 0.01 | 0.98 | Converges, but slowly |
| 0.1 | 0.8 | Converges smoothly (your setting) |
| 0.5 | 0.0 | Lands exactly on b = 1 in a single step |
| 1.0 | −1.0 | Jumps back and forth between 0 and 2 forever |
| 1.1 | −1.2 | Overshoots by more each time and blows up (diverges) |

This is the most important hyperparameter in training. Too small wastes time, and too large makes training unstable.

## 7. Tracing your loop

Each step shrinks the remaining error by a factor of 0.8:

| Step | b before | b.grad | Update (lr × grad) | b after |
|------|----------|--------|--------------------|---------|
| 1 | 0.000 | −2.000 | −0.200 | 0.200 |
| 2 | 0.200 | −1.600 | −0.160 | 0.360 |
| 3 | 0.360 | −1.280 | −0.128 | 0.488 |

After n steps, b = 1 − 0.8ⁿ. After 50 steps b ≈ 0.99998, which prints as 1.00.

## 8. What changes with other optimizers

All optimizers share the same interface (`zero_grad()` and `step()`). Only the update rule inside `step()` differs, and that is why you can swap optimizers by changing one line.

**SGD with momentum** (`optim.SGD([b], lr=0.1, momentum=0.9)`) keeps a running "velocity" in `optimizer.state[b]['momentum_buffer']`:

```
buf = momentum × buf + grad
b   = b − lr × buf
```

With your numbers: step 1 gives buf = −2.0 and b = 0.2. Step 2 gives buf = 0.9 × (−2.0) + (−1.6) = −3.4 and b = 0.2 + 0.34 = 0.54. Past gradients keep pushing b, which speeds up progress along consistent directions. The extra state stored per parameter is why `step()` is a method on an object rather than a plain function.

**Adam** (`optim.Adam([b], lr=0.1)`) keeps two running averages per parameter (of the gradient and of the squared gradient) and scales each parameter's step by them. This gives every parameter its own effective step size.

In all cases, `step()` follows the same pattern: read `.grad`, combine it with the optimizer's stored state and hyperparameters, and modify the parameter in place.

## 9. Summary

1. `loss.backward()` leaves a gradient in `b.grad`.
2. `optimizer.step()` loops over the parameters it owns and skips any whose `.grad` is `None`.
3. For each one, it computes a new value from the gradient (and any stored state such as momentum).
4. It writes that value into the parameter in place, hidden from autograd with `no_grad`.
5. It leaves `.grad` untouched, so `zero_grad()` must clear it before the next iteration.