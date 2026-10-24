# Softmax — Implementation, Numerical Stability, and Why It's Used

## Concept first: what softmax does

Softmax converts a vector of raw numbers ("logits") into a probability distribution — values that are all positive and sum to exactly 1. The formula for element `i` in a vector `x`:

softmax(x)_i = exp(x_i) / sum_j exp(x_j)

In words: exponentiate every value, then divide each by the total sum of all the exponentiated values. Larger input values get proportionally larger (but never negative or zero) output probabilities.

## Why the large-number test breaks it

`exp()` grows explosively fast. `exp(1000)` is astronomically large — far beyond what a standard floating-point number (float64) can represent (max representable is roughly `1.8 x 10^308`, and `exp(1000) ~= 10^434`). When a float64 tries to hold a number that large, it **overflows** to `inf`. And once you have `inf` values mixed into a division, you get `nan` ("not a number") — because operations like `inf / inf` are mathematically undefined.

## The fix: subtract the max first

Mathematically, subtracting a constant `c` from every element before exponentiating doesn't change the final softmax result at all — the `c`'s cancel out in the division (this is a genuine mathematical identity, not just a trick). But numerically, it changes everything: if you pick `c = max(x)`, then the largest value becomes `exp(0) = 1`, and every other value becomes `exp(negative number)`, which shrinks toward 0 instead of exploding toward infinity. No more overflow.

## The code

```python
"""
Softmax From Scratch: Naive vs. Numerically Stable
====================================================
Demonstrates the overflow problem with a naive softmax implementation
on large input values, then fixes it using the max-subtraction trick.
"""

import numpy as np


def softmax_naive(x):
    """Naive softmax: exponentiate directly, then normalize. Prone to overflow."""
    exp_values = np.exp(x)
    return exp_values / np.sum(exp_values)


def softmax_stable(x):
    """
    Numerically stable softmax: subtract the max value before exponentiating.
    Mathematically identical result to softmax_naive, but avoids overflow.
    """
    shifted_x = x - np.max(x)
    exp_values = np.exp(shifted_x)
    return exp_values / np.sum(exp_values)


def run_demo():
    small_vector = np.array([1.0, 2.0, 3.0])
    large_vector = np.array([1.0, 2.0, 1000.0])

    print("=" * 60)
    print("Test 1: Naive softmax on a SMALL vector (should work fine)")
    print("=" * 60)
    print("input:", small_vector)
    result_small_naive = softmax_naive(small_vector)
    print("naive softmax output:", result_small_naive)
    print("sum of output (should be 1.0):", np.sum(result_small_naive))

    print("\n" + "=" * 60)
    print("Test 2: Naive softmax on a LARGE-number vector (watch for nan/inf)")
    print("=" * 60)
    print("input:", large_vector)
    result_large_naive = softmax_naive(large_vector)
    print("naive softmax output:", result_large_naive)
    print("contains nan:", np.any(np.isnan(result_large_naive)))
    print("contains inf:", np.any(np.isinf(result_large_naive)))

    print("\n" + "=" * 60)
    print("Test 3: Stable softmax on the SAME large-number vector")
    print("=" * 60)
    print("input:", large_vector)
    result_large_stable = softmax_stable(large_vector)
    print("stable softmax output:", result_large_stable)
    print("sum of output (should be 1.0):", np.sum(result_large_stable))
    print("contains nan:", np.any(np.isnan(result_large_stable)))
    print("contains inf:", np.any(np.isinf(result_large_stable)))

    print("\n" + "=" * 60)
    print("Test 4: Confirm naive and stable versions AGREE on the small vector")
    print("=" * 60)
    result_small_stable = softmax_stable(small_vector)
    print("naive result: ", result_small_naive)
    print("stable result:", result_small_stable)
    print("match (within tolerance):", np.allclose(result_small_naive, result_small_stable))


if __name__ == "__main__":
    run_demo()
```

### Output when run

```
============================================================
Test 1: Naive softmax on a SMALL vector (should work fine)
============================================================
input: [1. 2. 3.]
naive softmax output: [0.09003057 0.24472847 0.66524096]
sum of output (should be 1.0): 1.0

============================================================
Test 2: Naive softmax on a LARGE-number vector (watch for nan/inf)
============================================================
input: [   1.    2. 1000.]
naive softmax output: [ 0.  0. nan]
contains nan: True
contains inf: False

============================================================
Test 3: Stable softmax on the SAME large-number vector
============================================================
input: [   1.    2. 1000.]
stable softmax output: [0. 0. 1.]
sum of output (should be 1.0): 1.0
contains nan: False
contains inf: False

============================================================
Test 4: Confirm naive and stable versions AGREE on the small vector
============================================================
naive result:  [0.09003057 0.24472847 0.66524096]
stable result: [0.09003057 0.24472847 0.66524096]
match (within tolerance): True
```

NumPy also printed diagnostic warnings pointing directly at the naive version's broken lines:

```
softmax.py:13: RuntimeWarning: overflow encountered in exp
  exp_values = np.exp(x)
softmax.py:14: RuntimeWarning: invalid value encountered in divide
  return exp_values / np.sum(exp_values)
```

## Line-by-line explanation

### `softmax_naive(x)`

```python
def softmax_naive(x):
    exp_values = np.exp(x)
    return exp_values / np.sum(exp_values)
```
`np.exp(x)` applies the exponential function element-wise to every value in `x` — no loop needed, NumPy vectorizes this automatically. `np.sum(exp_values)` adds up all the exponentiated values into a single scalar (the denominator from the formula). Dividing the array by that scalar broadcasts the single number across every element, giving the final probabilities.

### `softmax_stable(x)`

```python
def softmax_stable(x):
    shifted_x = x - np.max(x)
    exp_values = np.exp(shifted_x)
    return exp_values / np.sum(exp_values)
```
`np.max(x)` finds the single largest value in the vector (a scalar). `x - np.max(x)` subtracts that scalar from every element — broadcasting again, this time a plain scalar against an array (the simplest broadcasting case: a 0-dimensional value applies everywhere). After this shift, the largest element becomes exactly `0`, and every other element becomes negative (or zero, if there are ties for the max). The rest of the function is identical to the naive version — same exponentiate-and-normalize logic, just operating on the shifted values instead of the raw ones.

## Walking through what happened in each test

**Test 1** (`[1, 2, 3]`, naive): worked fine — output `[0.090, 0.245, 0.665]`, summing to `1.0`. Small numbers pose no overflow risk; `exp(3) ~= 20`, nowhere near float64's limits.

**Test 2** (`[1, 2, 1000]`, naive): this is where it broke. `exp(1000)` overflowed to `inf` — visible in NumPy's own warning: `RuntimeWarning: overflow encountered in exp`. Then, when computing the sum, `inf` plus finite numbers is still `inf`, so the denominator became `inf`. Dividing `exp(1)` and `exp(2)` (both small, finite numbers) by `inf` gives `0` — that's why the first two output values show as `0.`. But dividing `inf` (from `exp(1000)`) by `inf` (the sum) is mathematically undefined (`inf/inf`), which IEEE floating-point arithmetic represents as `nan` — hence the second warning, `invalid value encountered in divide`, and the final `nan` in the output. The `contains nan: True` check confirms this directly.

**Test 3** (`[1, 2, 1000]`, stable): `np.max(x)` found `1000`, so `shifted_x = [1-1000, 2-1000, 1000-1000] = [-999, -998, 0]`. Now `np.exp()` only has to handle values <= 0: `exp(-999)` and `exp(-998)` are both *extremely tiny* (essentially 0, but representable — underflowing to exactly `0.0` is safe, unlike overflowing to `inf`, since floats handle "too small" much more gracefully than "too large"), and `exp(0) = 1` exactly. Summing gives approximately `1.0`, and dividing gives `[0, 0, 1]` — no `nan`, no `inf`, and it sums correctly to `1.0`. This makes intuitive sense too: when one value (1000) is overwhelmingly larger than the others (1, 2), softmax should essentially assign it ~100% of the probability mass — which is exactly what we got.

**Test 4** (small vector, both versions): confirms the mathematical claim from earlier — that subtracting the max doesn't change the *result*, only the numerical path used to get there. `np.allclose()` returned `True`, showing naive and stable versions agree (within floating-point tolerance) whenever the naive version doesn't actually break. This is the real proof that the fix is "free" — it never changes correct answers, it only prevents incorrect ones (nan/inf) from happening in the first place.

One more detail worth knowing: `np.exp` in the stable version never even risks overflow now, since every shifted value is <= 0 — `exp()` of a non-positive number is always in the safe range `(0, 1]`, so there's no way for it to blow up regardless of how extreme your original input is. That's the real guarantee this fix provides, not just a patch for this one example.

---

# Where Softmax Is Used, and Why

## Where softmax is used in the ML world

**1. Multi-class classification — the output layer**
This is the single most common use. If you're building a model to classify an image as one of 10 digit classes (0-9), or a language model predicting the next word out of a 50,000-word vocabulary, the final layer of the network produces one raw number ("logit") per possible class. Softmax converts those raw numbers into a proper probability distribution — "73% confident this is a 7, 15% confident it's a 1, ..." — so you can actually interpret the output as a prediction with a confidence level, and so you can pick the most likely class (or examine the full distribution).

**2. Loss function during training — cross-entropy loss**
Almost universally, softmax is paired with **cross-entropy loss**. During training, the model produces logits, softmax turns them into probabilities, and cross-entropy loss compares that probability distribution against the true label (usually one-hot encoded) to measure how wrong the prediction was. This pairing is so common it's often bundled into a single function (`softmax_cross_entropy`) in ML libraries for numerical stability reasons very similar to what we just fixed.

**3. Attention mechanisms in Transformers (the architecture behind GPT, BERT, Claude, etc.)**
Inside every attention layer, the model computes a set of raw "attention scores" indicating how much each word/token should attend to every other token. Softmax converts these scores into weights that sum to 1 — literally determining "how much attention to pay to each other token" as a weighted average. This is one of the most consequential uses of softmax in modern AI — it runs inside every layer of every large language model, many times per forward pass.

**4. Reinforcement learning — policy outputs**
When an RL agent needs to choose among several possible actions, a "policy network" often outputs a softmax distribution over actions — turning learned preferences into action-selection probabilities (allowing for exploration, rather than always deterministically picking the "best" action).

**5. Anywhere you need to convert arbitrary real numbers into a probability distribution over discrete categories** — this is the general pattern; the specific applications above are just the most common instances of it.

## Why do we even need to calculate this?

Neural networks internally just do linear algebra — matrix multiplications, additions — which naturally produce arbitrary real numbers (could be negative, could be huge, could be tiny). These raw outputs ("logits") have no inherent meaning as probabilities. But for classification, we specifically need:
- **Non-negative values** (a probability can't be negative)
- **Values that sum to exactly 1** (probabilities across all possible outcomes must total 100%)
- Ideally, a transformation that **preserves the relative ordering** of the original scores (if the network was more "confident" about class A than class B, that should still hold after transformation) and is **smooth/differentiable** (essential, since neural networks learn via gradient descent — the loss function needs to be differentiable all the way through, including this transformation)

Raw logits satisfy none of these constraints out of the box. Softmax is the specific function that converts arbitrary real numbers into something with all these needed properties.

## The underlying logic: why exponent-divided-by-sum-of-exponents, specifically

You could imagine other ways to turn numbers into a distribution — why exponentials?

**1. Guarantees positivity, unconditionally**
`exp(x)` is **always** positive, for any real number `x` — even negative inputs. Contrast this with a naive approach like `x / sum(x)`: if any input is negative (very common for raw logits), this naive approach could produce a negative "probability," which is meaningless. Exponentiating first sidesteps this entirely, by construction — there's no failure case to guard against.

**2. Preserves ordering (monotonicity)**
`exp()` is a strictly increasing function — if `a > b`, then `exp(a) > exp(b)`, always. So the relative ranking of your original logits is never scrambled by this transformation; the class the network scored highest is still the class with the highest final probability.

**3. Amplifies differences — "winner takes most" behavior**
This is the more interesting, less obvious property. Exponential growth means that even modest differences in input get stretched into much larger differences in output. Compare logits `[1, 2, 3]` versus their softmax output `[0.09, 0.24, 0.67]` from our test earlier — the raw values only differ by a factor of 3, but the resulting probabilities differ by a factor of ~7. This "sharpening" behavior is actually desirable for classification: if the network is genuinely more confident about one class, softmax amplifies that confidence into a more decisive probability, rather than just weakly leaning toward it. Our large-number test (`[1, 2, 1000]` -> `[0, 0, 1]`) is this same property taken to its extreme — an overwhelming logit advantage becomes essentially 100% certainty.

**4. It's smooth and differentiable everywhere**
Because training relies on computing gradients (how much to nudge each weight), the function converting logits to probabilities needs a well-defined derivative everywhere. `exp()` is infinitely differentiable, and it turns out softmax's gradient has a particularly clean, elegant closed-form expression — which is part of why it became the standard choice mathematically, not just conceptually.

**5. A deeper connection — it's the smooth version of "just pick the max"**
If your only goal were "pick the single most likely class," you'd use `argmax` — just return the index of the largest logit, with 100% weight on that one and 0% on everything else. But `argmax` is a step function: it's not differentiable (a tiny nudge to the inputs either changes nothing, or causes a sudden jump), so it can't be used inside a network trained by gradient descent. Softmax is often described as a **"soft" version of argmax** — hence the name "soft-max" — it behaves *similarly* to argmax (the largest input dominates the output, as we saw), but does so smoothly and continuously, making it usable in gradient-based training. This is genuinely the origin of the name.

**6. Statistical/physical roots — the Boltzmann distribution**
This exact functional form (`exp(energy) / sum of exp(energies)`) predates machine learning by over a century — it comes from statistical mechanics, where it describes the probability of a physical system occupying a given energy state at a given temperature. Machine learning borrowed this form because it has exactly the mathematical properties needed (positivity, differentiability, ordering-preservation) — it wasn't invented from scratch for neural networks, but adopted because the underlying math already solved the same shape of problem.