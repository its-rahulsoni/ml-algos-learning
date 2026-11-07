# compute_gradients: A Dry Run From Scratch

Project 2: Logistic Regression from scratch, Stage 2.2 (Implementation).
This note explains what `compute_gradients(X, y, y_hat)` does, using a 3-row example worked by hand, and connects every line of code to the theory.

```python
def compute_gradients(X, y, y_hat):
    # X:     shape (n, features)  the data
    # y:     shape (n,)           true labels (0 or 1)
    # y_hat: shape (n,)           predicted probabilities
    n = X.shape[0]            # number of rows
    error = y_hat - y         # the "blame report": one error per row
    dw = (X.T @ error) / n    # gradient for w, shape (features,)
    db = np.sum(error) / n    # gradient for b, a single number
    return dw, db
```

---

## Part 1: Gradient basics (no formulas yet)

**The machine with two knobs.** Imagine a machine with two knobs, w and b. The model's quality depends on where the knobs are set. A display shows one number, the loss (how bad the forecasts are). Lower is better. The job is to turn the knobs until the display is as low as possible.

**What a gradient is.** For each knob, the gradient answers one question:

> "If I turn this knob up a tiny bit, does the display go up or down, and by how much?"

So there are two gradients, one per knob:

- ∂L/∂w: the effect on the loss of turning knob w up a little
- ∂L/∂b: the effect on the loss of turning knob b up a little

**How to read the sign:**

| Gradient | Meaning | What we do |
|---|---|---|
| positive | turning the knob up makes the loss worse | turn it down |
| negative | turning the knob up makes the loss better | turn it up |
| near zero | turning it barely matters | we are near the best setting |

That is why the update rule is w ← w − α·(gradient). The minus sign means "go the opposite way from where the loss rises". α (the learning rate) is how big a turn we make.

**The whole point of `compute_gradients`:** look at the current forecasts and produce numbers that tell us which way to turn each knob.

---

## Part 2: The 3 points

One feature, to keep it small. The model currently has w = 1 and b = 0, so z = 1·x + 0 = x, and ŷ = σ(z).

| Row | x | y (truth) | z = w·x + b | ŷ = σ(z) |
|---|---|---|---|---|
| 1 | 1.4 | 1 | 1.4 | 0.8 |
| 2 | −0.85 | 0 | −0.85 | 0.3 |
| 3 | −2.2 | 1 | −2.2 | 0.1 |

(ŷ is rounded for easy arithmetic.)

In code, the function receives:

- `X` = [[1.4], [−0.85], [−2.2]] (3 rows, 1 column)
- `y` = [1, 0, 1]
- `y_hat` = [0.8, 0.3, 0.1]

---

## Part 3: Dry run, line by line

### `n = X.shape[0]`

`X.shape` is (3, 1). The `[0]` picks the first number, which is the number of rows. So **n = 3**.

Why we need it: gradients are averages over rows, so we divide by 3 at the end.

### `error = y_hat - y`

Subtract row by row (the "blame report"):

| Row | ŷ | y | error = ŷ − y | What it says |
|---|---|---|---|---|
| 1 | 0.8 | 1 | **−0.2** | predicted a bit too low |
| 2 | 0.3 | 0 | **+0.3** | predicted a bit too high |
| 3 | 0.1 | 1 | **−0.9** | predicted way too low (confidently wrong) |

So `error` = [−0.2, 0.3, −0.9].

Sign meaning: negative = "the forecast needs to go up", positive = "the forecast needs to go down". Size = how wrong.

### `dw = (X.T @ error) / n`

This line means: **multiply each row's error by that row's x, add the results, then divide by n.**

| Row | x | error | x × error |
|---|---|---|---|
| 1 | 1.4 | −0.2 | 1.4 × (−0.2) = **−0.28** |
| 2 | −0.85 | 0.3 | −0.85 × 0.3 = **−0.255** |
| 3 | −2.2 | −0.9 | −2.2 × (−0.9) = **+1.98** |
| | | **Sum** | −0.28 − 0.255 + 1.98 = **1.445** |

Divide by n = 3: **dw = 1.445 / 3 ≈ 0.482**

Why multiply the error by x? Knob w only works through x. If a row has x = 0, turning w does nothing to that row, so that row should not push on w at all. A row with a big x is strongly affected by w, so its error counts more.

**What `X.T @ error` is, in plain terms.** It is a compact way to write this loop:

```python
total = 0
for i in range(n):
    total = total + X[i, 0] * error[i]
dw = total / n
```

`X.T` flips the table (3 rows × 1 column becomes 1 row × 3 columns), and `@` multiplies and adds, which is what the loop does.

### `db = np.sum(error) / n`

Knob b adds the same amount to every row's z, so it is not scaled by x (you can think of x as always being 1). We just add the errors and divide by n:

| Row | error |
|---|---|
| 1 | −0.2 |
| 2 | +0.3 |
| 3 | −0.9 |
| **Sum** | **−0.8** |

**db = −0.8 / 3 ≈ −0.267**

### Result

`return dw, db` gives **dw ≈ 0.482, db ≈ −0.267**.

---

## Part 4: Check that the gradient really is the slope

Claim: "the gradient tells you how the loss changes when you turn the knob up a little." Let's test it by actually turning the knobs.

| Setting | Loss (average log-loss over the 3 rows) |
|---|---|
| w = 1, b = 0 (current) | 0.9605 |
| w = 1.01, b = 0 (w up by 0.01) | 0.9653 |
| w = 1, b = 0.01 (b up by 0.01) | 0.9578 |

- Turning w up by 0.01 raised the loss by 0.0048, which is about 0.48 per unit of w. Our dw ≈ 0.482 says exactly that, and positive means "w up makes it worse".
- Turning b up by 0.01 lowered the loss by 0.0027, which is about −0.27 per unit of b. Our db ≈ −0.267 says exactly that, and negative means "b up makes it better".

So the gradient is the measured slope, computed by a formula instead of by trial and error.

---

## Part 5: Take the step

With α = 0.1, apply the update rule:

| Knob | Old | Gradient | Update: old − α × gradient | New |
|---|---|---|---|---|
| w | 1 | +0.482 | 1 − 0.1 × 0.482 | **0.952** |
| b | 0 | −0.267 | 0 − 0.1 × (−0.267) | **0.027** |

With the new knobs the loss goes from 0.9605 to 0.9306. It dropped, so one step of gradient descent worked. The training loop will repeat this many times.

---

## Part 6: Theory vs code, side by side

| Theory (formula) | Code | Plain English |
|---|---|---|
| number of rows n | `n = X.shape[0]` | count the rows |
| ŷ − y | `error = y_hat - y` | blame per row |
| Σ xᵢ(ŷᵢ − yᵢ) | `X.T @ error` | multiply each x by its row's error, add up |
| (1/n) Σ ... | `/ n` | turn the sum into an average |
| ∂L/∂w | `dw` | how the loss moves when w is nudged up |
| (1/n) Σ (ŷᵢ − yᵢ) | `np.sum(error) / n` | average blame, since b is not scaled by x |
| ∂L/∂b | `db` | how the loss moves when b is nudged up |

---

## Part 7: What changes with 2 features (our real data)

Each row has 2 numbers (x₁, x₂), and there are 2 w-knobs, one per feature. It is the same function and the same steps; the table just has 2 columns. Add a second feature to our 3 rows: x₂ = 1, 2, 3.

| Row | x₁ | x₂ | error | x₁ × error | x₂ × error |
|---|---|---|---|---|---|
| 1 | 1.4 | 1 | −0.2 | −0.28 | −0.2 |
| 2 | −0.85 | 2 | 0.3 | −0.255 | 0.6 |
| 3 | −2.2 | 3 | −0.9 | +1.98 | −2.7 |
| | | | **Sum** | 1.445 | −2.3 |

Divide by 3: **dw = [0.482, −0.767]**. `X.T @ error` produces both sums at once. Each w-knob gets its own gradient, using its own feature column and the same error column. db is unchanged (still −0.267) because the errors have not changed.

---

## One-sentence summary

The function computes the error for each row, multiplies it by how much each knob influences that row (x for w, 1 for b), and averages. The results tell us which way, and how strongly, to turn each knob.

---

## Q&A

### Question 1: Why does Row 3 dominate the gradient for w?

In the x × error column, Row 3 contributes +1.98, far more than the other two rows (−0.28 and −0.255). Name the two things about Row 3 that make its product so large.

**My answer (cleaned up):**
Its input was already large and also negative (however, I don't understand the difference between positive and negative inputs, and I want that explained). The error was also negative. When both are multiplied, the output is a large number with a positive sign, meaning that we should lower our effort for this input value.

**Feedback:**

What is right:

- Row 3 has a large input (−2.2, the biggest in size of the three) and a large error (−0.9, the biggest in size, since the model was confidently wrong).
- Negative × negative gives a positive product, and a large one, because both numbers are large.

What needed sharpening: "lower our effort for this input". A positive gradient means "turning w **up** makes the loss worse, so turn w **down**". It is a statement about the knob w, not about effort for one input. Row 3 contributes most of the push on that knob, because it is the row that is most wrong.

### Question 2: What is the difference between positive and negative inputs?

**Plain-English view.** The model computes z = w·x + b. Think of w as a "volume knob" that says how much a feature matters, and x as the evidence. A positive x is evidence above average, and a negative x is evidence below average. (Our blob data has negative values simply because the blobs are centered near 0. Real data is often centered the same way, for example "temperature compared to the average day".)

If w is positive, the model reads it as "higher x means more likely class 1":

- positive x pushes z up, so ŷ goes up (toward class 1)
- negative x pushes z down, so ŷ goes down (toward class 0)
- x = 0 pushes nothing

**What turning the w knob up does depends on the sign of x.** This is the key idea:

| x | Turning w up changes z by | So the forecast ŷ... |
|---|---|---|
| positive | up | goes up |
| negative | **down** | goes **down** |
| zero | nothing | does not move |

A negative x flips the knob's effect. That is why x is multiplied into the gradient: it carries the direction.

**Applied to all three rows (w = 1, current state):**

| Row | x | What the row needs | Does turning w up help? | x × error | Reading |
|---|---|---|---|---|---|
| 1 | +1.4 | ŷ up (y = 1, error −0.2) | yes (positive x, ŷ rises) | −0.28 (negative) | "raise w" |
| 2 | −0.85 | ŷ down (y = 0, error +0.3) | yes (negative x, ŷ falls) | −0.255 (negative) | "raise w" |
| 3 | −2.2 | ŷ up (y = 1, error −0.9) | **no** (negative x, ŷ falls) | **+1.98 (positive)** | "lower w" |

The sign of x × error is a vote: **negative = raising w helps this row, positive = raising w hurts this row.** Rows 1 and 2 vote weakly for raising w. Row 3 votes strongly for lowering it, and its vote is bigger than the other two combined, so the total is positive and w gets lowered (1 → 0.952).

**The story behind Row 3 in plain words.** With w = 1, the model reads x = −2.2 as strong evidence for class 0, but the truth is class 1. The model has the wrong relationship for this kind of point, so it needs to weaken w (even turn it negative) so that a negative x stops pulling the forecast down.

---

## Open exercise (not yet answered)

Same Row 3 situation, but suppose the input were **positive**: x = +2.2, y = 1, ŷ = 0.1, so the error is still −0.9.

1. Is x × error positive or negative?
2. According to the vote rule above, what does this row say about the w knob: raise it or lower it?

**My answer:** _(to fill in)_

---

## Pending from earlier steps

- Clipping question: with ε = 1e-15, a confidently wrong row costs −log(1e-15) ≈ 34.5, compared with 2.30 for Row 3 of the worked example (y = 1, ŷ = 0.1). Is 34.5 still "very heavily punished", or does clipping change the answer in a way that matters? (Parked for the concept ledger check.)
- Exercise B: write the sigmoid and log-loss formulas from memory (parked for the concept ledger check).
- Next step: the training loop.