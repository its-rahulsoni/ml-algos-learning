# One-Hot Encoding — Explained From Scratch

Let's start completely from scratch with a brand new, very concrete example — no assumptions this time.

## Why does one-hot encoding even exist? (The real-world problem)

Imagine you're building a machine learning model to classify fruits. You have 3 types: **apple**, **banana**, **cherry**. Computers don't understand words, so you'd naturally convert these into numbers:

```
apple  -> 0
banana -> 1
cherry -> 2
```

This seems fine at first, but it creates a hidden problem: by using plain integers, you've accidentally told the computer that `cherry (2)` is "twice as much" as `banana (1)`, and that `banana - apple = 1`, as if there's a meaningful mathematical relationship between fruit types — there isn't. Apple, banana, and cherry are just different categories with no natural order or numeric distance between them. But a model that does math (like multiplying by weights, or computing distances) will treat `2` as genuinely larger than `1`, which introduces a false assumption.

**One-hot encoding solves this** by representing each category as a separate "flag" — a position that's either "on" (1) or "off" (0) — with no category considered mathematically bigger or smaller than another.

## Building it by hand, one label at a time

Let's say we have 3 categories total (apple, banana, cherry), numbered 0, 1, 2. For each individual label, the one-hot vector has **exactly 3 positions** (one slot per possible category), and exactly **one** of those positions is `1` — the one matching that label — while every other position is `0`.

- Label `0` (apple) → `[1, 0, 0]` — the "1" sits in position 0, because the label is 0.
- Label `1` (banana) → `[0, 1, 0]` — the "1" sits in position 1, because the label is 1.
- Label `2` (cherry) → `[0, 0, 1]` — the "1" sits in position 2, because the label is 2.

Notice the pattern: **the position of the `1` always matches the label's numeric value.** That's the entire rule of one-hot encoding — nothing more.

## Our new working example

Let's say we have 5 fruits, in this order: banana, apple, cherry, cherry, banana. As labels:

```python
labels = [1, 0, 2, 2, 1]
```
(1=banana, 0=apple, 2=cherry, 2=cherry, 1=banana)

We want to produce a 2D array with **5 rows** (one per fruit in our list) and **3 columns** (one per possible category: apple/banana/cherry), where row `i` is the one-hot vector for `labels[i]`:

```
row 0 (label 1, banana) -> [0, 1, 0]
row 1 (label 0, apple)  -> [1, 0, 0]
row 2 (label 2, cherry) -> [0, 0, 1]
row 3 (label 2, cherry) -> [0, 0, 1]
row 4 (label 1, banana) -> [0, 1, 0]
```

Before touching NumPy at all, let's just write this out as plain nested Python lists, by hand, so the target output is 100% concrete:

```python
expected_output = [
    [0, 1, 0],   # banana
    [1, 0, 0],   # apple
    [0, 0, 1],   # cherry
    [0, 0, 1],   # cherry
    [0, 1, 0],   # banana
]
```

This is literally what we're trying to produce — no NumPy trickery yet, just the plain answer written out.

## Now — the identity matrix, explained from zero

An **identity matrix** is a special square grid of numbers where every cell is `0`, **except** the diagonal running from top-left to bottom-right, which is all `1`s. For a 3x3 identity matrix (3 rows, 3 columns):

```
[1, 0, 0]
[0, 1, 0]
[0, 0, 1]
```

Look at each row individually:
- Row 0 (top row): `[1, 0, 0]` — the `1` is in position 0.
- Row 1 (middle row): `[0, 1, 0]` — the `1` is in position 1.
- Row 2 (bottom row): `[0, 0, 1]` — the `1` is in position 2.

**Stop and compare this to our fruit one-hot vectors above.** Row 0 of the identity matrix, `[1, 0, 0]`, is exactly the one-hot vector for label 0 (apple). Row 1, `[0, 1, 0]`, is exactly the one-hot vector for label 1 (banana). Row 2, `[0, 0, 1]`, is exactly the one-hot vector for label 2 (cherry).

**This is not a coincidence — it's the entire trick.** The identity matrix, by its very definition (1 on the diagonal, 0 elsewhere), already contains every possible one-hot vector as its rows, pre-built. "Row `k` of the identity matrix" and "the one-hot vector for label `k`" are literally the same thing.

So instead of building each one-hot vector ourselves, we can just build the identity matrix once, and then for each label, **go fetch the matching row**.

## Fetching rows: ordinary indexing first (a warm-up)

Before the trick, let's make sure row-fetching itself is clear. If `identity` is our 3x3 identity matrix, then:

```python
identity[0]   # gives you [1, 0, 0]  -- the row at position 0
identity[1]   # gives you [0, 1, 0]  -- the row at position 1
identity[2]   # gives you [0, 0, 1]  -- the row at position 2
```

This is just standard indexing — pick one row by its position number. Nothing new here; it's the same as indexing a Python list.

## The actual trick: asking for many rows at once

Now, instead of fetching rows one at a time, NumPy lets you pass **a whole list of positions** at once, and it will fetch all of them, in that exact order, into a new array:

```python
identity[[0, 1, 2]]   # fetch row 0, then row 1, then row 2 -> gives back the whole identity matrix, unchanged
identity[[2, 0, 0]]   # fetch row 2, then row 0, then row 0 again
```

This is called **fancy indexing**. The key insight: whatever list of numbers you pass in, NumPy treats each number as "go fetch this row," and it doesn't matter if numbers repeat or are out of order — it just fetches whatever you ask, in whatever order you ask.

Now look at our actual `labels` list again: `[1, 0, 2, 2, 1]`. This is *already* a list of row-positions we want to fetch! Label `1` means "fetch row 1 (banana's one-hot vector)." Label `0` means "fetch row 0 (apple's)." And so on. So we can hand `labels` **directly** to the identity matrix as the fetch-list:

```python
identity[[1, 0, 2, 2, 1]]
```

Or equivalently, since `labels` already *is* `[1, 0, 2, 2, 1]`:

```python
identity[labels]
```

This single expression says: "for every value in `labels`, go fetch that row of the identity matrix, and stack the results, in order." That's it — that's the whole algorithm.

## Now let's actually run it and watch it build up

```python
import numpy as np

# Our fruit example: 1=banana, 0=apple, 2=cherry
labels = np.array([1, 0, 2, 2, 1])
print("labels:", labels)
print("(meaning: banana, apple, cherry, cherry, banana)")

num_classes = 3   # apple, banana, cherry -- 3 total categories
identity = np.eye(num_classes)
print("\nidentity matrix (3x3):")
print(identity)
print("row 0 =", identity[0], "  <- one-hot for label 0 (apple)")
print("row 1 =", identity[1], "  <- one-hot for label 1 (banana)")
print("row 2 =", identity[2], "  <- one-hot for label 2 (cherry)")

print("\n--- Now fetch rows one at a time, matching each label ---")
for i, label in enumerate(labels):
    print(f"labels[{i}] = {label}  ->  identity[{label}] = {identity[label]}")

print("\n--- Now do it all at once using fancy indexing ---")
result = identity[labels]
print("identity[labels] =")
print(result)

print("\nDoes this match our hand-written expected_output? Let's check:")
expected_output = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
])
print(np.array_equal(result, expected_output))
```

### Output

```
labels: [1 0 2 2 1]
(meaning: banana, apple, cherry, cherry, banana)

identity matrix (3x3):
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
row 0 = [1. 0. 0.]   <- one-hot for label 0 (apple)
row 1 = [0. 1. 0.]   <- one-hot for label 1 (banana)
row 2 = [0. 0. 1.]   <- one-hot for label 2 (cherry)

--- Now fetch rows one at a time, matching each label ---
labels[0] = 1  ->  identity[1] = [0. 1. 0.]
labels[1] = 0  ->  identity[0] = [1. 0. 0.]
labels[2] = 2  ->  identity[2] = [0. 0. 1.]
labels[3] = 2  ->  identity[2] = [0. 0. 1.]
labels[4] = 1  ->  identity[1] = [0. 1. 0.]

--- Now do it all at once using fancy indexing ---
identity[labels] =
[[0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 1. 0.]]

Does this match our hand-written expected_output? Let's check:
True
```

Look at the middle section of that output very closely — it prints exactly which row gets fetched for each label:

```
labels[0] = 1  ->  identity[1] = [0. 1. 0.]     (banana's row)
labels[1] = 0  ->  identity[0] = [1. 0. 0.]     (apple's row)
labels[2] = 2  ->  identity[2] = [0. 0. 1.]     (cherry's row)
labels[3] = 2  ->  identity[2] = [0. 0. 1.]     (cherry's row, again)
labels[4] = 1  ->  identity[1] = [0. 1. 0.]     (banana's row, again)
```

This is the **exact same result**, row for row, as `identity[labels]` computed all at once. That loop is what's happening conceptually — `identity[labels]` is just doing all 5 of those individual fetches in one shot, and stacking the answers together into a single 2D array.

And the final check confirms it: our fancy-indexed result matches the `expected_output` we hand-wrote at the very start, before touching any NumPy trick — proving the identity-matrix-and-fetch approach genuinely produces the answer we defined by hand.

## The concept boiled down to its simplest form

1. **One-hot encoding** = represent each category as a vector that's all zeros except a single `1` at the position matching that category's number.
2. **The identity matrix already contains every possible one-hot vector**, one per row, purely because of how it's defined (1s only on the diagonal).
3. **Your labels list is already a list of "which row do I want"** — so indexing the identity matrix with your labels array, all at once (`identity[labels]`), fetches the correct one-hot vector for every single label, in one line, with no loop.