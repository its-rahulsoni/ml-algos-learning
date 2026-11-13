# A3. Deriving Bayes' Theorem (Every Step)

In classification we usually know "given the class, how do the features look?", which is like `P(free | spam)`, because we can study the training data class by class.

But what we want is the reverse: "given the features, which class is it?", which is like `P(spam | free)`.

---

## Step 1. Start from the conditional probability formula, rearranged

Multiply both sides of `P(A | B) = P(A and B) ÷ P(B)` by `P(B)`:

```
P(A and B) = P(A | B) × P(B)        ... (equation 1)
```

## Step 2. Swap the roles of A and B

The same logic works with the roles of A and B swapped:

```
P(B and A) = P(B | A) × P(A)        ... (equation 2)
```

## Step 3. Equate the two equations

"A and B" and "B and A" are the same event (both happen), so `P(A and B) = P(B and A)`. That means the right-hand sides of equations 1 and 2 are equal:

```
P(A | B) × P(B) = P(B | A) × P(A)
```

## Step 4. Solve for P(A | B)

We want `P(A | B)` alone, so divide both sides by `P(B)`:

```
P(A | B) = P(B | A) × P(A) ÷ P(B)
```

**That is Bayes' theorem.** Nothing was skipped; it is just the conditional probability formula used twice.

---

## Summary

| Term | Name | Meaning |
|---|---|---|
| `P(A \| B)` | Posterior | What we want: probability of A given B |
| `P(B \| A)` | Likelihood | Probability of B given A (learned from training data) |
| `P(A)` | Prior | Probability of A before seeing B |
| `P(B)` | Evidence | Overall probability of B |