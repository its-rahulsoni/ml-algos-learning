# Walkthrough of `fit()` in the From-Scratch Gaussian Naive Bayes (Part E)

> Context: banana vs apple classification with two features (length and thickness).
> This note explains the `fit` method of `MyGaussianNB`, line by line, using the real 10 training fruits.
> Full code: `gaussian_nb_from_scratch.py`

**Question:**

> The `fit` method is hard to follow. Please explain each line of it, especially what happens inside the loop.

**Answer:**

`fit` is harder to read than the Part C/D code because it is general. The earlier code had one variable per number (`banana_length_mean`), and `fit` keeps everything in tables and loops over the classes. So I'll walk through it with the real 10 fruits and show what every variable holds at each step.

## The one idea

The loop does the same three jobs once per class:
1. Pick out the fruits of that class.
2. Work out how common the class is (prior).
3. Work out the mean and variance of each feature.

It is Part C's `banana_len.mean()` and `banana_len.var()`, run first for apples and then for bananas, with both features handled at once.

## The data we will trace

X is a table with one row per fruit (length, thickness), and y holds the labels:

```
row 0   [16, 3]   banana
row 1   [18, 4]   banana
row 2   [20, 4]   banana
row 3   [22, 4]   banana
row 4   [24, 5]   banana
row 5   [ 6, 5]   apple
row 6   [ 7, 7]   apple
row 7   [ 8, 7]   apple
row 8   [ 9, 7]   apple
row 9   [10, 9]   apple
```

## Lines before the loop

**`n_samples, n_features = X.shape`**
`X.shape` is the pair (number of rows, number of columns) = `(10, 2)`. Python splits it into two names: `n_samples = 10` and `n_features = 2`. We need 10 to compute priors, and 2 to size the tables.

**`self.classes_ = np.unique(y)`**
Lists each label once, sorted alphabetically: `['apple', 'banana']`. So **apple is position 0 and banana is position 1** for the rest of the method.

**`n_classes = len(self.classes_)`**
Counts that list: `2`.

**`self.epsilon_ = self.var_smoothing * X.var(axis=0).max()`**
This is the tiny safety number from Part D, in three steps:
- `X.var(axis=0)` computes the variance of each column over all 10 fruits: `[41.0, 3.25]`.
- `.max()` picks the biggest: `41.0`.
- Multiply by `1e-9`: epsilon = `4.1e-08`.

**The three `np.zeros(...)` lines** create empty tables to fill in later:

```
priors_ = [0, 0]                    one slot per class
means_  = [[0, 0],                  one row per class,
           [0, 0]]                  one column per feature
vars_   = [[0, 0],
           [0, 0]]
```

Each row is a class and each column is a feature. Row 0 is apple and row 1 is banana. We create them first because we need somewhere to write the answers. Without them, `self.priors_[i] = ...` would fail because nothing exists yet.

## Inside the loop

**`for i, c in enumerate(self.classes_):`**
`enumerate` hands out a position and a label together, one pair per pass:
- Pass 1: `i = 0`, `c = "apple"`
- Pass 2: `i = 1`, `c = "banana"`

`i` says which row of the tables to write into. `c` says which fruits to study. The three lines inside run once per pass.

### Pass 1: i = 0, c = "apple"

**`X_c = X[y == c]`** has two parts.
- `y == c` compares every label with `"apple"` and gives one True/False per fruit:
  ```
  [False False False False False  True  True  True  True  True]
  ```
- Using that as an index, `X[...]`, keeps only the rows marked True:
  ```
  X_c = [[ 6, 5],
         [ 7, 7],
         [ 8, 7],
         [ 9, 7],
         [10, 9]]
  ```
So `X_c` is just the 5 apples, with both columns. This is the step that separates the classes.

**`self.priors_[i] = X_c.shape[0] / n_samples`**
`X_c.shape[0]` is the number of rows of `X_c` = 5. So `5 / 10 = 0.5`, written into slot 0. Now `priors_ = [0.5, 0]`.

**`self.means_[i] = X_c.mean(axis=0)`**
`axis=0` means "average down the rows, once per column":
- length column: (6 + 7 + 8 + 9 + 10) ÷ 5 = 8
- thickness column: (5 + 7 + 7 + 7 + 9) ÷ 5 = 7

Result `[8, 7]`, written into row 0 of `means_`. This is Part C's `apple_len.mean()`, done for both features at once.

**`self.vars_[i] = X_c.var(axis=0) + self.epsilon_`**
`X_c.var(axis=0)` is the variance of each column:
- length: squared distances 4, 1, 0, 1, 4 give 10 ÷ 5 = 2
- thickness: squared distances 4, 0, 0, 0, 4 give 8 ÷ 5 = 1.6

Result `[2, 1.6]`. We then add epsilon (`4.1e-08`), giving about `[2.00000004, 1.60000004]`, written into row 0 of `vars_`. This is Part C's `apple_len.var()`, again for both features at once.

### Pass 2: i = 1, c = "banana"

Exactly the same three lines, with different data:
- `y == "banana"` gives `[True True True True True False False False False False]`, so `X_c` is rows 0 to 4 (the five bananas).
- `priors_[1] = 5 / 10 = 0.5`.
- `means_[1] = [20, 4]`.
- `vars_[1] = [8, 0.4]` plus epsilon.

## After the loop

```
                    length   thickness
means_  row 0 apple   8         7
        row 1 banana  20        4

vars_   row 0 apple   2         1.6
        row 1 banana  8         0.4        (each plus 4.1e-08)

priors_ = [0.5, 0.5]
```

This is exactly the table from Part D. It is the whole trained model.

**`return self`** hands the finished model back. It is what lets you write `model = MyGaussianNB().fit(X, y)` in one line. It plays no part in the maths.

## The loop with no loop

Written out for our two classes, the loop is just this:

```python
X_c = X[y == "apple"]                       # pass 1
priors_[0] = len(X_c) / 10
means_[0] = X_c.mean(axis=0)
vars_[0] = X_c.var(axis=0) + epsilon

X_c = X[y == "banana"]                      # pass 2
priors_[1] = len(X_c) / 10
means_[1] = X_c.mean(axis=0)
vars_[1] = X_c.var(axis=0) + epsilon
```

It is the same three lines twice. The loop only avoids writing them again for every class, so the code works for 2 classes or 20.

## Run this to see it yourself

```python
import numpy as np
X = np.array([[16,3],[18,4],[20,4],[22,4],[24,5],[6,5],[7,7],[8,7],[9,7],[10,9]], dtype=float)
y = np.array(["banana"] * 5 + ["apple"] * 5)
for i, c in enumerate(np.unique(y)):
    X_c = X[y == c]
    print(i, c, X_c.shape[0], X_c.mean(axis=0), X_c.var(axis=0))
```

Expected output:
```
0 apple 5 [8. 7.] [2.  1.6]
1 banana 5 [20.  4.] [8.  0.4]
```

Each printed row is one pass of the loop: the position, the class, the count, the means and the variances.

## Quick check

If we added a third class, "pear", what would change in the loop, and what would not?