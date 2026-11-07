# Understanding `apply_minmax`

**The concept**

Each feature is measured in its own units, such as years for age and rupees for salary. Min-max scaling replaces every value with how far along its own range it sits:

- the feature's smallest value becomes 0
- its largest value becomes 1
- a value halfway between them becomes 0.5

After that, every feature lives on the same 0 to 1 ruler, so none of them can dominate the distance just because its numbers are bigger.

**How the code does it**

Say age runs from 20 to 60 and salary from 20,000 to 100,000, so `mins = [20, 20000]` and `maxs = [60, 100000]`. We scale three people: `X = [[30, 50000], [60, 100000], [20, 20000]]`.

1. `span = maxs - mins` gives the width of each range: `[40, 80000]`. The span is the divisor in the formula, (value − min) / (max − min).

2. `span = np.where(span == 0, 1.0, span)` is a safety net. `np.where(condition, a, b)` checks each entry and uses `a` where the condition is true, otherwise `b`. If a feature never changes, its max equals its min, so its span is 0, and dividing by 0 would give nan or inf. We replace a 0 span with 1. That's harmless, because for such a feature `X - mins` is 0 anyway, so the scaled value is simply 0. In our example no span is 0, so nothing changes.

3. `(X - mins) / span` does the real work, in two broadcasts:
   - `X - mins`: shape (3, 2) minus shape (2,), so `mins` is subtracted from every row. Result: `[[10, 30000], [40, 80000], [0, 0]]`. This is "how far above the minimum".
   - `/ span`: each column is divided by its own span (40 for age, 80000 for salary). Result: `[[0.25, 0.375], [1, 1], [0, 0]]`.

Check the first row: age 30 is 10 above the minimum of 20, out of a range of 40, so 0.25. Salary 50,000 is 30,000 above 20,000, out of 80,000, so 0.375. Those are the same numbers we used in the age/salary example earlier.

**Why `mins` and `maxs` are passed in rather than computed inside the function**

We compute them once from the training data (`fit_minmax`) and reuse those same numbers for the test points and the grid spots. If each set were scaled by its own min and max, the same raw value would land on different scaled positions in different sets, like measuring with two different rulers. That's the "training data only" rule from before. It also explains the word "roughly" in the docstring: training values land exactly in 0 to 1, but a test point outside the training range can come out slightly below 0 or above 1.