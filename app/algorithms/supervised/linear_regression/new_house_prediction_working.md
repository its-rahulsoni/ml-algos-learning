# How apply_standardizer works on a new house

```python
# NEW: turns raw columns into z-scores using SAVED means and stds
def apply_standardizer(X, means, stds):
    # (value - average) / s, column by column
    return (X - means) / stds
```

The function does the same thing for one house as for 160 houses. For a new house it does it on a single row.

# The inputs

The new house is [age 15, bedrooms 3, plot 1800], and the saved numbers from training are:

| | age | bedrooms | plot_area |
|---|---|---|---|
| means | 22.02 | 3.14 | 1825.98 |
| stds | 11.64 | 1.42 | 717.9 |

# Step 0: make the house a table with one row

Inside predict_new_house, this line runs before apply_standardizer:

```python
row = np.array(raw_house, dtype=float).reshape(1, -1)
```

It turns the list [15, 3, 1800] into shape (1, 3): one house, three columns. This matches the shape of X_train, (160, 3), where each row is a house. The function expects a table of houses, so a single house is a table with one row.

# Step 1: X − means

numpy lines up the 3 saved averages with the 3 columns and subtracts each one from its own column:

- age: 15 − 22.02 = **−7.02**
- bedrooms: 3 − 3.14 = **−0.14**
- plot area: 1800 − 1825.98 = **−25.98**

These are the distances from the average house, still in the original units (years, rooms, sq ft).

# Step 2: divide by stds

Each distance is divided by its own column's s:

- age: −7.02 / 11.64 = **−0.603**
- bedrooms: −0.14 / 1.42 = **−0.099**
- plot area: −25.98 / 717.9 = **−0.036**

Result: [−0.603, −0.099, −0.036], shape (1, 3). Read it as "this house is 0.6 typical steps newer than average in age, about average in bedrooms, and about average in plot area."

# What happens next in predict_new_house

1. `add_ones` puts a 1 in front, giving [1, −0.603, −0.099, −0.036], shape (1, 4). This is the same ones column we use in training, and it stands for b.
2. `predict` multiplies this row by θ = [81.85, −4.64, 7.15, 21.68]:

81.85·1 + (−4.64)(−0.603) + (7.15)(−0.099) + (21.68)(−0.036) = 81.85 + 2.80 − 0.71 − 0.78 ≈ **83.16**

Your code printed 83.17, and the gap is rounding in my hand numbers.

# Why the same function also works on many houses

If X has 40 test houses, shape (40, 3), the same line subtracts the 3 averages from every row and divides every row by the 3 stds. Nothing in the code needs to change. A new house is just the case with 1 row.

# One thing to notice

Nowhere in this function are the averages or s's recomputed. They come in as arguments (means, stds), which are the arrays from fit_standardizer on the training rows. That is the train-only rule in code form.