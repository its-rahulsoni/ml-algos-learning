import os
import numpy as np

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

# TOPIC 3: Matrix Multiplication (Dot Product & ML Use-Cases)

print("TOPIC 3: Matrix Multiplication (Dot Product & ML Use-Cases)")
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

# ✅ 1️⃣ Dot Product (1D vectors)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("✅ 1️⃣  Dot Product (1D vectors)")
print(np.dot(a, b))

"""
### ✅ What is Dot Product (for 1D vectors)?

For two 1D vectors, the dot product is:

Multiply each pair of corresponding elements,
then add all products.

Mathematically:
a⋅b = a1b1 + a2b2 + a3b3

"""
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

# ✅ 2️⃣ Matrix × Matrix Multiplication (2D arrays)

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print("✅ 2️⃣  Matrix × Matrix Multiplication (2D arrays)")
print(A @ B)

"""
Compute row × column:
Row1⋅Col1 = 15 + 27 = 19
Row1⋅Col2 = 16 + 28 = 22
Row2⋅Col1 = 35 + 47 = 43
Row2⋅Col2 = 36 + 48 = 50

⭐ Important Rules (You MUST remember them)

To multiply A(m × n) with B(n × p):
Inner dimensions (n) must match
Result is shape → (m × p)
"""

print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

# ✅ 3️⃣ Matrix × Vector multiplication

W = np.array([
    [0.2, 0.8, -0.5],
    [1.0, -1.0, 0.3]
])   # shape (2,3)

x = np.array([1, 2, 3])  # shape (3,)

print("✅ 3️⃣  Matrix × Vector multiplication")
print(W @ x)
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

# ✅ 4️⃣ Inner Product (Dot-like operation)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.inner(a, b)
print("✅ 4️⃣  Inner Product (Dot-like operation)")
print("Inner Product: ", result)
print("--------------------------------\n")

"""
For 1D vectors, inner product = sum of elementwise products, same as dot product.
inner(a,b)=a1b1 + a2b2 + a3b3 + ......
Result - Single numerical value.
"""

# -----------------------------------------------------------------------------------------------------------

# ✅ 5️⃣ Outer Product

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.outer(a, b)
print("✅ 5️⃣  Outer Product")
print("Outer Product: \n", result)
print("--------------------------------\n")

"""
It creates a matrix where:
every element of a multiplies
every element of b

outer(a,b) =
[ a1b1  a1b2  ]
[ a2b1  a2b2  ]

"""
# -----------------------------------------------------------------------------------------------------------

"""
⭐ What are mean / variance / standard deviations ?

Mean (average) — gives the central tendency of data. In ML/data-analysis, you often inspect mean to understand the “typical” value in a feature.

Variance — measures how spread out values are around the mean. High variance → data is spread out. Useful in understanding feature distribution, detecting outliers.

Standard Deviation (std) — square root of variance; in same units as data. Helps gauge “typical deviation” from mean.
"""
print("1D array - mean / variance / standard deviations")
# Example 1D data: ages of students
ages = np.array([18, 20, 22, 19, 21, 20, 23, 22])

# Compute mean
mean_age = np.mean(ages) # computes arithmetic mean: sum of all values divided by number of values.
print("Mean age:", mean_age) 

# Compute variance
var_age = np.var(ages) # computes variance: average of squared deviations from mean.
print("Variance of age:", var_age)

# Compute standard deviation
std_age = np.std(ages) # computes standard deviation: sqrt of variance.
print("Standard deviation of age:", std_age)

print("--------------------------------\n")

data = np.array([
    [18, 50000, 85],
    [20, 55000, 90],
    [22, 60000, 88],
    [19, 52000, 87]
])

print("2D array - mean / variance / standard deviations")

# Mean of each column (feature-wise stats)
feature_means = np.mean(data, axis=0)
print("Feature means:", feature_means)

# Std dev of each column
feature_std = np.std(data, axis=0)
print("Feature std:", feature_std)

# If you want row-wise mean (per sample)
row_means = np.mean(data, axis=1)
print("Row means:", row_means)

print("--------------------------------\n")

# Explanation for the above implementation.
# The data as a table

# Think of your array as a small table with 4 rows (people) and 3 columns (features: age, salary, score):

# | Row | Age | Salary | Score |
# |-----|-----|--------|-------|
# | 0 | 18 | 50000 | 85 |
# | 1 | 20 | 55000 | 90 |
# | 2 | 22 | 60000 | 88 |
# | 3 | 19 | 52000 | 87 |

# ## axis=0 → operate down each column

# `np.mean(data, axis=0)` collapses the **rows**, leaving one result per **column**. In other words, it walks down each column and averages it. This gives you "feature-wise" statistics 
# — the average age across all people, the average salary across all people, and so on.

# - Age column: (18+20+22+19)/4 = 19.75
# - Salary column: (50000+55000+60000+52000)/4 = 54250
# - Score column: (85+90+88+87)/4 = 87.5

# So `feature_means = [19.75, 54250, 87.5]`

# Mentally: **axis=0 means "squash the rows together."**

# ## Standard deviation, same idea

# `np.std(data, axis=0)` also collapses rows, but instead of averaging the raw values, it measures how spread out each column is around its own mean. For each column it computes 
# the average squared distance from the mean, then takes the square root.

# - Age std ≈ 1.48
# - Salary std ≈ 3766.6
# - Score std ≈ 1.80

# (Note: NumPy's default `std` uses population standard deviation — divides by n, not n−1.)

# ## axis=1 → operate across each row

# `np.mean(data, axis=1)` does the opposite: it collapses the **columns**, leaving one result per **row**. It walks across each row and averages the values in that row.

# - Row 0: (18+50000+85)/3 ≈ 16701.0
# - Row 1: (20+55000+90)/3 ≈ 18370.0
# - Row 2: (22+60000+88)/3 ≈ 20036.67
# - Row 3: (19+52000+87)/3 ≈ 17368.67

# Mentally: **axis=1 means "squash the columns together."**

# ## Why row_means looks odd here

# Notice those row means are dominated by the salary number — since age, salary, and score are on wildly different scales, averaging across a row doesn't really mean anything 
# meaningful in this dataset (it's mixing apples and oranges). Row-wise stats make sense when all columns represent the same kind of quantity (e.g., three test scores per student). 
# Column-wise stats (axis=0) are the natural choice here since each column is a distinct feature.

# ## Quick rule of thumb

# - axis=0 → "down the rows" → one output per column
# - axis=1 → "across the columns" → one output per row

# A trick some people use: axis=0 removes the row dimension (shape (4,3) → (3,)), and axis=1 removes the column dimension (shape (4,3) → (4,)).

# -----------------------------------------------------------------------------------------------------------