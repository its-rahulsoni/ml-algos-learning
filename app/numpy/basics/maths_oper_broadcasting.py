import os
import numpy as np

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

# ⭐ TOPIC 2 — NumPy Math Operations & Broadcasting (FULL DEEP DIVE)

print("⭐ TOPIC 2 — NumPy Math Operations & Broadcasting (FULL DEEP DIVE)")
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

# ✅ 1️⃣ Element-wise Math Operations
print("✅ 1️⃣  Element-wise Math Operations:\n")

a = np.array([10, 20, 30])
b = np.array([1, 2, 3])

print("Element-wise addition")
print(a + b)
print("--------------------------------\n")

print("Element-wise multiplication")
print(a * b)
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

# ✅ 2️⃣ Operations with Scalars ( VERY IMPORTANT )

a = np.array([10, 20, 30])

print("✅ 2️⃣  Operations with Scalars ( VERY IMPORTANT )")
print(a + 5)
print(a * 10)
print(a / 2)
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

# ⭐ Broadcasting BEGINNER-Friendly Definition:
# Broadcasting allows NumPy to apply operations between arrays of different shapes by stretching the smaller one without creating copies.

# -----------------------------------------------------------------------------------------------------------

# ✅ 3️⃣ Broadcasting Example (Different Shapes)

print("✅ 3️⃣  Broadcasting Example (Different Shapes)")
mat = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

vec = np.array([1, 2, 3])

print("Broadcasting allows NumPy to apply operations between arrays of different shapes by stretching the smaller one without creating copies.")
print(mat + vec)
print("--------------------------------\n")

"""
Explanation:

mat shape → (2,3)
vec shape → (3,)

NumPy stretches vec to match each row.
This is row-wise broadcasting.
"""

# -----------------------------------------------------------------------------------------------------------

# ✅ 4️⃣ Broadcasting on Columns

col_vec = np.array([[1], [2]])  # shape (2,1)

mat = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print("✅ 4️⃣  Broadcasting on Columns")
print(mat + col_vec)

"""
Explanation:

col_vec → (2,1)
mat → (2,3)

Stretched across columns.
"""