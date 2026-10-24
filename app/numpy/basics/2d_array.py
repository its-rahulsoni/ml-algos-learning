import os
import numpy as np

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

### 1️⃣ Creating an ndarray from a Python list

# Line-by-line explanation:
# np.array(...)
# Converts a Python list into a NumPy array.
#
# print(arr)
# Shows the array: [10 20 30 40]
#
# dtype
# Shows data type (e.g., int32 or int64)
#
# shape
# Tells structure: here (4,) meaning 1-D array with 4 elements.

arr = np.array([10, 20, 30, 40])
print("### 1️⃣  Creating an ndarray from a Python list")
print("arr:", arr)
print("arr.dtype: ", arr.dtype)
print("arr.shape: ", arr.shape)
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

### 2️⃣ Creating a 2D Matrix (Very important for ML)

# Explanation:
#
# Creates a 2x3 matrix
# shape = (2, 3) → 2 rows, 3 columns
#
# Used in:
# representing datasets: rows = samples, columns = features

mat = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print("### 2️⃣  Creating a 2D Matrix (Very important for ML)")
print("Multi Dim Arr: \n", mat)
print("Array Dimensions: ", mat.shape)
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

### 3️⃣ zeros() → all zeros (default dtype float)

# Explanation:
#
# np.zeros((3,4))
# Creates a matrix of shape 3×4 filled with 0.0
#
# This is used when:
# initializing weight matrices
# placeholder arrays
# building algorithms step-by-step

zeros = np.zeros((3, 4))
print("### 3️⃣  zeros() → all zeros (default dtype float)")
print("zeros array: \n", zeros)
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

### 4️⃣ ones() → all ones

# Used when:
#
# Creating base feature arrays
# Normalization or scaling matrices

ones = np.ones((2, 3))
print("### 4️⃣  ones() → all ones")
print("ones arr: \n", ones)
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

### 5️⃣ arange() → like Python range() but returns ndarray

# Explanation:
#
# Start: 1
# Stop: 10 (excluded)
# Step: 2

# Output: [1 3 5 7 9]
#
# Used when:
# Generating index arrays
# Generating time steps in simulations

arr = np.arange(1, 10, 2)
print("### 5️⃣  arange() → like Python range() but returns ndarray")
print("Time stepping / jumping in array: \n", arr)
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

### 6️⃣ linspace() → evenly spaced numbers

# Output:
# [0. 0.25 0.5 0.75 1.]

# Explanation:

# Start at 0
# End at 1
# Generate 5 evenly spaced values
#
# Critical in:
# Signal processing
# Plotting curves
# ML training schedules

arr = np.linspace(0, 1, 5)
print("### 6️⃣  linspace() → evenly spaced numbers")
print("Evenly placed array: ", arr)
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

### 7️⃣ random.rand() → Random values (VERY IMPORTANT)

# Explanation:
#
# Creates a 3×2 matrix with random numbers in [0,1]
#
# Used to initialize:
# neural network weights
# random samples
# synthetic datasets

print("### 7️⃣  random.rand() → Random values (VERY IMPORTANT)")
weights = np.random.rand(3, 2)
print("Random weights \n", weights)

weights = np.random.randint(0, 10, (3, 2))
print("Random integer weights:\n", weights)

print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------
