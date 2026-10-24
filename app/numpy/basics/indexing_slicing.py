import os
import numpy as np

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

### 1️⃣ Basic Indexing (1D Arrays)

# Explanation:
#
# arr[0] → 10
# arr[3] → 40
# arr[-1] → 50 (negative index = from end)

arr = np.array([10, 20, 30, 40, 50])

print("### 1️⃣  Basic Indexing (1D Arrays)")
print(arr[0])   # first element
print(arr[3])   # 4th element
print(arr[-1])  # last element
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

### 2️⃣ Slicing (1D Arrays)

# Key Concepts:

# [start:end] → end excluded
# [::step] → skip values
# Negative indexes allowed
# Negative steps (reverse)

# 2️⃣ : → All columns
# The colon without numbers means:
# select every column
# from start to end

arr = np.array([10, 20, 30, 40, 50])

print("### 2️⃣  Slicing (1D Arrays)")
print(arr[1:4])     # 20 30 40
print(arr[:3])      # 10 20 30  // Skip values from index 3 i.e. consider only values BEFORE this mentioned index ....
print(arr[2:])      # 30 40 50  // Skip values up-to index 2 i.e. consider only values AFTER this mentioned index ....
print(arr[::2])     # step = 2 → 10 30 50
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

### 3️⃣ 2D Array Indexing
print("### 3️⃣  2D Array Indexing ###")

### 4️⃣ Accessing Specific Elements

# Syntax:
# mat[row, column]

mat = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
print("### 4️⃣  Accessing Specific Elements")
print(mat[0, 0])   # 10
print(mat[1, 2])   # 60
print(mat[2, 1])   # 80
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

### 5️⃣ Slicing Rows
print("### 5️⃣  Slicing Rows")
print(mat[0])      # entire 1st row
print(mat[1, :])   # same as above (2nd row)
print(mat[2, 1:])  # 80, 90 // Meaning select 2nd Row and skip 1st element and then print rest all of them ....
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

### 6️⃣ Slicing Columns
print("### 6️⃣  Slicing Columns")
print(mat[:, 0])   # first column → 10 40 70   // : means select every row from start to end ....
print(mat[:, 2])   # third column → 30 60 90
print(mat[:2, 2]) 
print("--------------------------------\n")

"""
Explanation:

1️⃣ mat[:2, 2] → is 2D slicing
It has two parts:
rows    , columns
[:2]    , 2

✔ What does :2 mean for rows ?

mat[:2] means:
start at row index 0
go up to (but NOT including) row index 2.
So it includes rows 0 and 1:
Row 0 → [10, 20, 30]
Row 1 → [40, 50, 60]
(Stops before row 2)

✔ What does 2 mean for columns ?
Column index 2 is the 3rd column.
Values at column 2:
From row 0 → 30
From row 1 → 60

🎯 Final Output

Putting them together:
[30 60]
These are the elements at column 2 from the first two rows.
"""

# -----------------------------------------------------------------------------------------------------------

### 7️⃣ Extracting Sub-Matrix (VERY IMPORTANT)

# Explanation:
# rows → 0 & 1
# columns → 1 & 2

# Extracted a 2x2 window → widely used in:
# CNN image processing
# Sliding windows
# Cropping datasets

print("### 7️⃣  Extracting Sub-Matrix (VERY IMPORTANT)")
sub = mat[0:2, 1:3]   # mat[row_start:row_end , col_start:col_end]
print(sub)
print("--------------------------------\n")

"""
### 1️⃣ What does mat[0:2, 1:3] mean?

NumPy slicing format for 2D arrays is:
mat[row_start:row_end , col_start:col_end]
Both row_end and col_end are exclusive.

So:
🔹 0:2 → rows 0 and 1
(Stops before row index 2)

🔹 1:3 → columns 1 and 2
(Stops before column index 3)

"""
# -----------------------------------------------------------------------------------------------------------

### 8️⃣ Boolean Indexing (SUPER IMPORTANT in ML)

print("### 8️⃣  Boolean Indexing (SUPER IMPORTANT in ML)")
print(mat[mat > 50])
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

### 🔥 9️⃣ Fancy Indexing (important)

print("### 🔥9️⃣ Fancy Indexing (important)")

arr = np.array([10, 20, 30, 40, 50])
print(arr[[0, 2, 4]])   # pick selected positions
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

