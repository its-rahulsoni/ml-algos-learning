import os
import numpy as np

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

# TASK: 
# Create a 1D array of 20 numbers. Reverse it without using a loop. Get every 3rd element. Get all elements greater than 10 using boolean indexing (not a loop).

# Create a 1D array from 1 to 20
mat = np.arange(1, 21) # np.arange(start, stop): Generates numbers starting from start up to, but not including, stop.
print("\n 1-Dimensional array form 1 to 20 numbers: \n", mat)
print("\n--------------------------------\n")
# --------------------------------------------------------------------------------------
# Reversing this array ....

# Approach #01: With a loop — swap from both ends ....

reversed_mat = mat.copy()  # don't modify original, without .copy() it would be just another name pointing to old array. Any modification done to this array would aalso modify old array ....
print("\n New array created as copy of old one: \n", reversed_mat)
# What .copy() actually does:
#
# mat.copy() does two things at once:
# 01. Allocates a brand new block of memory, the same size and shape as mat
# 02. Copies every single value from mat into that new memory, in the same positions
#
# So it's not just "same size, empty" — it's "same size, filled with the same values."

left = 0
right = len(reversed_mat) - 1

while left < right:
    # This is a simultaneous swap — Python evaluates the right-hand side first (as a pair: (reversed_mat[right], reversed_mat[left])), then assigns both at once ....
    reversed_mat[left], reversed_mat[right] = reversed_mat[right], reversed_mat[left]
    left += 1
    right -= 1

print("\n Approach #01 - Reversed array using loop: \n", reversed_mat)
print("\n--------------------------------\n")
# --------------------------------------------------------------------------------------

# Approach #02: With a loop — build a new array by walking backward ....

reversed_mat = np.empty_like(mat)
for i in range(len(mat)):
    reversed_mat[i] = mat[len(mat) - 1 - i]

print("\n Approach #02 - Reversed array using loop: \n", reversed_mat)
print("\n--------------------------------\n")
# --------------------------------------------------------------------------------------

# Approach #03: Without a loop — slicing with a negative step ....

# This returns a view, not a copy. If you modify reversed_mat, it will also change mat, since they share the same underlying memory ....
#reversed_mat = mat[::-1]

# Us this approach, if you want an independent copy ....
reversed_mat = mat[::-1].copy()

# Logic: recall start:stop:step — leaving start and stop blank means "the whole array," and step = -1 means "walk backward." 
# So this reads as "give me the whole array, traversed from end to start." ....

print("\n Approach #03 - Without a loop — slicing with a negative step: \n", reversed_mat)
print("\n--------------------------------\n")
# --------------------------------------------------------------------------------------

# Approach #04: Without a loop — np.flip() ....

mat2 = np.array([
    [11, 12, 13, 14, 15],
    [21, 22, 23, 24, 25],
    [31, 32, 33, 34, 35],
    [41, 42, 43, 44, 45],
    [51, 52, 53, 54, 55]
    ])

# Flip along axis=0 (flip rows — upside down).
# This reverses the order of the rows, top becomes bottom. Each row's internal content stays the same — only which row appears where changes.
reversed_mat = np.flip(mat2, axis = 0)
print("\n Approach #04 - Without a loop — np.flip(mat2, axis = 0): \n", reversed_mat)

# Flip along axis=1 (flip columns — left to right).
# This reverses the order of the columns within each row — left becomes right. Each row is still in its original position; only the order of values inside each row is mirrored.
reversed_mat = np.flip(mat2, axis = 1)
print("\n Approach #04 - Without a loop — np.flip(mat2, axis = 1): \n", reversed_mat)

# Flip along both axes at once (180° rotation)
# If you leave out the axis argument entirely, np.flip() flips along every axis simultaneously — rows are reversed AND columns are reversed, which is the same as rotating the whole matrix 180 degrees.
reversed_mat = np.flip(mat2)
print("\n Approach #04 - Without a loop — np.flip(): \n", reversed_mat)

print("\n--------------------------------\n")
# --------------------------------------------------------------------------------------

# Approach #05: Without a loop — flipud (for completeness) ....

mat2 = np.array([
    [11, 12, 13, 14, 15],
    [21, 22, 23, 24, 25],
    [31, 32, 33, 34, 35],
    [41, 42, 43, 44, 45],
    [51, 52, 53, 54, 55]
    ])

reversed_mat = np.flipud(mat2)
print("\n Approach #05 - Without a loop — flipud (for completeness) \n", reversed_mat)

print("\n--------------------------------\n")
# --------------------------------------------------------------------------------------
