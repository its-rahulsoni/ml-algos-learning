import os
import numpy as np

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

# TASK: 
# Create a 5x5 array of random integers. Extract: the second row, the last column, a 2x2 sub-block from the corner, every other row.

mat = np.array([
    [11, 12, 13, 14, 15],
    [21, 22, 23, 24, 25],
    [31, 32, 33, 34, 35],
    [41, 42, 43, 44, 45],
    [51, 52, 53, 54, 55]
    ])

print("###Creating a 5*5 Matrix")
print("Multi Dim Arr: \n", mat)
print("Array Dimensions: ", mat.shape)
print("Row #02: ", mat[1])
print("Row #02: ", mat[1, :])
print("Row #02 (only first 2 elements): ", mat[1, :2])
print("Row #02 (skips first 2 elements): ", mat[1, 2:])
print("\n--------------------------------\n")
# --------------------------------------------------------------------------------------


print("Last Column: ", mat[:, 4]) # Note: the format is matrix[row, column] ....
print("Last Column (using -1): ", mat[:, -1]) # Note: '-1' here denotes the last ....
print("\n--------------------------------\n")
# --------------------------------------------------------------------------------------


print("A 2x2 sub-block from top corner: \n", mat[0:2, 0:2]) # mat[row_start:row_stop, col_start:col_stop] ....
print("A 2x2 sub-block from bottom corner: \n", mat[3:5, 3:5]) # mat[row_start:row_stop, col_start:col_stop] ....
print("\n--------------------------------\n")
# Explanation for the above 2 print statements logic is below: 
# The core rule: start:stop (stop is excluded)
#
# Every slice start:stop means: "begin at index start, and keep going up to — but not including — index stop."
# That "stop is excluded" part is the single most important thing to internalize. It trips up almost everyone at first.
#
# Applying it to rows: 3:5
# Your matrix has row indices 0, 1, 2, 3, 4.
#
# 3:5 means "start at row index 3, stop before row index 5." Since there is no row index 5 (the last one is 4), this just means "give me row 3 and row 4" — rows 3 through 4 inclusive.
# row3:    41   42   43   44   45   ← included
# row4:    51   52   53   54   55   ← included
# Applying it to columns: 3:5
#
# Same logic, but now on the column axis. Column indices are also 0, 1, 2, 3, 4.
#
# 3:5 means "start at column index 3, stop before column index 5" — so it grabs column 3 and column 4.
#         col3 col4
#          44   45   ← from row3
#          54   55   ← from row4
# Putting it together
# mat[3:5, 3:5] = "rows 3 through 4, AND columns 3 through 4" = the bottom-right 2×2 block:
# [[44, 45],
#  [54, 55]]
#
# The comma is what separates the two independent instructions — one for rows, one for columns. NumPy applies them like a coordinate system: mat[row_start:row_stop, col_start:col_stop].
# --------------------------------------------------------------------------------------

print("\n1::2 -> 'Start at index 1, go to the end, step by 2 (only take every 2nd index)' → picks up the odd-indexed rows instead: \n", mat[1::2])
print("\n::2, : -> Every other row, every column: \n", mat[::2, :])
print("\n:, ::2 ->Every column, every other row — same thing written differently, but let's do every other column instead: \n", mat[:, ::2])
print("\n::2, ::2 -> Every other row AND every other column at once:: \n", mat[::2, ::2])
print("\n--------------------------------\n")

# The three knobs
#
# A slice has three parts, separated by colons:
# start : stop : step
# start — where to begin (inclusive)
# stop — where to end (exclusive — never included)
# step — how many indices to jump each time
#
# Any of the three can be left out, and NumPy fills in a sensible default.
#
# Defaults when you omit something
# You write	        NumPy reads it as
# :	            start=0, stop=end, step=1 (everything)
# 2:	        start=2, stop=end, step=1
# :3	        start=0, stop=3, step=1
# ::2	        start=0, stop=end, step=2
# 1:8:2	        start=1, stop=8, step=2

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# Quick rule of thumb
#
# NumPy 2D indexing always follows the pattern mat[row_selector, column_selector]:
#
# A single number → one row or column
# : → everything along that axis
# start:stop → a range (stop is excluded)
# start:stop:step → a range with skipping
# Negative numbers count from the end (-1 is last, -2 is second-to-last)
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

