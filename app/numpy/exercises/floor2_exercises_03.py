import os
import numpy as np

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

# TASK:
# Take a 3x4 array and flatten it, then reshape it to 4x3, then to 2x6. Predict the output before running each reshape — write your prediction as a comment, then check ....

# Create the 3×4 array ....
mat = np.arange(1, 13).reshape(3, 4) # np.arange(1, 13) gives numbers 1 through 12 ....
print("\n 3×4 Dimensional array: \n", mat)
print("\n--------------------------------\n")
# --------------------------------------------------------------------------------------

# Flatten this 3x4 array into a 1D array ....
# Flattening collapses a 2D array into 1D by reading it row by row, left to right, top to bottom (this reading order is called "row-major" or "C order," and it's NumPy's default) ....
flat = mat.flatten()
print("\n Flatten the 3×4 Dimensional array: \n", flat)
print("\n--------------------------------\n")
# --------------------------------------------------------------------------------------

# Reshape the flattened array to 4×3 ....
# Reshaping doesn't reorder the values — it just re-draws the boundaries of where each row starts and stops, while keeping the same left-to-right, top-to-bottom fill order ....
reshaped_4x3 = flat.reshape(4, 3)
#reshaped_4x3 = mat.reshape(4, 3) # Same result ....
# .reshape() as always secretly doing "flatten, then reshape" under the hood — the current shape is irrelevant, only the total count and the element order matter ....

print("\n Reshape the flattened array to 4×3 array: \n", reshaped_4x3)
print("\n--------------------------------\n")
# --------------------------------------------------------------------------------------

