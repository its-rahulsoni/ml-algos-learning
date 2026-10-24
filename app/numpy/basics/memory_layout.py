import os
import numpy as np
import time

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

# -----------------------------------------------------------------------------------------------------------

# 📌 Check Memory Layout of an Array

a = np.array([[1, 2, 3],
              [4, 5, 6]])

print("📌 Check Memory Layout of an Array\n")
print(a.flags)
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

# 📌 Create Arrays in Specific Layouts

print("# 📌 Create Arrays in Specific Layouts\n")

# C-order (default)
print("C-order (default)\n")
a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(a.flags)

# F-order (column-major)
print("F-order (column-major)\n")
b = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print("F_CONTIGUOUS: ", b.flags['F_CONTIGUOUS'])   # True
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------

# 🧪 Demonstration of performance impact

print("🧪 Demonstration of performance impact\n")
a_c = np.random.rand(10000, 10000)        # C-order
a_f = np.asfortranarray(a_c)              # convert to F-order

# Column sum on C-order
start = time.time()
a_c.sum(axis=0)
print("Column sum on C-order:", time.time() - start)

# Column sum on F-order
start = time.time()
a_f.sum(axis=0)
print("Column sum on F-order:", time.time() - start)
print("--------------------------------\n")

# -----------------------------------------------------------------------------------------------------------


