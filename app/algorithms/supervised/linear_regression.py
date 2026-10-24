import os
import numpy as np

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

print("**** Ml Algo: Linear Regression with Equation: XᵀXθ = Xᵀy ****")

"""
Step #01: We are creating required arrays ....
"""
x = np.array([1, 2, 3])
y = np.array([2, 2, 5])

col_ones = np.ones(len(x)) 

X = np.column_stack((col_ones, x))

print("X: \n", X)

y = np.array([
    [2],
    [2],
    [5]])

print("y: \n", y)    

# Note: This is the java equivalent of printing multiple values in a print statement ....
print("Shapes of each matrix. x: {}, y: {}, X: {}".format(x.shape, y.shape, X.shape))

# Assertions ....
n = len(x)
assert X.shape == (n, 2), f"X should be ({n}, 2), got {X.shape}"
assert y.shape == (n, 1), f"y should be ({n}, 1), got {y.shape}"


"""
Step #02: We are performing matrix multiplications ....
"""

matrix_mul_a = X.T @ X
print("matrix_mul_a: \n", matrix_mul_a, " ShapeA: ", matrix_mul_a.shape)

matrix_mul_b = X.T @y
print("matrix_mul_b: \n", matrix_mul_b, " ShapeB: ", matrix_mul_b.shape)

# Assertions ....
assert matrix_mul_a.shape == (2, 2), f"XtX should be (2, 2), got {matrix_mul_a.shape}"
assert matrix_mul_b.shape == (2, 1), f"Xty should be (2, 1), got {matrix_mul_b.shape}"

assert np.allclose(matrix_mul_a, [[3, 6], [6, 14]]), f"XtX values wrong:\n{matrix_mul_a}"
assert np.allclose(matrix_mul_b, [[9], [21]]), f"Xty values wrong:\n{matrix_mul_b}"

"""
Step #03: We are calculating 'θ' from the Equation: XᵀXθ = Xᵀy ....
"""
theta = np.linalg.solve(matrix_mul_a, matrix_mul_b)
print("theta: \n", theta, " theta shape: ", theta.shape)

assert np.allclose(theta, [[0], [1.5]]), f"XtX values wrong:\n{theta}"