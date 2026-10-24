"""
The broadcasting rule, precisely

When NumPy performs an operation between two arrays of different shapes, it compares their shapes dimension by dimension, starting from the rightmost (trailing) dimension and moving left. 
For each pair of dimensions being compared, they're compatible if:

they're equal, or
one of them is 1

If one array has fewer dimensions than the other, NumPy conceptually pads the shorter shape with 1's on the left until both have the same number of dimensions. This padding isn't 
something you do — it's implicit, just how the comparison works.
"""

"""
Task:
Broadcasting drill: Given a matrix of shape (5,3) and a vector of shape (3,), subtract the vector from every row without a loop. Then given a matrix of shape (5,3) 
and a vector of shape (5,), subtract it from every column without a loop (you'll need to reshape the vector — figure out to what shape, don't look it up first, then verify).

"""

import numpy as np
import time

def matix_vector_multiplication():
    matrix = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
        [11, 12, 13],
        [14, 15, 16]
    ])
    print("\nMatrix Shape: ", matrix.shape)
    print("\nMatrix:\n", matrix)
    print()
    print("\n--------------------------------\n")
    # --------------------------------------------------------------------------------------

    """
    np.array([1, 2, 3]) is built from a single flat Python list — no nested lists. NumPy looks at that structure and creates a 1-dimensional array: there's only one axis, 
    and that axis has length 3. That's genuinely different from a 1×3 matrix, which is 2-dimensional (it has two axes — one of length 1, one of length 3) even though it holds 
    the same 3 numbers.

    Notice the printed output itself gives a visual clue: vector_1d prints as [1 2 3] — a single set of brackets. row_matrix prints as [[1 2 3]] — double brackets, because it's a list 
    containing one list. That extra layer of nesting is exactly what tells NumPy "this has 2 axes, not 1."

    So the honest answer to Q1: vector.shape is (3,) because that's genuinely how you constructed it — as a flat, single-axis object. It's not that NumPy is hiding a "1" from you; 
    there simply is no second axis to report. (3,) isn't shorthand for (1,3) — they're structurally different objects, even though a human might think of both as "a list of 3 numbers.
    " The trailing comma in (3,) is Python's tuple syntax for "a tuple with exactly one element" — it's confirming, not omitting, that there's only one axis.
    """
    vector = np.array([1, 2, 3])
    print("\nVector1 Shape: ", vector.shape)
    print("\nVector1:\n", vector)
    print()
    print("\n--------------------------------\n")
    # --------------------------------------------------------------------------------------

    """
    Notice the double square brackets: [[1, 2, 3]]. That's a list containing one inner list. NumPy reads the outer list as "how many rows" (1) and the inner list as "how many columns" (3). 
    This is the most direct way to construct a 2D array from scratch — you're explicitly telling NumPy about two levels of nesting, so it gives you two axes. 
    Compare this to np.array([1, 2, 3]) from before — single brackets, single axis, shape (3,).
    """
    vector2 = np.array([[1, 2, 3]])
    print("\nVector2 Shape: ", vector2.shape)
    print("\nVector2:\n", vector2)
    print()
    print("\n--------------------------------\n")
    # --------------------------------------------------------------------------------------

    vector3 = np.array([[1], [2], [3]])
    print("\nVector3 Shape: ", vector3.shape)
    print("\nVector3:\n", vector3)
    print()
    print("\n--------------------------------\n")
    # --------------------------------------------------------------------------------------

    """
    This starts from your existing 1D vector and inserts a new axis at position 0 (the front). np.newaxis is literally just an alias for None — it's a signal to NumPy's 
    indexing machinery: "add a dimension of size 1 right here." The : after the comma means "keep the existing axis exactly as it is." So you're explicitly saying: 
    add a row-axis in front of, and preserve, the column-axis I already have.
    """
    vector4 = vector[np.newaxis, :]
    print("\nVector1 is added a new axis using 'np.newaxis'. Updated Shape: ", vector4.shape)
    print("\nVector4:\n", vector4)
    print()
    print("\n--------------------------------\n")
    # --------------------------------------------------------------------------------------

    """
    .reshape() takes the same 3 underlying values and re-labels how they're organized into axes — here, explicitly "1 row, 3 columns." This works because reshape only requires 
    that the total element count match (1×3 = 3, same as the original 3 elements) — it doesn't care what the original shape was, just that the new shape holds the same number of values.
    """
    vector5 = vector.reshape(1, 3)
    print("\nVector1 is added a new axis using 'vector.reshape'. Updated Shape: ", vector5.shape)
    print("\nVector5:\n", vector5)
    print()
    print("\n--------------------------------\n")
    # --------------------------------------------------------------------------------------

    """
    This is the actual broadcasting operation. Internally, NumPy runs the alignment check we walked through: pad (3,) on the left to get (1, 3), compare against (5, 3) right-to-left 
    — rightmost (3 vs 3) matches exactly, next (5 vs 1) is compatible because of the 1 — so it's allowed. NumPy then conceptually stretches vector into a virtual 5×3 array 
    (same 3 values repeated in every row) and subtracts element-by-element. You get this without writing any loop yourself — the looping happens inside NumPy's compiled C code 
    instead of the Python interpreter, which is also why it's fast.
    """
    result = matrix - vector
    print("\nResult: \n", result)
    print()
    print("\n--------------------------------\n")
    # --------------------------------------------------------------------------------------

    """
    This creates a new array with the same shape and dtype as matrix, filled with zeros — a container to fill in as we go. zeros_like is convenient because you don't have 
    to separately specify the shape (5, 3) yourself; it copies it from matrix.
    """
    manual_result = np.zeros_like(matrix)
    print("\nZero matrix with shape as that of previous matrix (5*3): \n", manual_result)
    print()
    print("\n--------------------------------\n")
    # --------------------------------------------------------------------------------------

    """
    This is the "prove it by hand" step — deliberately doing with an explicit loop what broadcasting did automatically, so you can see they're equivalent. matrix.shape[0] is 5 
    (the row count), so i runs 0, 1, 2, 3, 4. On each iteration, matrix[i] pulls out one full row (shape (3,) — same shape as vector), so matrix[i] - vector is a plain, 
    ordinary element-wise subtraction between two equally-shaped 1D arrays — no broadcasting rule even needed here, since shapes already match exactly. 
    That row-result gets stored into manual_result[i].
    """
    for i in range(matrix.shape[0]):
        manual_result[i] = matrix[i] - vector

    comparision_result = np.array_equal(result, manual_result)    
    print("\nComparing results of both approaches: ", comparision_result)
    print()
    print("\n--------------------------------\n")
    # --------------------------------------------------------------------------------------    


"""
Task:
Deliberately write code that triggers a broadcasting shape-mismatch error (e.g., try to add a (5,3) array and a (4,) array). Read the actual error message. 
Write one sentence explaining what it means in your own words.
"""
def matix_vector_wrong_multiplication():

    # np.ones(shape) creates an array filled entirely with 1.0s, in the given shape. It's a quick way to get an array with a specific shape without caring about the actual values ....
    array_a = np.ones((5, 3))

    # Same idea, but shape (4,) — a 1D array of four 1.0s ....
    array_b = np.ones((4,))

    try:
        """
        Reasoning: Shapes: (5, 3) and (4,).
        Pad (4,) on the left with a 1: → (1, 4)

        Compare right to left against (5, 3):

        Rightmost position: matrix has 3, vector (padded) has 4. Equal? No. Is either one 1? No. → Incompatible, immediately.
        """
        result = array_a + array_b
    except ValueError as e:
        print("Error type:", type(e).__name__) # type(e).__name__ gets the class name of the exception (ValueError) — confirming what kind of error it is ....
        print("Error message:", e) # print(..., e) prints the actual message text NumPy attached to the exception — the human-readable explanation ....


def main():
    matix_vector_multiplication()
    matix_vector_wrong_multiplication()
 
 
if __name__ == "__main__":
    main()