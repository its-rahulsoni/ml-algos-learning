import numpy as np

# Our fruit example: 1=banana, 0=apple, 2=cherry
# Creates a 1D NumPy array from a plain Python list [1, 0, 2, 2, 1]. Each number represents one fruit's category: recall our mapping is 1=banana, 0=apple, 2=cherry. 
# So this array represents the sequence: banana, apple, cherry, cherry, banana.
labels = np.array([1, 0, 2, 2, 1])
print("labels:", labels)
print("(meaning: banana, apple, cherry, cherry, banana)")

"""
We manually set num_classes = 3 here because we already know, by design, that there are exactly 3 possible fruit categories (0, 1, 2). This tells NumPy how big to make the 
identity matrix in the next step — it needs one row/column per possible category, not per label in our specific labels array (which might not even contain every category, 
or might contain duplicates).
"""
num_classes = 3   # apple, banana, cherry -- 3 total categories

# np.eye(3) builds a 3×3 identity matrix — 1s on the diagonal, 0s everywhere else ....
identity = np.eye(num_classes)

# Method 1: Specify dtype directly when creating the array ....
identity_int = np.eye(3, dtype=int)
# Method 2: Convert an existing array using .astype() ....
identity_float = np.eye(3)
identity_int = identity_float.astype(int)

# Notice these values print with a trailing dot (1. not 1) — np.eye() produces floating-point numbers by default, even though conceptually we're just thinking of them as 0s and 1s ....
print("\nidentity matrix (3x3):")
print(identity)
print("row 0 =", identity[0], "  <- one-hot for label 0 (apple)")
print("row 1 =", identity[1], "  <- one-hot for label 1 (banana)")
print("row 2 =", identity[2], "  <- one-hot for label 2 (cherry)")

"""
This starts a loop over labels. enumerate(labels) is a Python built-in that, for each item in labels, gives you both its position (i) and its value (label) together, as a pair, 
n each iteration. So the loop runs 5 times (since labels has 5 elements):

iteration 1: i=0, label=1
iteration 2: i=1, label=0
iteration 3: i=2, label=2
iteration 4: i=3, label=2
iteration 5: i=4, label=1

Without enumerate, a plain for label in labels: would give you just the values, not their positions — we want both here so we can print "at position i, the label is label."
"""
print("\n--- Now fetch rows one at a time, matching each label ---")
for i, label in enumerate(labels):
    print(f"labels[{i}] = {label}  ->  identity[{label}] = {identity[label]}")
# This is an f-string (formatted string) — the f before the quotes means anything inside {curly braces} gets evaluated as a Python expression and inserted into the string as text ....


"""
This is the actual one-hot encoding step. Instead of indexing with a single number (like identity[0]), we index with the entire labels array at once. NumPy interprets this as: 
"for every value inside labels, go fetch that row of identity, and stack all the fetched rows together into a new array, in the same order as labels." This produces, 
in a single expression, exactly the same 5 rows the loop above printed one at a time — but now they're collected together into one 2D array of shape (5, 3) 
(5 labels, 3 columns per one-hot vector), stored in result.
"""
print("\n--- Now do it all at once using fancy indexing ---")
result = identity[labels]
print("identity[labels] =")
print(result)

print("\nDoes this match our hand-written expected_output? Let's check:")
expected_output = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
])
print(np.array_equal(result, expected_output))