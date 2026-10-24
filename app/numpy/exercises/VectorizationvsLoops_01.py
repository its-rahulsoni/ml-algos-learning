"""
Exercise: Compare a pure-Python (for-loop) implementation of mean/variance
against NumPy's vectorized np.mean() / np.var(), and time both on a
1,000,000-element array.
"""
import time
import timeit
import numpy as np

def mean_variance_loop(arr):
    """
    Compute mean and (population) variance of a 1D array using a for loop.
    """

    n = len(arr)

    # --- mean ---
    total = 0.0
    for x in arr:
        total += x

    mean = total / n

    # --- variance ---
    sq_diff_sum = 0.0
    for x in arr:
        sq_diff_sum += (x - mean) ** 2

    variance = sq_diff_sum / n

    return mean, variance

def mean_variance_numpy(arr):
    return np.mean(arr), np.var(arr)


def time_with_time_module(data_list, data):
    """
    Time both implementations using time.time() (single run each).
    """
    start = time.time()
    loop_mean, loop_var = mean_variance_loop(data_list)
    loop_time = time.time() - start
 
    start = time.time()
    np_mean, np_var = mean_variance_numpy(data)
    np_time = time.time() - start

    # The basic mechanism: f-strings
    # The f before the opening quote (f"...") makes it an f-string (formatted string literal). Anything inside { } gets evaluated as a Python expression and its result gets inserted 
    # into the string.

    # The .6f   part — format specification:
    # -> 'f'  — display the number as a fixed-point decimal (not scientific notation, not an integer).
    # -> '.6' — show exactly 6 digits after the decimal point.
 
    print("--- Using time.time() ---")
    print(f"Loop  -> mean: {loop_mean:.6f}, variance: {loop_var:.6f}, time: {loop_time:.4f} s")
    print(f"NumPy -> mean: {np_mean:.6f}, variance: {np_var:.6f}, time: {np_time:.6f} s")
    print(f"Speedup (loop_time / numpy_time): {loop_time / np_time:.1f}x")
    print()
    print("\n--------------------------------\n")
    # --------------------------------------------------------------------------------------
 
    return loop_time, np_time
 
 
def time_with_timeit_module(data_list, data, number=5):
    """
    Time both implementations using timeit.timeit() (averaged over
    'number' runs each), then do one plain call to get actual values
    to print (timeit only returns timing, not the function's result).
    """
    loop_time = timeit.timeit(lambda: mean_variance_loop(data_list), number=number) / number
    np_time = timeit.timeit(lambda: mean_variance_numpy(data), number=number) / number
 
    loop_mean, loop_var = mean_variance_loop(data_list)
    np_mean, np_var = mean_variance_numpy(data)
 
    print("--- Using timeit.timeit() ---")
    print(f"Loop  -> mean: {loop_mean:.6f}, variance: {loop_var:.6f}, time: {loop_time:.4f} s")
    print(f"NumPy -> mean: {np_mean:.6f}, variance: {np_var:.6f}, time: {np_time:.6f} s")
    print(f"Speedup (loop_time / numpy_time): {loop_time / np_time:.1f}x")
    print()
    print("\n--------------------------------\n")
    # --------------------------------------------------------------------------------------

    return loop_time, np_time


def main():
    data = [1, 2, 3, 4, 5]
    mean, var = mean_variance_loop(data)
    print(f"Loop -> mean: {mean}, variance: {var}")

    mean, var = mean_variance_numpy(data)
    print(f"NumPy -> mean: {mean}, variance: {var}")
    print("\n--------------------------------\n")
    # --------------------------------------------------------------------------------------

    # Create a test array of 1,000,000 elements
    np.random.seed(0)
    data = np.random.rand(1_000_000)

    # For the pure-Python loop version, use a plain list so we are
    # timing "regular Python" iteration, not NumPy's optimized iterator.
    data_list = data.tolist()
 
    # --- Time the for-loop version ---
    start = time.time()
    loop_mean, loop_var = mean_variance_loop(data_list)
    loop_time = time.time() - start
 
    # --- Time the NumPy version ---
    start = time.time()
    np_mean, np_var = mean_variance_numpy(data)
    np_time = time.time() - start
 
    print(f"Loop  -> mean: {loop_mean:.6f}, variance: {loop_var:.6f}, time: {loop_time:.4f} s")
    print(f"NumPy -> mean: {np_mean:.6f}, variance: {np_var:.6f}, time: {np_time:.6f} s")
    print(f"Speedup (loop_time / numpy_time): {loop_time / np_time:.1f}x")

    print("\n--------------------------------\n")
    # --------------------------------------------------------------------------------------

    time_with_time_module(data_list, data)
    time_with_timeit_module(data_list, data)
 
    # ------------------------------------------------------------------
    # SPEED DIFFERENCE:
    # NumPy is consistently faster than the pure-Python for loop because
    # np.mean() and np.var() run in compiled C code over contiguous
    # memory using vectorized operations, avoiding the per-element
    # interpreter overhead (type checks, dynamic dispatch) that the
    # manual loop pays on every single element. timeit's averaged
    # result is more reliable than a single time.time() measurement.
    # ------------------------------------------------------------------


# Every Python module has a built-in variable called __name__.
#
# If you run the file directly (python myfile.py), Python sets __name__ = "__main__".
# If you import the file from another file (import myfile), Python sets __name__ = "myfile" instead.
#
# So that if check means: "only call main() when this file is being run directly, not when someone else is importing my functions."
if __name__ == "__main__":
    main()    