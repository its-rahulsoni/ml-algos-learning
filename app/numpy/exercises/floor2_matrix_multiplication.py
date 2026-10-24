"""
Matrix Multiplication: Manual Triple-Loop vs NumPy Vectorized
================================================================
Compares a hand-written O(n^3) triple-loop implementation against
NumPy's built-in @ / matmul operator, both for correctness (on a
small 3x3 example) and for speed (on 200x200 matrices).
"""
 
import numpy as np
import time


# I'll use np.random.default_rng(seed) rather than the older np.random.seed() — it's the modern numpy recommendation 
# because each generator instance is independent (no global state getting mutated from elsewhere in your program) ....

# Since you passed (rows, cols) as the shape, NumPy internally figures out it needs rows × cols numbers total, 
# generates that many raw values in a flat buffer (this is done in a fast C loop, not a Python loop — that's part of 
# why numpy's own random generation is fast), and then reshapes that flat buffer into a 2D array of the requested shape ....

"""
Seed: seed -> f(seed) -> f(f(seed)) -> f(f(f(seed))) -> ... so on. The seed is simply the starting point of that chain.
seed=42/10/11/13/56 -> Each of these will give you a different matrix (different starting point → different chain), but each one is individually 
reproducible — running seed=12 today and seed=12 next year gives the identical matrix both times. What matters isn't which 
number you pick, just that you picked one at all and kept it fixed if you want repeatability.
"""
def generate_random_matrix(rows, cols, seed=None):
    """Generate a random matrix of given shape using a seeded RNG."""
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols))


def matmul_loop(A, B):
    """Matrix multiplication using triple nested for loops (textbook definition)."""
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "Inner dimensions must match for multiplication"
 
    result = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            total = 0.0
            for k in range(m):
                total += A[i, k] * B[k, j]
            result[i, j] = total
    return result
 
 
def matmul_numpy(A, B):
    """Matrix multiplication using NumPy's @ operator (calls np.matmul under the hood)."""
    return A @ B
 
def verify_results(result1, result2, atol=1e-8):
    """Check two matrices are numerically equal within a small tolerance."""
    return np.allclose(result1, result2, atol=atol)

 
def time_function(func, *args, repeats=1):
    """Time a function call, averaged over `repeats` runs. Returns (last_result, avg_seconds)."""
    start = time.perf_counter()
    for _ in range(repeats):
        result = func(*args)
    end = time.perf_counter()
    avg_time = (end - start) / repeats
    return result, avg_time
 
def run_timing_benchmark(size=200):
    """Time both implementations on size x size random matrices."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Timing on {size}x{size} matrices")
    print("=" * 60)
 
    A = generate_random_matrix(size, size, seed=1)
    B = generate_random_matrix(size, size, seed=2)
 
    _, loop_time = time_function(matmul_loop, A, B, repeats=1)
    print(f"Triple-loop time:  {loop_time:.4f} seconds")
 
    _, numpy_time = time_function(matmul_numpy, A, B, repeats=10)
    print(f"NumPy (@) time:    {numpy_time:.6f} seconds (average of 10 runs)")
 
    speedup = loop_time / numpy_time
    print(f"\nNumPy is approximately {speedup:,.0f}x faster than the manual loop")
 
 
def run_small_verification():
    """Run both implementations on a small 3x3 example and confirm they agree."""
    print("=" * 60)
    print("STEP 1: Verification on a 3x3 example")
    print("=" * 60)
 
    A = generate_random_matrix(3, 3, seed=42)
    B = generate_random_matrix(3, 3, seed=7)
 
    result_loop = matmul_loop(A, B)
    result_numpy = matmul_numpy(A, B)
 
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("\nTriple-loop result:\n", result_loop)
    print("\nNumPy (@) result:\n", result_numpy)
 
    match = verify_results(result_loop, result_numpy)
    print(f"\nResults match within tolerance: {match}")
    return match
 

def main():
    run_small_verification()
    run_timing_benchmark(size=200)
 
 
if __name__ == "__main__":
    main()




