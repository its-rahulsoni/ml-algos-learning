"""
Softmax From Scratch: Naive vs. Numerically Stable
====================================================
Demonstrates the overflow problem with a naive softmax implementation
on large input values, then fixes it using the max-subtraction trick.
"""

import numpy as np


def softmax_naive(x):
    """Naive softmax: exponentiate directly, then normalize. Prone to overflow."""
    exp_values = np.exp(x)
    return exp_values / np.sum(exp_values)


def softmax_stable(x):
    """
    Numerically stable softmax: subtract the max value before exponentiating.
    Mathematically identical result to softmax_naive, but avoids overflow.
    """
    shifted_x = x - np.max(x)
    exp_values = np.exp(shifted_x)
    return exp_values / np.sum(exp_values)


def run_demo():
    small_vector = np.array([1.0, 2.0, 3.0])
    large_vector = np.array([1.0, 2.0, 1000.0])

    print("=" * 60)
    print("Test 1: Naive softmax on a SMALL vector (should work fine)")
    print("=" * 60)
    print("input:", small_vector)
    result_small_naive = softmax_naive(small_vector)
    print("naive softmax output:", result_small_naive)
    print("sum of output (should be 1.0):", np.sum(result_small_naive))

    print("\n" + "=" * 60)
    print("Test 2: Naive softmax on a LARGE-number vector (watch for nan/inf)")
    print("=" * 60)
    print("input:", large_vector)
    result_large_naive = softmax_naive(large_vector)
    print("naive softmax output:", result_large_naive)
    print("contains nan:", np.any(np.isnan(result_large_naive)))
    print("contains inf:", np.any(np.isinf(result_large_naive)))

    print("\n" + "=" * 60)
    print("Test 3: Stable softmax on the SAME large-number vector")
    print("=" * 60)
    print("input:", large_vector)
    result_large_stable = softmax_stable(large_vector)
    print("stable softmax output:", result_large_stable)
    print("sum of output (should be 1.0):", np.sum(result_large_stable))
    print("contains nan:", np.any(np.isnan(result_large_stable)))
    print("contains inf:", np.any(np.isinf(result_large_stable)))

    print("\n" + "=" * 60)
    print("Test 4: Confirm naive and stable versions AGREE on the small vector")
    print("=" * 60)
    result_small_stable = softmax_stable(small_vector)
    print("naive result: ", result_small_naive)
    print("stable result:", result_small_stable)
    print("match (within tolerance):", np.allclose(result_small_naive, result_small_stable))


if __name__ == "__main__":
    run_demo()