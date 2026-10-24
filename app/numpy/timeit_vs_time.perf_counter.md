# timeit vs time.perf_counter()

Good catch — this is a legitimate design choice worth explaining, not just a stylistic pick.

## What `timeit` actually is

`timeit` is a module specifically built for **micro-benchmarking small, fast pieces of code**. It does several things automatically that you'd otherwise have to hand-roll:

1. **Runs your code many times in a loop** (you specify `number=...`) and by default reports either total or best/average time.
2. **Disables the garbage collector** during the timed run — Python's automatic memory cleanup can kick in mid-measurement and add noise; `timeit` temporarily turns it off so GC pauses don't pollute your numbers.
3. **Takes the minimum time across repeated trials** when you use `timeit.repeat()` — the idea being that the *fastest* run is closest to the "true" cost of the code, since slower runs are usually explained by external noise (OS scheduling a different process, cache eviction, etc.), not the code itself getting slower.
4. Designed for **short snippets** — often passed as a string, executed thousands of times, to average out noise for something that takes microseconds.

## What `time.perf_counter()` actually is

`perf_counter()` is much simpler — it's just a **clock**. It returns the current value of the highest-resolution monotonic timer your OS provides. It does nothing else: no repetition, no GC handling, no statistics. You call it before and after your code, subtract, and that's your elapsed time. All the methodology (how many times to run, how to average) is left entirely up to you.

## Why `timeit` made sense in the mean/variance exercise

In that exercise you were almost certainly comparing something like a Python-loop-based mean/variance calculation against a numpy vectorized one, where **both operations were fast** (milliseconds or less). At that speed, a single measurement is dominated by noise — things like Python's startup jitter, OS scheduling, or cache warm-up can matter more than the actual computation. `timeit`'s repeated-trials-and-take-the-minimum approach was the right tool because it filters that noise out and gets you a clean, stable number for a short-running operation.

## Why `perf_counter()` made more sense here

The asymmetry in this exercise is the key difference: the triple-loop version takes **~2 seconds**, while the NumPy version takes **~0.0008 seconds** — roughly a 2,500x gap.

- For the **loop version**, 2 seconds is already long enough that OS noise is a rounding error by comparison — a single `perf_counter()` measurement is already stable and reliable. Running it many times via `timeit` would just mean waiting 2 seconds × N for no real benefit, and would make the script annoyingly slow to run.
- For the **NumPy version**, I did still want multiple runs to smooth out noise (that's why I wrote a `repeats=10` parameter and averaged manually) — which is honestly `timeit`'s job by design. I could have used `timeit.timeit(lambda: matmul_numpy(A, B), number=10)` and gotten an equally valid, probably slightly more rigorous number.

So the honest answer: **I could have used `timeit` for the NumPy half**, and it would arguably have been the more "correct" choice there for the same reason as your previous exercise (fast operation, want noise-averaging). Using `perf_counter()` for the loop version was reasonable given how slow it already is. Mixing tools like this in one script is a bit inconsistent though — here's a cleaner version using `timeit` throughout, which also fixes that inconsistency.

Here's how the same timing function looks rewritten with `timeit`, so you can compare both approaches directly:

```python
import timeit

def time_function_v2(func, *args, repeats=1):
    """Time a function call using timeit, averaged over `repeats` runs."""
    timer = timeit.Timer(lambda: func(*args))
    total_time = timer.timeit(number=repeats)   # runs `repeats` times, GC disabled
    avg_time = total_time / repeats
    return avg_time
```

Key differences you'd notice using this instead:

- `timer.timeit(number=repeats)` returns **total** time for all runs combined, not per-run — you divide yourself, same as before.
- Garbage collection gets disabled automatically for the duration, which `perf_counter()` never does — this matters more for the loop version, actually, since it allocates a lot of small Python objects (the running `total` float gets reassigned 8 million times) that could trigger GC mid-run and skew a single `perf_counter()` reading slightly.
- It doesn't return the function's actual result — only the timing — because `timeit` is built purely for benchmarking, not for capturing output. That's a real limitation if you also need the computed matrix, which is part of why I used the simpler `perf_counter()` wrapper in the original: it hands back both the result and the timing together in one call.

**Bottom line**: for the 200x200 case, either tool gives you a trustworthy number — the gap between 2 seconds and 0.0008 seconds is far too large for measurement noise to matter. `timeit` earns its keep more clearly when the two things you're comparing are close enough in speed that noise could plausibly flip the comparison — which was likely the case in your mean/variance exercise.