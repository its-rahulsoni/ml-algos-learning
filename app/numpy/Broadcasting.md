# NUMPY_MATH_BROADCASTING.md

## 🟢 1. Objective

Understand how NumPy handles:

* Element-wise math operations
* Scalar broadcasting
* Array-to-array broadcasting (row vectors, column vectors)
* The broadcasting compatibility rule
* A real use case: centering data by feature mean

---

## 🟢 2. Element-Wise Math Operations

NumPy applies operators to **every element individually**, position by position — no loop required.

```python id="elementwise1"
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

a + b   # [11, 22, 33, 44]
a - b   # [-9, -18, -27, -36]
a * b   # [10, 40, 90, 160]
a / b   # [0.1, 0.1, 0.1, 0.1]
```

* `a[0]` pairs with `b[0]`, `a[1]` with `b[1]`, and so on
* Runs in optimized C internally → this is why NumPy is fast

---

### 🔹 2.1 Plain Python Lists Behave Differently

```python id="listtrap1"
py_list_a = [1, 2, 3, 4]
py_list_b = [10, 20, 30, 40]

py_list_a + py_list_b
# [1, 2, 3, 4, 10, 20, 30, 40]   <- concatenation, NOT addition
```

> ⚠️ NumPy arrays behave mathematically. Plain Python lists do not.

---

## 🟢 3. Scalar Operations

A single number ("scalar") is applied to every element automatically.

```python id="scalar1"
a = np.array([1, 2, 3, 4])

a + 10   # [11, 12, 13, 14]
a * 2    # [2, 4, 6, 8]
a ** 2   # [1, 4, 9, 16]
```

This is the simplest form of **broadcasting** — a shape-`()` value stretched across shape-`(4,)`.

---

## 🟢 4. What Broadcasting Is

> **Broadcasting** = the rule set NumPy uses to operate on arrays of *different shapes*, by virtually stretching the smaller one to match the larger one — without copying data in memory.

```text id="broadcastflow1"
Scalar (10)
   ↓ stretched
[10, 10, 10, 10]
   ↓ added to
[1, 2, 3, 4]
   ↓ result
[11, 12, 13, 14]
```

---

## 🟢 5. Broadcasting: 2D Array + Row Vector

```python id="rowbroadcast1"
mat = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

row = np.array([10, 20, 30])

mat + row
```

```text id="rowbroadcast2"
row (shape (3,))
   ↓ stretched down every row
[[10, 20, 30],
 [10, 20, 30],
 [10, 20, 30]]
   ↓ added to mat
[[11, 22, 33],
 [14, 25, 36],
 [17, 28, 39]]
```

---

## 🟢 6. Broadcasting: 2D Array + Column Vector

```python id="colbroadcast1"
col = np.array([[100], [200], [300]])   # shape (3, 1)
mat = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])   # shape (3, 3)

mat + col
```

```text id="colbroadcast2"
col (shape (3,1))
   ↓ stretched across each row
[[101, 102, 103],
 [204, 205, 206],
 [307, 308, 309]]
```

> Row vector `(3,)` stretches **vertically** (down the rows).
> Column vector `(3,1)` stretches **horizontally** (across the columns).

---

## 🟢 7. Broadcasting Compatibility Rule

Shapes are compared **right to left**, dimension by dimension. Compatible if, for each pair:

* the dimensions are equal, OR
* one of them is `1`

| Shape A | Shape B | Compatible? | Why |
|---|---|---|---|
| (3, 3) | (3,) | ✅ | (3,) → (1,3), columns match |
| (3, 4) | (4,) | ✅ | columns match |
| (3, 4) | (3,) | ❌ | 4 ≠ 3, neither is 1 |
| (3, 1) | (1, 4) | ✅ | both broadcast to (3,4) |
| (5,) | (1,) | ✅ | 1 always broadcasts |

---

## 🟢 8. Real Use Case — Centering Data by Feature Mean

```python id="usecase1"
data = np.array([
    [18, 50000, 85],
    [20, 55000, 90],
    [22, 60000, 88],
    [19, 52000, 87]
])

feature_means = np.mean(data, axis=0)   # shape (3,)
centered = data - feature_means         # broadcasting!
```

```text id="usecase2"
data            shape (4, 3)
feature_means   shape (3,)
                   ↓ broadcast across all 4 rows
centered        shape (4, 3)
```

No explicit loop needed — every column's mean is subtracted from every value in that column automatically.

---

## 🟢 9. Quick Reference

| Concept | Meaning |
|---|---|
| Element-wise ops | `+ - * /` apply position-by-position on same-shaped arrays |
| Scalar broadcasting | A single number stretched to match every element |
| Array broadcasting | A smaller-shaped array stretched along matching dimensions |
| Compatibility rule | Compare shapes right→left; dims must match or be 1 |
| Why it matters | Skip explicit loops for normalization, scaling, centering data |