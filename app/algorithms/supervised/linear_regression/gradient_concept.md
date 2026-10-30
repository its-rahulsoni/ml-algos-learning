# The gradient is the two derivatives you already took

You did have the gradient. You just did not call it that. "Gradient" is the name for **the list of the derivatives you took, stacked in a column.** Nothing new was added.

## What you did by hand at the start

Loss: L = (1/n) · Σ (b + w·xᵢ − yᵢ)²

You took the derivative with respect to b and with respect to w, and got two expressions:

- dL/db = (2/n) · Σ residualᵢ
- dL/dw = (2/n) · Σ xᵢ · residualᵢ

Then you set both equal to zero and solved for b and w. That gave the two equations with the five sums, which became the normal equation.

## What the gradient is

Put those two derivatives in one column:

```
gradient = [ dL/db ]  =  [ (2/n) · Σ residual       ]
           [ dL/dw ]     [ (2/n) · Σ x · residual   ]
```

That column is the gradient. Same two expressions, just stacked.

## Where the matrix form comes from

Look at the two sums inside:

- Σ residual = row of ones times the residual column
- Σ x·residual = row of x times the residual column

Those are the two rows of Xᵀ times the residual column, which is exactly Xᵀ @ residuals. So stacking gives:

```
gradient = (2/n) · Xᵀ @ residuals
```

With your numbers, at θ = [0, 0], residuals = [−2, −2, −5]:

- dL/db = (2/3)·(−2 − 2 − 5) = (2/3)·(−9) = −6
- dL/dw = (2/3)·(1·−2 + 2·−2 + 3·−5) = (2/3)·(−21) = −14

These are the same numbers as the matrix version gave. The matrix form is shorthand for your two hand derivatives.

## How the two approaches connect

Both use the same two derivatives. They differ in what they do next:

- **Your hand method (normal equation):** set both derivatives = 0 and solve for b and w in one go.
- **Gradient descent:** don't solve. Evaluate both derivatives at the current guess, and step against them: θ_new = θ − α · gradient. Repeat until they are close to 0.

So "gradient = 0" is the normal equation, and "walk until the gradient is ~0" is gradient descent. At the best θ = [0, 1.5], both derivatives are 0, which you verified in step 14 of the hand calculation.

## One-line version for your code comments

```
# gradient = [dL/db, dL/dw] = the two derivatives from the paper derivation, stacked in a column
#          = (2/n) · Xᵀ @ residuals   (the two sums are the two rows of Xᵀ times residuals)
# normal equation: set gradient = 0 and solve. gradient descent: step against it and repeat.
```

## Optional check

Do your original hand derivative on one point, (1, 2), at θ = [0, 0]: the residual is −2, so dL/db = 2·(−2)·1 = −4 and dL/dw = 2·(−2)·1 = −4. The matrix form on that single point gives the same −4 and −4.