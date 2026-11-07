# NumPy: library for arrays and math. "np" is the standard nickname.
# Needed here for np.inf (infinity) and np.mean (average).
import numpy as np

# sklearn's ready-made logistic regression model.
from sklearn.linear_model import LogisticRegression

# Reuse my own code from logistic_regression_basics.py:
#   make_data     -> builds the two overlapping Gaussian blobs
#   predict_proba -> sigmoid(X @ w + b): probabilities in [0, 1]
#   train         -> my gradient descent loop, returns w, b, losses
from logistic_regression_basics import make_data, predict_proba, train

# X has shape (200, 2): two features for each of 200 points.
# y has shape (200,): flat array of labels, 0 or 1.
X, y = make_data()

# Train my model: learning rate 0.1, 1000 iterations.
# w has shape (2,), b is one float, losses is the loss at each iteration.
w, b, losses = train(X, y, lr=0.1, n_iters=1000)

# The fair setting. sklearn applies L2 regularization by default (C=1.0),
# which punishes large weights. My loop has no penalty, so for a fair
# comparison we switch it off. C is the INVERSE of the penalty strength,
# so C = infinity means no penalty: both minimize plain log-loss.
model = LogisticRegression(C=np.inf)

# fit() runs sklearn's own optimizer (not my fixed-learning-rate loop)
# until it converges to the bottom of the loss valley.
model.fit(X, y)

# Shapes matter:
#   model.coef_       has shape (1, 2) -> a table with ONE row
#                     (sklearn supports many classes, one row per class).
#                     [0] picks that row, giving shape (2,) like my w.
#   model.intercept_  has shape (1,)   -> flat array with one number.
#                     [0] picks that number, giving a plain float.
w_sk = model.coef_[0]
b_sk = model.intercept_[0]

# Compare weights and bias. Expect same signs and similar ratios, but
# sklearn's weights are larger: it fully converged, while mine was still
# creeping down at iteration 1000.
print("sklearn w:", w_sk, " b:", b_sk)
print("mine    w:", w, " b:", b)

# Boundary slope and intercept, from w1*x1 + w2*x2 + b = 0
# solved for x2: x2 = -(w1/w2)*x1 - b/w2.
#   slope     = -w1/w2   (w_sk[0] is w1, w_sk[1] is w2)
#   intercept = -b/w2
# These depend only on the RATIOS, so different weight sizes can still
# give nearly the same line.
print("sklearn slope, intercept:", -w_sk[0]/w_sk[1], -b_sk/w_sk[1])
print("mine    slope, intercept:", -w[0]/w[1], -b/w[1])

# Accuracy for both.
# model.score(X, y) is sklearn's built-in accuracy (fraction correct).
print("sklearn acc:", model.score(X, y))

# For mine, step by step:
#   predict_proba(X, w, b) -> probabilities, shape (200,)
#   (... >= 0.5)           -> True/False per row: "predicts class 1?"
#   (y == 1)               -> True/False per row: "is truly class 1?"
#   ==                     -> True where prediction and truth agree
#   np.mean of booleans    -> True counts as 1, False as 0, so the
#                             result is the fraction correct.
print("mine    acc:", np.mean((predict_proba(X, w, b) >= 0.5) == (y == 1)))