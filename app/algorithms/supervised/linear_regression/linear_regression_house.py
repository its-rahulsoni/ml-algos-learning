# linear_regression_house.py
# Linear regression with standardizing: age, bedrooms, plot_area -> price (lakhs)
# Self-contained: does NOT import your earlier files. All functions are NEW.

# numpy does the matrix math
import numpy as np
# pandas reads and writes the CSV
import pandas as pd
# matplotlib draws the plots
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# PART 1: DATA (make the csv once, then read it)
# ---------------------------------------------------------------

# NEW: builds a fake housing.csv from a hidden rule plus noise
def make_housing_csv(path, n=200, seed=0):
    # random generator with a seed so the data is repeatable
    rng = np.random.default_rng(seed)
    # age of the house in years, whole numbers 1 to 40
    age = rng.integers(1, 41, size=n)
    # bedrooms, whole numbers 1 to 5
    bedrooms = rng.integers(1, 6, size=n)
    # plot area in sq ft, between 500 and 3000
    plot_area = rng.uniform(500, 3000, size=n)
    # noise: the part of price no feature can explain
    noise = rng.normal(0, 3, size=n)
    # hidden rule: price = 20 + 0.03*plot + 5*bedrooms - 0.4*age + noise
    price = 20 + 0.03 * plot_area + 5 * bedrooms - 0.4 * age + noise
    # put everything in one table
    table = pd.DataFrame({"age": age, "bedrooms": bedrooms,
                          "plot_area": plot_area.round(1), "price": price.round(2)})
    # save it to disk without the row-number column
    table.to_csv(path, index=False)


# NEW: reads the csv and returns X (n,3) features and y (n,1) prices
def load_data(path):
    # read the csv into a table
    table = pd.read_csv(path)
    # the three feature columns, as a numpy array of shape (n, 3)
    X = table[["age", "bedrooms", "plot_area"]].to_numpy(dtype=float)
    # the price column, reshaped to a column of shape (n, 1)
    y = table[["price"]].to_numpy(dtype=float)
    # hand both back
    return X, y


# NEW: shuffles rows and holds back a fraction as the test set
def split_train_test(X, y, test_fraction=0.2, seed=0):
    # random generator with a seed so the shuffle is repeatable
    rng = np.random.default_rng(seed)
    # shuffled row numbers 0..n-1
    order = rng.permutation(len(X))
    # how many rows go to the test set
    n_test = int(len(X) * test_fraction)
    # the first n_test shuffled rows are test, the rest are train
    test_rows, train_rows = order[:n_test], order[n_test:]
    # pick the rows out of X and y
    return X[train_rows], X[test_rows], y[train_rows], y[test_rows]


# ---------------------------------------------------------------
# PART 2: STANDARDIZING
# ---------------------------------------------------------------

# NEW: average and s of every column, computed on TRAIN rows only
def fit_standardizer(X_train):
    # average of each column, shape (3,)
    means = X_train.mean(axis=0)
    # s of each column: square the distances, average, root (np.std does exactly this)
    stds = X_train.std(axis=0)
    # save these two arrays: they are needed for test rows and new houses
    return means, stds


# NEW: turns raw columns into z-scores using SAVED means and stds
def apply_standardizer(X, means, stds):
    # (value - average) / s, column by column
    return (X - means) / stds


# ---------------------------------------------------------------
# PART 3: MODEL, LOSS, GRADIENT
# ---------------------------------------------------------------

# NEW: puts a column of ones in front, so b becomes theta[0]
def add_ones(X):
    # one 1 per row
    ones = np.ones((X.shape[0], 1))
    # glue the ones column on the left: shape (n, d+1)
    return np.hstack([ones, X])


# NEW: predictions for every row: X @ theta
def predict(X_design, theta):
    # (n, d+1) @ (d+1, 1) -> (n, 1)
    return X_design @ theta


# NEW: loss = average of squared residuals
def compute_loss(X_design, y, theta):
    # residual = prediction - truth
    residuals = predict(X_design, theta) - y
    # square, then average
    return float(np.mean(residuals ** 2))


# NEW: gradient = (2/n) * Xt (X theta - y), shape (d+1, 1)
def compute_gradient(X_design, y, theta):
    # number of rows
    n = X_design.shape[0]
    # residuals, shape (n, 1)
    residuals = predict(X_design, theta) - y
    # Xt @ residuals: one number per weight, then scale by 2/n
    return (2 / n) * (X_design.T @ residuals)


# ---------------------------------------------------------------
# PART 4: TWO OPTIMIZERS
# ---------------------------------------------------------------

# NEW: gradient descent, returns final theta and the loss at every step
def gradient_descent(X_design, y, lr, n_iters):
    # start with all weights at 0, shape (d+1, 1)
    theta = np.zeros((X_design.shape[1], 1))
    # list that will hold the loss after each step
    loss_history = []
    # repeat the step n_iters times
    for _ in range(n_iters):
        # slope of the loss at the current theta
        grad = compute_gradient(X_design, y, theta)
        # step opposite to the slope
        theta = theta - lr * grad
        # record the loss after this step
        loss_history.append(compute_loss(X_design, y, theta))
    # hand back the answer and the history
    return theta, loss_history


# NEW: closed form, solves (Xt X) theta = Xt y
def fit_closed_form(X_design, y):
    # the (d+1, d+1) matrix
    XtX = X_design.T @ X_design
    # the (d+1, 1) vector
    Xty = X_design.T @ y
    # solve is more stable than inverting XtX
    return np.linalg.solve(XtX, Xty)


# ---------------------------------------------------------------
# PART 5: CONVERT WEIGHTS BACK, PREDICT NEW HOUSES, SCORE
# ---------------------------------------------------------------

# NEW: turns weights learned on z-scores into weights for raw units
def weights_to_raw_units(theta_z, means, stds):
    # weights as a flat list: [b, w1, w2, w3]
    b_z, w_z = theta_z[0, 0], theta_z[1:, 0]
    # raw weight = z weight / s  (lakhs per raw unit)
    w_raw = w_z / stds
    # raw b = z b minus what the shifted averages contributed
    b_raw = b_z - np.sum(w_z * means / stds)
    # hand back as (4, 1) in the same layout as theta
    return np.concatenate([[b_raw], w_raw]).reshape(-1, 1)


# NEW: predicts the price of ONE raw house using the trained z-model
# READ: new_house_prediction_working.md for the entire calculation with example of one house ....
def predict_new_house(raw_house, theta_z, means, stds):
    # shape (1, 3): one row of raw numbers
    row = np.array(raw_house, dtype=float).reshape(1, -1)
    # convert with the SAVED train averages and s
    row_z = apply_standardizer(row, means, stds)
    # add the ones column and predict
    return float(predict(add_ones(row_z), theta_z)[0, 0])


# NEW: RMSE = square root of the loss, in lakhs
def compute_rmse(X_design, y, theta):
    return float(np.sqrt(compute_loss(X_design, y, theta)))


# NEW: R2 = 1 - (sum sq residuals) / (sum sq distances from average y)
def compute_r2(X_design, y, theta):
    # squared errors of the model
    ss_res = np.sum((predict(X_design, theta) - y) ** 2)
    # squared errors of "always guess the average"
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------
# PART 6: PLOTS
# ---------------------------------------------------------------

# NEW: two panels, loss curve and predicted vs actual
def plot_results(loss_history, y_test, y_pred):
    # one window, two side-by-side panels
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # left panel: loss per step, log scale so the flat part is visible
    axes[0].plot(loss_history)
    axes[0].set_yscale("log")
    axes[0].set_title("Loss during gradient descent (standardized)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    # right panel: one dot per test house
    axes[1].scatter(y_test, y_pred, s=15)
    # diagonal: prediction = truth
    lo, hi = float(y_test.min()), float(y_test.max())
    axes[1].plot([lo, hi], [lo, hi], "k--")
    axes[1].set_title("Predicted vs actual (test rows)")
    axes[1].set_xlabel("actual price (lakhs)")
    axes[1].set_ylabel("predicted price (lakhs)")
    # show the window (blocks until closed)
    plt.show()


# ---------------------------------------------------------------
# MAIN: the whole story in order
# ---------------------------------------------------------------
if __name__ == "__main__":
    # 1. make the csv (one time) and read it back
    make_housing_csv("housing.csv")
    X, y = load_data("housing.csv")
    print("X shape", X.shape, "y shape", y.shape)

    # 2. hold back 20 percent of the houses
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print("train rows", len(X_train), "test rows", len(X_test))

    # 3. WITHOUT standardizing: gradient descent on raw columns
    Xr_train = add_ones(X_train)
    with np.errstate(all="ignore"):
        _, raw_hist = gradient_descent(Xr_train, y_train, lr=0.01, n_iters=20)
    print("raw columns, lr 0.01, loss after 20 steps:", raw_hist[-1])

    # 4. standardize using TRAIN rows only, then convert both sets
    means, stds = fit_standardizer(X_train)
    print("train means", means.round(2), "train stds", stds.round(2))
    Xz_train = add_ones(apply_standardizer(X_train, means, stds))
    Xz_test = add_ones(apply_standardizer(X_test, means, stds))

    # 5. gradient descent on standardized columns
    theta_gd, history = gradient_descent(Xz_train, y_train, lr=0.1, n_iters=500)
    print("GD theta (z units)", theta_gd.ravel().round(4))
    print("loss start", round(history[0], 3), "end", round(history[-1], 3))
    # print the loss at a few chosen steps (index 0 is after step 1)
    print("Some Random Loss Values: ", [round(history[i], 3) for i in [0, 9, 49, 99, 499]])

    # 6. closed form on the same standardized columns: must match GD
    theta_cf = fit_closed_form(Xz_train, y_train)
    print("closed form theta (z units)", theta_cf.ravel().round(4))

    # 7. weights in raw units: compare with the hidden rule 20, -0.4, 5, 0.03
    raw_w = weights_to_raw_units(theta_gd, means, stds)
    print("raw units [b, age, bedrooms, plot_area]", raw_w.ravel().round(4))

    # 8. closed form on RAW columns: must equal raw_w above
    theta_raw_cf = fit_closed_form(Xr_train, y_train)
    print("closed form on raw columns        ", theta_raw_cf.ravel().round(4))

    # 9. score on the test houses
    print("test RMSE", round(compute_rmse(Xz_test, y_test, theta_gd), 3),
          "test R2", round(compute_r2(Xz_test, y_test, theta_gd), 4))

    # 10. predict a brand new raw house: age 15, 3 bedrooms, plot 1800
    price = predict_new_house([15, 3, 1800], theta_gd, means, stds)
    print("new house [15, 3, 1800] -> predicted price", round(price, 2), "lakhs")

    # 11. cross-check against scikit-learn (trusted reference)
    from sklearn.linear_model import LinearRegression
    ref = LinearRegression().fit(X_train, y_train.ravel())
    print("sklearn [b, age, bedrooms, plot_area]",
          np.concatenate([[ref.intercept_], ref.coef_]).round(4))

    # 12. plots
    plot_results(history, y_test, predict(Xz_test, theta_gd))