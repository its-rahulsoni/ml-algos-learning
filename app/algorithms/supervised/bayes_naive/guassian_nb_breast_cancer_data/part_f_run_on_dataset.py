# =============================================================================
# PART F: RUN OUR GAUSSIAN NAIVE BAYES ON A REAL DATASET (BREAST CANCER)
# =============================================================================
#
# REQUIRES: my_gaussian_nb.py in the same folder (it holds our class
# MyGaussianNB from Part E), plus numpy, matplotlib and scikit-learn.
#
# -----------------------------------------------------------------------------
# 1. THE PROBLEM
# -----------------------------------------------------------------------------
# scikit-learn ships a dataset of 569 breast tumours. For each tumour, a
# computer measured 30 numbers from an image of the cell nuclei, and doctors
# recorded whether the tumour was MALIGNANT (cancer, 212 cases) or BENIGN
# (not cancer, 357 cases). We want to predict the label from the measurements.
#
# We use just TWO features, because Part G draws the decision boundary and we can
# only draw pictures in 2D:
#     mean radius   average size of the cell nuclei
#     mean texture  how much the grey-scale values vary in the image
#
# WHY NOT TITANIC? Gaussian Naive Bayes assumes each feature is a bell curve
# per class. Titanic's useful columns are mostly categories (sex, class), and
# its continuous ones are awkward (Fare is very skewed, Age has missing
# values). Breast-cancer measurements are continuous and roughly bell-shaped,
# so the model fits the data naturally. (We can revisit Titanic later.)
#
# -----------------------------------------------------------------------------
# 2. THE OBJECTIVE
# -----------------------------------------------------------------------------
#   (a) Load the data and keep two features.
#   (b) SPLIT it into a training part and a test part.
#   (c) FIT our model on the training part only.
#   (d) PREDICT the test part and report ACCURACY.
#   (e) Judge the accuracy: compare with a "dumb" baseline, with the training
#       accuracy, and look at which mistakes are made.
#
# -----------------------------------------------------------------------------
# 3. NEW IDEAS (read once)
# -----------------------------------------------------------------------------
#   TRAIN / TEST SPLIT
#     If we grade the model on the same fruits it learned from, it can look
#     great just by memorising (we saw "train accuracy 1.0" in Part E). So we
#     hide some rows (the TEST set), learn from the rest (the TRAIN set), then
#     grade on the hidden rows. The test score estimates how the model behaves
#     on genuinely new data. This is called GENERALISATION.
#
#   ACCURACY
#     accuracy = (number of correct predictions) / (number of predictions)
#     Example: 123 correct out of 143 test points -> 123 / 143 = 0.86.
#
#   BASELINE
#     The score of a trivial strategy: always answer the most common class.
#     Benign is about 63 percent of the data, so "always benign" scores about
#     0.63 without looking at any measurement. Accuracy only means something
#     relative to this.
#
#   CONFUSION COUNTS
#     Accuracy hides WHICH mistakes are made. We count, for each true class,
#     how many points were predicted as each class. In medicine, calling a
#     malignant tumour "benign" is a much worse mistake than the reverse.
#
# -----------------------------------------------------------------------------
# 4. CODE MAP
# -----------------------------------------------------------------------------
#   Section A  imports
#   Section B  load the data and pick two features
#   Section C  split into train and test
#   Section D  fit the model and look at what it learned
#   Section E  predict and measure accuracy
#   Section F  look at the mistakes
#   Section G  a picture of the two classes
# =============================================================================


# -----------------------------------------------------------------------------
# SECTION A: IMPORTS
# -----------------------------------------------------------------------------

# numpy: arrays and maths. "as np" is the usual nickname.
import numpy as np

# matplotlib: for drawing the scatter plot in Section G.
import matplotlib.pyplot as plt

# load_breast_cancer: a function that returns the dataset described above.
from sklearn.datasets import load_breast_cancer

# train_test_split: a function that shuffles the rows and cuts them into a
# training part and a test part. We use scikit-learn's version of the SPLIT
# only; the model itself is our own code.
from sklearn.model_selection import train_test_split

# Our own class from Part E, saved in gaussian_nb_breast_cancer_data.py. "from FILE import NAME"
# lets us use MyGaussianNB here without copying its code.
# If removed: the model lines below fail with NameError.
from gaussian_nb_breast_cancer_data import MyGaussianNB


# -----------------------------------------------------------------------------
# SECTION B: LOAD THE DATA AND PICK TWO FEATURES
# -----------------------------------------------------------------------------

# Load the dataset. `data` is a bundle holding the measurements (data.data),
# the labels as numbers (data.target), the column names (data.feature_names)
# and the label names (data.target_names).
data = load_breast_cancer()

# Find which COLUMN holds each feature we want. data.feature_names lists the 30
# column names in order. list(...).index("name") returns the position of that
# name in the list. (Radius happens to be column 0 and texture column 1.)
radius_col = list(data.feature_names).index("mean radius")
texture_col = list(data.feature_names).index("mean texture")

# Keep only those two columns of the measurement table.
# data.data has shape (569, 30). [:, [a, b]] means "all rows, only columns a
# and b". Result: X has shape (569, 2), one row per tumour, one column per
# feature, exactly the layout MyGaussianNB expects (Part E).
X = data.data[:, [radius_col, texture_col]]

# Labels as words. data.target is 0 or 1 for each tumour, and data.target_names
# is ["malignant", "benign"]. Indexing the names with the numbers turns
# 0 -> "malignant" and 1 -> "benign" for every row at once. We use words so the
# printed predictions are readable.
y = data.target_names[data.target]

print("---- DATA ----")
print("X shape:", X.shape)                                  # (569, 2)
print("benign count   :", np.sum(y == "benign"))            # 357
print("malignant count:", np.sum(y == "malignant"))         # 212


# -----------------------------------------------------------------------------
# SECTION C: SPLIT INTO TRAIN AND TEST
# -----------------------------------------------------------------------------

# train_test_split shuffles the rows and returns four pieces, in this order:
# training features, test features, training labels, test labels.
#   test_size=0.25    keep 25 percent of the rows (143 tumours) for testing.
#   random_state=42   fixes the shuffle, so you get the same split every run.
#                     (42 is an arbitrary number. Change it and the split, and
#                     therefore the accuracy, will change a little.)
#   stratify=y        keeps the benign/malignant proportions the same in the
#                     train and test parts. Without it, the test set could by
#                     bad luck contain too few malignant tumours to judge.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

print("\n---- SPLIT ----")
print("train rows:", X_train.shape[0], " test rows:", X_test.shape[0])   # 426 143
print("train benign/malignant:", np.sum(y_train == "benign"), np.sum(y_train == "malignant"))  # 267 159
print("test  benign/malignant:", np.sum(y_test == "benign"), np.sum(y_test == "malignant"))    # 90 53


# -----------------------------------------------------------------------------
# SECTION D: FIT THE MODEL AND LOOK AT WHAT IT LEARNED
# -----------------------------------------------------------------------------

# Create the model and train it on the TRAINING part only. The test part must
# never be used here, or the final score would be dishonest.
model = MyGaussianNB().fit(X_train, y_train)

# What the model learned: the "notebook" from Part E. Rows follow classes_
# (sorted alphabetically: benign first, malignant second). Columns follow the
# features (mean radius, mean texture).
print("\n---- LEARNED MODEL ----")
print("classes :", model.classes_)                  # ['benign' 'malignant']
print("priors  :", np.round(model.priors_, 3))      # [0.627 0.373]
print("means   :\n", np.round(model.means_, 2))      # benign [12.17 18.10] / malignant [17.53 21.58]
print("variances:\n", np.round(model.vars_, 2))      # benign [3.24 16.15] / malignant [10.81 14.36]


# -----------------------------------------------------------------------------
# SECTION E: PREDICT AND MEASURE ACCURACY
# -----------------------------------------------------------------------------

# Predict a label for every test row. The model never saw these rows in fit.
y_pred = model.predict(X_test)

# y_pred == y_test compares predictions with the truth, one True/False per row.
# np.mean of True/False counts True as 1 and False as 0, so the mean is the
# fraction of correct predictions = accuracy.
test_accuracy = np.mean(y_pred == y_test)

# For comparison: accuracy on the training rows themselves. If this were far
# above the test accuracy, the model would be memorising instead of learning.
train_accuracy = np.mean(model.predict(X_train) == y_train)

# Baseline: the accuracy of always answering "benign" (the most common class).
baseline = np.mean(y_test == "benign")

print("\n---- ACCURACY ----")
print("test accuracy :", round(test_accuracy, 4))     # 0.8601 (123 of 143)
print("train accuracy:", round(train_accuracy, 4))    # 0.8897
print("baseline      :", round(baseline, 4))          # 0.6294


# -----------------------------------------------------------------------------
# SECTION F: LOOK AT THE MISTAKES (CONFUSION COUNTS)
# -----------------------------------------------------------------------------

# Count each combination of (true class, predicted class). The & means "and":
# np.sum counts the test rows where BOTH conditions are True.
benign_ok = np.sum((y_test == "benign") & (y_pred == "benign"))
benign_wrong = np.sum((y_test == "benign") & (y_pred == "malignant"))
malignant_missed = np.sum((y_test == "malignant") & (y_pred == "benign"))
malignant_ok = np.sum((y_test == "malignant") & (y_pred == "malignant"))

print("\n---- CONFUSION COUNTS ----")
print("true benign    , predicted benign   :", benign_ok)          # 87
print("true benign    , predicted malignant:", benign_wrong)       # 3
print("true malignant , predicted benign   :", malignant_missed)   # 17  (the dangerous kind)
print("true malignant , predicted malignant:", malignant_ok)       # 36
print("total mistakes:", benign_wrong + malignant_missed)          # 20

# The first five test tumours: truth next to prediction, to get a feel for it.
print("\nfirst 5 (truth, prediction):")
for truth, guess in zip(y_test[:5], y_pred[:5]):
    print("  ", truth, "->", guess)


# -----------------------------------------------------------------------------
# SECTION G: A PICTURE OF THE TWO CLASSES (TEST POINTS)
# -----------------------------------------------------------------------------

# Select the test rows of each true class with a boolean mask (as in Part E),
# and draw them as dots: x = mean radius (column 0), y = mean texture (column 1).
benign_pts = X_test[y_test == "benign"]
malignant_pts = X_test[y_test == "malignant"]
plt.scatter(benign_pts[:, 0], benign_pts[:, 1], label="benign (true)")
plt.scatter(malignant_pts[:, 0], malignant_pts[:, 1], label="malignant (true)")

plt.xlabel("mean radius")
plt.ylabel("mean texture")
plt.title("Test tumours by true class")
plt.legend()
plt.show()

# WHAT TO LOOK FOR: benign tumours (small radius) sit mostly to the left and
# malignant ones to the right, but the two clouds OVERLAP in the middle. Points
# in the overlap are the ones any classifier will get wrong using only these two
# features. That overlap is the main reason accuracy stops around 86 percent
# instead of 100 percent.