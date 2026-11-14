# =============================================================================
# MY GAUSSIAN NAIVE BAYES (MODULE): log_gaussian + class MyGaussianNB
# =============================================================================
#
# WHAT THIS FILE IS
# In Parts C and D (file gaussian_nb_parts_c_d.py) we did everything by hand for
# ONE new fruit with variables named banana_length_mean, apple_thick_var, ...
# That cannot scale: a real dataset has many rows, many features and any number
# of classes. Here we turn the same maths into a class with two methods:
#       fit(X, y)    learn the model from labelled training data
#       predict(X)   give a class for every row of new data
# This is the exact interface scikit-learn uses, so Part H can swap ours for
# sklearn's GaussianNB and compare.
#
# DATA LAYOUT (very important, everything below depends on it)
#   X  a 2D numpy array, one ROW per sample (fruit), one COLUMN per feature.
#      X.shape = (n_samples, n_features). Here (10, 2): column 0 = length,
#      column 1 = thickness.
#   y  a 1D numpy array of labels, one per row of X ("banana" or "apple").
#      y[i] is the class of the fruit in row X[i].
#
# WHAT fit LEARNS AND STORES (all it keeps; the raw data is not needed after)
#   classes_   the distinct class labels, sorted.             shape (n_classes,)
#   priors_    P(class) = fruits in class / total fruits.     shape (n_classes,)
#   means_     mean of every feature, per class.              shape (n_classes, n_features)
#   vars_      variance of every feature, per class, PLUS epsilon.
#                                                             shape (n_classes, n_features)
#   epsilon_   var_smoothing x (largest feature variance of the whole training
#              set). Added to every variance so none is exactly 0.
#   (A trailing underscore is scikit-learn's naming habit for "learned during
#   fit". It lets you tell settings you chose from values the model learned.)
#
# WHAT predict DOES (formulas from Part D, section 4 of that file)
#   For each class c and each row x:
#       log score(c) = log P(c) + SUM over features j of log f(x_j | c)
#       log f(x_j | c) = -1/2 × log(2π σ²) - (x_j - μ)² / (2σ²)
#   (μ and σ² are the stored mean and variance of feature j for class c.)
#   Then pick the class with the biggest log score. Using logs avoids numbers
#   shrinking to 0 (underflow); the winner is the same as with plain scores.
#
# NUMPY TOOLS USED (explained where they first appear)
#   X.shape, np.unique, np.zeros, boolean masks X[y == c], mean/var with
#   axis=0, broadcasting, .sum(axis=1), np.argmax(..., axis=1), enumerate.
#
# CODE MAP
#   Section A  imports
#   Section B  function log_gaussian        (the Gaussian log-likelihood)
#   Section C  class MyGaussianNB
#                - __init__   store the settings
#                - fit        learn priors, means, variances, epsilon
#                - log_scores compute the log score of every class for every row
#                - predict    pick the best class per row
#   (The banana/apple test lives in gaussian_nb_from_scratch.py. This file holds
#   ONLY the reusable code, so other scripts can do:
#       from my_gaussian_nb import MyGaussianNB
#   without re-running that test.)
# =============================================================================


# -----------------------------------------------------------------------------
# SECTION A: IMPORTS
# -----------------------------------------------------------------------------

# numpy: arrays and maths on whole arrays at once. "as np" is the usual nickname.
# If removed: every np. line fails with a NameError.
import numpy as np


# -----------------------------------------------------------------------------
# SECTION B: THE GAUSSIAN LOG-LIKELIHOOD FUNCTION
# -----------------------------------------------------------------------------

# log_gaussian(X, mean, var)
#
# PURPOSE: Computes log f(x | class) for every sample and every feature at the
# same time, using the log-Gaussian formula derived in Part D:
#       log f(x) = -1/2 × log(2π σ²)  -  (x - μ)² / (2σ²)
# X is a table (n_samples rows, n_features columns). mean and var are one
# number per feature (a row of length n_features) for ONE class. numpy
# "broadcasting" automatically lines the row of means up with every row of X,
# so we never write a loop over samples or features. The result is a table of
# the same shape as X: entry [i, j] is the log-likelihood of sample i's feature
# j under this class's bell for feature j. The caller later ADDS across
# features (the naive step) to get one number per sample.
#
# INPUT : X     array (n_samples, n_features): the points to evaluate.
#         mean  array (n_features,): this class's mean for each feature.
#         var   array (n_features,): this class's variance for each feature.
# OUTPUT: array (n_samples, n_features) of log bell heights.
def log_gaussian(X, mean, var):
    # First part: -1/2 × log(2π σ²). It depends only on the variance, so it
    # has shape (n_features,). It is the log of the area-normalising constant.
    # If removed: bells of different widths would be compared unfairly.
    first_part = -0.5 * np.log(2 * np.pi * var)

    # Second part: -(x - μ)² / (2σ²). X has shape (n_samples, n_features) and
    # mean has shape (n_features,). numpy repeats the mean row for every sample
    # row (broadcasting), so (X - mean) is the distance of every value from its
    # own feature's centre. ** 2 squares each distance, dividing by 2 * var
    # applies each feature's own width.
    # If removed: the result would not depend on x, so every value would look
    # equally typical.
    second_part = -((X - mean) ** 2) / (2 * var)

    # Add the two parts. first_part (one row) is broadcast over all samples.
    # Result: array (n_samples, n_features).
    return first_part + second_part


# -----------------------------------------------------------------------------
# SECTION C: THE CLASS
# -----------------------------------------------------------------------------

# class MyGaussianNB
#
# PURPOSE: A class bundles data (what the model has learned) with the functions
# that use it (fit and predict). After you call fit, the object REMEMBERS
# classes_, priors_, means_, vars_ and epsilon_, so predict can use them later
# without being told again. "self" inside a method means "this particular
# model object".
class MyGaussianNB:

    # __init__(self, var_smoothing)
    #
    # PURPOSE: Runs once when you create the model, e.g. MyGaussianNB().
    # It only stores a SETTING (not anything learned). var_smoothing is the
    # scale factor for epsilon: the tiny number added to every variance to
    # avoid division by zero (Part D, section 6). 1e-9 (0.000000001) matches
    # scikit-learn's default.
    def __init__(self, var_smoothing=1e-9):
        # Remember the setting on the object so fit can read it later.
        # If removed: fit would not know how big epsilon should be.
        self.var_smoothing = var_smoothing

    # fit(self, X, y)
    #
    # PURPOSE: The TRAINING step. Goes through the labelled data once and, for
    # every class, computes: the prior (how common the class is), the mean of
    # each feature, and the variance of each feature (plus epsilon). Nothing
    # else is stored: this is the whole "model". It is cheap because we only
    # count, average and compute spreads. It does not iterate or optimise.
    #
    # INPUT : X array (n_samples, n_features), y array (n_samples,) of labels.
    # OUTPUT: self (so you can write MyGaussianNB().fit(X, y) in one line).
    def fit(self, X, y):
        # X.shape is a pair (rows, columns). Unpack it into two names.
        # n_samples is needed for the priors (class size / total).
        # n_features is needed to size the means and variances tables.
        n_samples, n_features = X.shape

        # np.unique returns the distinct labels, sorted, without repeats:
        # ["apple", "banana"]. These are the classes we must learn a bell for.
        # The ORDER here fixes the row order of priors_, means_ and vars_.
        self.classes_ = np.unique(y)

        # How many classes there are, used to size the tables below.
        n_classes = len(self.classes_)

        # Epsilon: X.var(axis=0) computes the variance of each COLUMN over all
        # samples of all classes (axis=0 means "down the rows"). .max() picks
        # the widest feature. Multiply by var_smoothing to get a tiny number.
        # If removed: a zero-variance feature would give nan (Part D, Step 8).
        self.epsilon_ = self.var_smoothing * X.var(axis=0).max()

        # Pre-build empty tables of zeros to fill in. np.zeros(shape) makes an
        # array of the given shape full of 0.0. Priors: one number per class.
        self.priors_ = np.zeros(n_classes)

        # Means: one row per class, one column per feature.
        self.means_ = np.zeros((n_classes, n_features))

        # Variances: same shape as the means table.
        self.vars_ = np.zeros((n_classes, n_features))

        # Loop over the classes. enumerate gives the pair (position, label):
        # (0, "apple"), (1, "banana"). The position i says which row of the
        # tables to fill; the label c says which samples to pick.
        for i, c in enumerate(self.classes_):
            # Boolean mask: (y == c) is an array of True/False, True where the
            # label equals c. Using it as an index keeps ONLY those rows of X.
            # Result: X_c holds just this class's samples (all columns).
            # If removed: we would mix all classes and get one blurry bell.
            X_c = X[y == c]

            # Prior = fruits in this class / total fruits. X_c.shape[0] is the
            # number of rows of X_c, i.e. the class size.
            self.priors_[i] = X_c.shape[0] / n_samples

            # Mean of each feature for this class: axis=0 averages down the
            # rows, giving one mean per column (shape (n_features,)).
            self.means_[i] = X_c.mean(axis=0)

            # Variance of each feature for this class (average squared distance
            # from the class mean, dividing by n like scikit-learn), plus the
            # tiny epsilon so it can never be exactly 0.
            self.vars_[i] = X_c.var(axis=0) + self.epsilon_

        # Return the model itself so calls can be chained. If removed: fit
        # returns None and MyGaussianNB().fit(X, y) could not be used directly.
        return self

    # log_scores(self, X)
    #
    # PURPOSE: For every row of X and every class, compute the LOG SCORE:
    #       log prior + SUM over features of log f(feature | class)
    # Result is a table with one row per sample and one column per class.
    # The SUM over features is the naive assumption in code: features are
    # combined by simply adding their logs (multiplying their likelihoods),
    # never modelling how they relate to each other. We keep this as its own
    # method (instead of hiding it inside predict) because later parts reuse
    # it, for example to draw the decision boundary or compute probabilities.
    #
    # INPUT : X array (n_samples, n_features).
    # OUTPUT: array (n_samples, n_classes) of log scores.
    def log_scores(self, X):
        # Empty table to fill: one row per sample, one column per class.
        scores = np.zeros((X.shape[0], len(self.classes_)))

        # Loop over classes by position i, the same order as classes_ and the
        # rows of priors_, means_ and vars_.
        for i in range(len(self.classes_)):
            # Log of this class's prior. np.log is the natural logarithm.
            # If removed: the model would ignore how common each class is.
            log_prior = np.log(self.priors_[i])

            # log_gaussian gives a table (n_samples, n_features) of log heights
            # for this class's bells. .sum(axis=1) adds ACROSS the columns
            # (the features), giving one number per sample. Adding logs is
            # multiplying the likelihoods: the naive step.
            # If removed (.sum): we would still have one number per feature
            # instead of one per sample.
            log_like = log_gaussian(X, self.means_[i], self.vars_[i]).sum(axis=1)

            # Fill column i of the scores table: log prior + log likelihood.
            # [:, i] means "all rows, column i". The single log_prior number is
            # added to every row.
            scores[:, i] = log_prior + log_like

        # Hand back the full table.
        return scores

    # predict(self, X)
    #
    # PURPOSE: Decide the class of every row of X. Computes the log score of
    # every class, then for each row picks the class with the LARGEST log score.
    # Largest log score is the same as largest plain score (logs preserve
    # order), so this is exactly "the class with the biggest score wins" from
    # Parts B and D. It returns the actual labels ("banana"/"apple"), not just
    # numbers.
    #
    # INPUT : X array (n_samples, n_features).
    # OUTPUT: array (n_samples,) of predicted labels.
    def predict(self, X):
        # np.argmax(table, axis=1) looks along each ROW and returns the COLUMN
        # POSITION of its largest value. Column positions are class positions,
        # so best[k] says which class wins for sample k (0 or 1 here).
        # If axis=1 were removed: argmax would flatten the table and return one
        # single position for the whole table, which is meaningless.
        best = np.argmax(self.log_scores(X), axis=1)

        # Turn positions back into labels. classes_[best] picks classes_[0] or
        # classes_[1] for every sample, e.g. [1, 0, 1] -> ["banana", "apple", "banana"].
        return self.classes_[best]