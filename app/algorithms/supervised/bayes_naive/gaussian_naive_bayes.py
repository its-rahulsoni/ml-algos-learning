# =============================================================================
# GAUSSIAN NAIVE BAYES BY HAND (PARTS C + D COMBINED): BANANA vs APPLE
# =============================================================================
#
# NOTE: UNDERSTANDING THE OUTPUT GRAPH OF THIS CODE:
#       Read the file - learning_gaussian_naive_bayes.md in this folder.
#
# HOW TO READ THIS FILE (written for a reader coming back after a long time)
#   1. Sections 1 to 8 of this header explain the PROBLEM, the IDEAS, the
#      FORMULAS and the full MATHS SOLUTION in plain words, before any code.
#   2. Then the code follows in the same order as the maths (Sections A to J).
#   3. Every function has a paragraph before it. Every line has a comment.
#   4. Expected printed values are written next to the print lines, so you can
#      run the file and check that you get the same numbers.
#
# -----------------------------------------------------------------------------
# 1. THE PROBLEM STATEMENT
# -----------------------------------------------------------------------------
# We have a basket of fruit. Each fruit is either a BANANA or an APPLE (the
# "class"). For every fruit we measured TWO numbers (the "features"):
#       length    in cm
#       thickness in cm
# (NOTE: we say "thickness", not "width", because the word "width" is also used
#  for how wide a BELL CURVE is. They are unrelated things.)
#
# Training data (10 fruits, we know the class of each):
#       Banana lengths:    16, 18, 20, 22, 24     Banana thickness: 3, 4, 4, 4, 5
#       Apple  lengths:     6,  7,  8,  9, 10     Apple  thickness: 5, 7, 7, 7, 9
#
# A NEW fruit arrives with length = 14 cm and thickness = 5 cm. We do not know
# what it is. QUESTION: is it a banana or an apple?
#
# -----------------------------------------------------------------------------
# 2. THE OBJECTIVE OF THIS CODE
# -----------------------------------------------------------------------------
#   (a) TRAIN: from the 10 fruits, learn a mean and a variance for each
#       (class, feature) pair, and a prior for each class. That is the whole
#       "model". The raw fruit measurements are not needed afterwards.
#   (b) CLASSIFY the new fruit by computing a score per class and picking the
#       biggest. We do this twice: with plain probabilities (easy to follow)
#       and with LOGS (what real software does, and why).
#   (c) Show two practical problems and their fixes:
#         - numbers so tiny that the computer rounds them to 0 (fix: logs)
#         - a variance of exactly 0 that breaks the formula (fix: epsilon,
#           also called "variance smoothing")
#   (d) DRAW the bell curves, the new fruit, and the heights we read off.
#
# -----------------------------------------------------------------------------
# 3. GLOSSARY (every term used below)
# -----------------------------------------------------------------------------
#   class        the label we want to predict: banana or apple.
#   feature      one measurement of a fruit: length, or thickness.
#   prior        P(class): how common the class is BEFORE looking at any
#                measurement. Here 5 bananas out of 10 fruits -> 0.5.
#   likelihood   P(measurement | class): how typical the measurement is IF the
#                fruit belongs to that class. For a number (continuous
#                feature) we use the HEIGHT of that class's bell curve.
#   posterior    P(class | measurements): what we finally want. Score
#                (below) is proportional to it.
#   evidence     P(measurements): the same for every class, so it cannot
#                change the winner. We skip it.
#   mean (μ)     the average value. The CENTRE of the bell.
#   variance (σ²) the average squared distance from the mean. Controls how
#                WIDE the bell is. Standard deviation σ = √variance.
#   density      the height of a bell curve. NOT a probability (probability is
#                the AREA under the curve). Comparing heights of two classes at
#                the same x is still fair, which is all we need.
#   score        prior x likelihood of feature 1 x likelihood of feature 2.
#   underflow    a number so tiny the computer cannot store it and rounds it
#                to exactly 0.
#   log (ln)     the reverse of exp: log(exp(a)) = a. Turns products into sums.
#   epsilon (ε)  a tiny number added to every variance to avoid dividing by 0.
#
# -----------------------------------------------------------------------------
# 4. THE FORMULAS AND THEIR COMPONENTS
# -----------------------------------------------------------------------------
# (4a) BAYES' THEOREM (Part A):
#        P(class | features) = P(features | class) × P(class) / P(features)
#      P(features) is the same for all classes -> drop it when only comparing.
#
# (4b) THE NAIVE ASSUMPTION (Part B): once the class is known, the features
#      do not influence each other, so the joint likelihood splits into a
#      product of single-feature likelihoods:
#        score(class) = P(class) × f(length | class) × f(thickness | class)
#      f(length | banana) means: "the height of the BANANA length bell at this
#      fruit's length". The bar | means "given": use that class's mean and
#      variance. (Why "naive"? Real features often DO influence each other.)
#
# (4c) THE GAUSSIAN FORMULA (Part C): the height of a bell at x.
#        f(x) = 1 / √(2π σ²)  ×  exp( -(x - μ)² / (2σ²) )
#        x       the value we ask about (the new fruit's length, say).
#        μ       the class mean: the centre of the bell.
#        σ²      the class variance: how wide the bell is.
#        π       3.14159...; part of the constant that makes total area = 1.
#        (x-μ)²  squared distance from the centre (makes the bell symmetric).
#        minus   far from the centre -> very negative exponent -> exp near 0
#                -> low height.
#        2σ²     width control: bigger variance -> slower fall -> wider bell.
#        exp(.)  e to the power of (.). At the centre exp(0) = 1 (the peak).
#        1/√(2πσ²) rescales so the area under the bell is exactly 1.
#
# (4d) THE LOG OF THE GAUSSIAN (Part D), derived step by step:
#        start:  f(x) = 1/√(2πσ²) × exp( -(x-μ)² / (2σ²) )
#        1) log of a product is a sum of logs:
#           log f = log( 1/√(2πσ²) ) + log( exp( -(x-μ)² / (2σ²) ) )
#        2) 1/√(2πσ²) is (2πσ²) to the power -1/2, so its log is
#           -1/2 × log(2πσ²)
#        3) log and exp cancel, leaving just the exponent: -(x-μ)² / (2σ²)
#        result:  log f(x) = -1/2 × log(2πσ²)  -  (x-μ)² / (2σ²)
#
# (4e) THE LOG SCORE:
#        log score(class) = log P(class) + log f(length|class)
#                                        + log f(thickness|class)
#
# (4f) VARIANCE SMOOTHING (epsilon):
#        variance used = σ² + ε
#        ε = var_smoothing × (largest variance among all features, measured on
#            the whole training set with all classes together)
#        var_smoothing = 1e-9 (that is 0.000000001), scikit-learn's default.
#
# -----------------------------------------------------------------------------
# 5. WHY WE USE LOGS (the reason, in full)
# -----------------------------------------------------------------------------
# Every extra feature multiplies in another number below 1, so scores shrink
# fast. A computer cannot store numbers below roughly 1e-308. Below that it
# rounds to exactly 0 (UNDERFLOW). If both classes underflow to 0 we cannot
# tell them apart. Logs fix this because:
#   fact 1: log(a × b) = log(a) + log(b)   -> a product of 200 tiny numbers
#           becomes a SUM of 200 ordinary numbers (no underflow).
#   fact 2: log of a number between 0 and 1 is negative; smaller number ->
#           more negative.
#   fact 3: bigger number -> bigger log. So the class with the biggest score
#           is ALSO the class with the biggest log score. The winner does not
#           change. (log score gap = log of the ratio of the scores.)
#
# -----------------------------------------------------------------------------
# 6. WHY WE USE EPSILON / VARIANCE SMOOTHING (the reason, in full)
# -----------------------------------------------------------------------------
# If every training fruit of a class has the exact same value for a feature
# (say thickness always 4.0), its variance is exactly 0. The Gaussian formula
# divides by the variance (1/√(2π×0) and (x-μ)²/(2×0)), so the computer gets
# inf or nan ("not a number") and the whole classification breaks.
# Adding a tiny ε to every variance makes it never exactly 0, so there is no
# division by zero. ε is so small (1e-9 times the biggest variance) that for
# normal features the results do not visibly change. For a zero-variance
# feature it gives a very tall, very narrow spike instead of an error. It is a
# safety net, not a clever model.
#
# -----------------------------------------------------------------------------
# 7. THE MATHS SOLUTION, STEP BY STEP (what the code reproduces)
# -----------------------------------------------------------------------------
# STEP 1: PRIORS = fruits in class / total fruits.
#         Banana 5/10 = 0.5      Apple 5/10 = 0.5
#
# STEP 2: MEAN of each (class, feature) = sum / count.
#         Banana length    (16+18+20+22+24)/5 = 100/5 = 20
#         Banana thickness (3+4+4+4+5)/5      =  20/5 =  4
#         Apple  length    (6+7+8+9+10)/5     =  40/5 =  8
#         Apple  thickness (5+7+7+7+9)/5      =  35/5 =  7
#
# STEP 3: VARIANCE = average of squared distances from the mean.
#         Banana length:    distances -4,-2,0,2,4  squares 16,4,0,4,16
#                           sum 40  ->  variance 40/5 = 8
#         Banana thickness: distances -1,0,0,0,1   squares 1,0,0,0,1
#                           sum 2   ->  variance 2/5 = 0.4
#         Apple length:     distances -2,-1,0,1,2  squares 4,1,0,1,4
#                           sum 10  ->  variance 10/5 = 2
#         Apple thickness:  distances -2,0,0,0,2   squares 4,0,0,0,4
#                           sum 8   ->  variance 8/5 = 1.6
#
#         The TRAINED MODEL (all we keep):
#                       length μ  length σ²  thick μ  thick σ²  prior
#             banana       20         8          4        0.4      0.5
#             apple         8         2          7        1.6      0.5
#
# STEP 4: THE NEW FRUIT (length 14, thickness 5): READ FOUR BELL HEIGHTS.
#   f(length 14 | banana)  [μ=20, σ²=8]
#       coefficient = 1/√(2π×8)  = 1/√50.27 = 1/7.09 ≈ 0.1411
#       exponent    = -(14-20)² / (2×8) = -36/16 = -2.25
#       exp(-2.25) ≈ 0.1054        height = 0.1411 × 0.1054 ≈ 0.01487
#   f(length 14 | apple)   [μ=8, σ²=2]
#       coefficient = 1/√(2π×2)  = 1/√12.57 ≈ 0.2821
#       exponent    = -(14-8)² / (2×2) = -36/4 = -9
#       exp(-9) ≈ 0.0001234        height = 0.2821 × 0.0001234 ≈ 0.0000348
#   f(thickness 5 | banana) [μ=4, σ²=0.4]
#       coefficient = 1/√(2π×0.4) = 1/√2.513 = 1/1.585 ≈ 0.6308
#       exponent    = -(5-4)² / (2×0.4) = -1/0.8 = -1.25
#       exp(-1.25) ≈ 0.2865        height = 0.6308 × 0.2865 ≈ 0.1807
#   f(thickness 5 | apple)  [μ=7, σ²=1.6]
#       coefficient = 1/√(2π×1.6) = 1/√10.05 = 1/3.171 ≈ 0.3154
#       exponent    = -(5-7)² / (2×1.6) = -4/3.2 = -1.25
#       exp(-1.25) ≈ 0.2865        height = 0.3154 × 0.2865 ≈ 0.0904
#
# STEP 5: SCORES = prior × length height × thickness height.
#         Banana: 0.5 × 0.01487 × 0.1807   ≈ 0.001343
#         Apple : 0.5 × 0.0000348 × 0.0904 ≈ 0.00000157
#         BANANA WINS, by about 854 times (length alone gives ~427 times,
#         thickness adds ~2 times, the priors tie).
#         Optional: divide by the total to get probabilities:
#         banana ≈ 0.9988, apple ≈ 0.0012.
#
# STEP 6: THE SAME IN LOGS (formula 4d and 4e).
#         log score = log prior + log length height + log thickness height
#         Banana: -0.693 + (-4.209) + (-1.711) = -6.613
#         Apple : -0.693 + (-10.266) + (-2.404) = -13.363
#         Banana has the larger (less negative) log score -> BANANA WINS.
#         Gap = 6.75, and exp(6.75) ≈ 854 matches Step 5's ratio.
#
# STEP 7: UNDERFLOW DEMO. 0.01 multiplied by itself 200 times = 1e-400, too
#         small to store -> prints 0.0. In logs: 200 × log(0.01) ≈ -921.03,
#         an ordinary number.
#
# STEP 8: ZERO-VARIANCE DEMO. Thickness always 4.0 -> variance 0 -> formula
#         gives nan. Add epsilon -> a finite, very tall, very narrow spike.
#
# -----------------------------------------------------------------------------
# 8. HOW THE CODE IS ORGANISED
# -----------------------------------------------------------------------------
#   Section A  imports
#   Section B  input data (training fruits, new fruit)
#   Section C  functions: estimate_mean_var, gaussian_pdf,
#                         log_gaussian_pdf, compute_epsilon
#   Section D  TRAIN: priors, means, variances      (Steps 1 to 3)
#   Section E  CLASSIFY with plain scores           (Steps 4 and 5)
#   Section F  CLASSIFY with log scores             (Step 6)
#   Section G  underflow demo                       (Step 7)
#   Section H  zero variance and epsilon demo       (Step 8)
#   Section I  draw the bells and the new fruit
# =============================================================================


# -----------------------------------------------------------------------------
# SECTION A: IMPORTS
# -----------------------------------------------------------------------------

# numpy: arrays (lists of numbers that support maths on the whole list at once)
# plus sqrt, exp, log, pi. "as np" is the usual nickname.
# If removed: every line that starts with np. fails with a NameError.
import numpy as np

# matplotlib.pyplot: the plotting library. "as plt" is the usual nickname.
# If removed: we could compute everything but could not draw anything.
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# SECTION B: THE INPUT DATA
# -----------------------------------------------------------------------------

# Training data for the BANANA class: lengths in cm, one number per fruit.
# np.array makes a numpy array so that whole-array maths works (for example
# "banana_length - 20" subtracts 20 from every element at once).
banana_length = np.array([16, 18, 20, 22, 24])

# Training data for the BANANA class: thickness in cm, for the SAME five
# fruits in the same order (the first length and the first thickness belong to
# the same banana).
banana_thick = np.array([3, 4, 4, 4, 5])

# Training data for the APPLE class: lengths in cm.
apple_length = np.array([6, 7, 8, 9, 10])

# Training data for the APPLE class: thickness in cm (same five apples).
apple_thick = np.array([5, 7, 7, 7, 9])

# The NEW fruit we want to classify. We know its measurements, not its class.
new_length = 14   # cm
new_thick = 5     # cm


# -----------------------------------------------------------------------------
# SECTION C: FUNCTIONS
# -----------------------------------------------------------------------------

# estimate_mean_var(values)
#
# PURPOSE: Given the measurements of ONE feature of ONE class (for example the
# five banana lengths), work out the two numbers that completely describe its
# bell curve: the mean (centre) and the variance (width). This is the
# "learning" step of Gaussian Naive Bayes: the model never memorises the
# fruits, it only keeps these two numbers per class per feature. The maths is
# written out by hand (Steps 2 and 3 of the header) rather than calling numpy's
# shortcuts, so each maths step is one visible line of code. We divide by n
# (not n - 1) to match scikit-learn's GaussianNB.
#
# INPUT : values, a numpy array of measurements.
# OUTPUT: a pair (mean, variance).
def estimate_mean_var(values):
    # How many measurements there are (call it n). Needed as the divisor in
    # both the mean and the variance. If removed: nothing to divide by.
    n = len(values)

    # Mean = sum / n. For banana lengths: 100 / 5 = 20.0
    # If removed: the bell would have no centre.
    mean = values.sum() / n

    # Distance of every value from the mean. Subtracting one number from an
    # array subtracts it from every element ("broadcasting").
    # For banana lengths: [-4, -2, 0, 2, 4]
    distances = values - mean

    # Square every distance (** 2 means "to the power of 2"). Makes all
    # distances positive and makes big distances count more.
    # If removed (averaging raw distances): positives and negatives cancel and
    # the answer is always 0, which says nothing about spread.
    squared_distances = distances ** 2

    # Variance = average squared distance = sum / n. For banana lengths:
    # 40 / 5 = 8.0. If removed: the bell would have no width.
    variance = squared_distances.sum() / n

    # Hand both numbers back to the caller as a pair.
    return mean, variance


# gaussian_pdf(x, mean, var)
#
# PURPOSE: The Gaussian formula (4c in the header). For a class with a given
# mean and variance, it returns the HEIGHT of that class's bell curve at x. A
# large height means "x is typical for this class", a tiny height means "x
# would be very unusual for this class". This is the "likelihood" f(feature |
# class) used in the score. It works for one number (14) and also for a whole
# array of numbers (300 points used to draw a smooth curve), because numpy
# applies each operation to every element. "pdf" stands for "probability
# density function".
#
# INPUT : x    a number or an array of numbers.
#         mean the class mean for this feature (μ).
#         var  the class variance for this feature (σ²).
# OUTPUT: the curve height(s), same shape as x.
def gaussian_pdf(x, mean, var):
    # The front constant 1 / √(2π σ²). np.pi is π, np.sqrt is square root.
    # Its job: make the total area under the bell exactly 1.
    # If removed: right shape but wrong height; classes with different
    # variances could not be compared fairly.
    coefficient = 1 / np.sqrt(2 * np.pi * var)

    # The exponent -(x - μ)² / (2σ²). (x - mean) is the distance from the
    # centre, ** 2 squares it, the leading minus makes the curve FALL away
    # from the centre, and dividing by 2 * var sets the width.
    # If the minus sign is removed: the curve explodes away from the centre.
    exponent = -((x - mean) ** 2) / (2 * var)

    # np.exp is e to a power: exp(0) = 1 at the centre, near 0 far away.
    # Multiply by the constant to get the final height and hand it back.
    return coefficient * np.exp(exponent)


# log_gaussian_pdf(x, mean, var)
#
# PURPOSE: The LOG of the Gaussian formula (4d in the header):
#       log f(x) = -1/2 × log(2π σ²)  -  (x - μ)² / (2σ²)
# It returns the log of the bell height at x. We use it because multiplying
# many tiny heights underflows to 0 on a computer (Section 5 of the header),
# while ADDING their logs never does. The exp has disappeared from the formula
# (log and exp cancel), so there is nothing left that can shrink to 0. The
# class with the highest SUM of logs is the same class as the one with the
# highest PRODUCT, so the winner never changes. This is what a real predict
# function uses.
#
# INPUT : x, mean, var  (same meaning as in gaussian_pdf)
# OUTPUT: the log of the bell height(s) at x.
def log_gaussian_pdf(x, mean, var):
    # First part: -1/2 × log(2π σ²). This is the log of the front constant
    # (the power -1/2 comes down in front of the log). np.log is the natural
    # logarithm. If removed: the heights of wide and narrow bells would be
    # compared unfairly (the area-normalising part would be lost).
    first_part = -0.5 * np.log(2 * np.pi * var)

    # Second part: -(x - μ)² / (2σ²). This is just the exponent of the
    # original formula, because log(exp(a)) = a. If removed: the curve would
    # not depend on x at all, so every value would look equally typical.
    second_part = -((x - mean) ** 2) / (2 * var)

    # The log of a product is the sum of the logs, so we simply add the parts.
    return first_part + second_part


# compute_epsilon(all_features, var_smoothing)
#
# PURPOSE: Work out the tiny number ε (epsilon) added to every variance
# (variance smoothing, section 6 of the header). It follows scikit-learn's
# rule: ε = var_smoothing × (largest variance among all features, measured on
# the WHOLE training set with all classes together). Scaling by the largest
# variance keeps ε tiny compared with the real spreads in the data, whatever
# the units are (cm, kg, dollars...). A fixed number like 1e-9 could be huge
# for data measured in tiny units and invisible for data in huge units.
#
# INPUT : all_features  a 2D array, one row per fruit, one column per feature.
#         var_smoothing  the scale factor (default 1e-9).
# OUTPUT: epsilon, one small positive number.
def compute_epsilon(all_features, var_smoothing=1e-9):
    # .var(axis=0) computes the variance of each COLUMN (each feature) over
    # all rows (all fruits, all classes). axis=0 means "down the rows".
    # Result: one variance per feature, e.g. [length variance, thick variance].
    per_feature_variance = all_features.var(axis=0)

    # .max() picks the biggest of those variances (the widest feature).
    largest_variance = per_feature_variance.max()

    # Scale it down by var_smoothing. If removed: no epsilon, and a
    # zero-variance feature would still produce nan.
    return var_smoothing * largest_variance


# -----------------------------------------------------------------------------
# SECTION D: TRAIN (Steps 1 to 3)
# -----------------------------------------------------------------------------

# Prior = number of fruits in the class / total number of fruits.
# The total is the sum of both class sizes. len() counts array elements.
# Without priors the model would ignore how common each class is.
n_total = len(banana_length) + len(apple_length)       # 5 + 5 = 10
banana_prior = len(banana_length) / n_total            # 5 / 10 = 0.5
apple_prior = len(apple_length) / n_total              # 5 / 10 = 0.5

# Mean and variance for each (class, feature) pair. The function returns a
# pair and Python lets us unpack it into two names in one line.
banana_length_mean, banana_length_var = estimate_mean_var(banana_length)
banana_thick_mean, banana_thick_var = estimate_mean_var(banana_thick)
apple_length_mean, apple_length_var = estimate_mean_var(apple_length)
apple_thick_mean, apple_thick_var = estimate_mean_var(apple_thick)

# Print the trained model so you can compare it with the table in Step 3.
print("---- TRAINED MODEL ----")
print("priors            :", banana_prior, apple_prior)                 # 0.5 0.5
print("banana length     :", banana_length_mean, banana_length_var)     # 20.0 8.0
print("banana thickness  :", banana_thick_mean, banana_thick_var)       # 4.0 0.4
print("apple  length     :", apple_length_mean, apple_length_var)       # 8.0 2.0
print("apple  thickness  :", apple_thick_mean, apple_thick_var)         # 7.0 1.6

# Cross-check one value against numpy's built-in shortcut. .var() also divides
# by n by default, so it must give the same answer as our hand-written function.
print("numpy check       :", banana_length.mean(), banana_length.var())  # 20.0 8.0

# Epsilon needs the whole training set as a table: one row per fruit, one
# column per feature. np.concatenate joins the banana and apple arrays into one
# long array (10 values); np.column_stack puts the two long arrays side by side
# as two columns.
all_lengths = np.concatenate([banana_length, apple_length])
all_thicks = np.concatenate([banana_thick, apple_thick])
all_features = np.column_stack([all_lengths, all_thicks])
epsilon = compute_epsilon(all_features)
print("epsilon           :", epsilon)    # about 4.1e-08 (1e-9 x biggest variance 41.0)


# -----------------------------------------------------------------------------
# SECTION E: CLASSIFY THE NEW FRUIT WITH PLAIN SCORES (Steps 4 and 5)
# -----------------------------------------------------------------------------

# Step 4: read four bell heights. Each call is "plug the new fruit's value into
# that class's bell for that feature". First argument: the new fruit's value.
# The other two: the class's mean and variance for that feature from training.
# These four numbers are the likelihoods f(length | banana), f(length | apple),
# f(thickness | banana), f(thickness | apple).
h_length_banana = gaussian_pdf(new_length, banana_length_mean, banana_length_var)
h_length_apple = gaussian_pdf(new_length, apple_length_mean, apple_length_var)
h_thick_banana = gaussian_pdf(new_thick, banana_thick_mean, banana_thick_var)
h_thick_apple = gaussian_pdf(new_thick, apple_thick_mean, apple_thick_var)

print("\n---- PLAIN SCORES (new fruit: length 14, thickness 5) ----")
print("height length|banana  :", round(h_length_banana, 5))    # 0.01487
print("height length|apple   :", round(h_length_apple, 7))     # 3.48e-05
print("height thick |banana  :", round(h_thick_banana, 4))     # 0.1807
print("height thick |apple   :", round(h_thick_apple, 4))      # 0.0904

# Step 5: score = prior x length height x thickness height. Multiplying the
# heights of the two features is the NAIVE assumption in code form: we never
# model how length and thickness relate to each other.
banana_score = banana_prior * h_length_banana * h_thick_banana
apple_score = apple_prior * h_length_apple * h_thick_apple
print("banana score          :", banana_score)               # about 0.001343
print("apple  score          :", apple_score)                # about 1.57e-06

# Optional: turn scores into probabilities by dividing by their total (this is
# the "divide by evidence" step from Part A). Only needed if we want actual
# probabilities; for picking a winner the raw scores are enough.
total_score = banana_score + apple_score
print("P(banana | fruit)     :", round(banana_score / total_score, 4))   # 0.9988
print("P(apple  | fruit)     :", round(apple_score / total_score, 4))    # 0.0012

# The decision: the class with the larger score wins.
print("winner (plain)        :", "banana" if banana_score > apple_score else "apple")


# -----------------------------------------------------------------------------
# SECTION F: CLASSIFY WITH LOG SCORES (Step 6)
# -----------------------------------------------------------------------------

# Same recipe as Section E, but every product becomes a sum of logs
# (log score = log prior + log length height + log thickness height).
# np.log(prior) is the log of the prior. No call to exp appears anywhere, so
# nothing can underflow.
banana_log_score = (np.log(banana_prior)
                    + log_gaussian_pdf(new_length, banana_length_mean, banana_length_var)
                    + log_gaussian_pdf(new_thick, banana_thick_mean, banana_thick_var))
apple_log_score = (np.log(apple_prior)
                   + log_gaussian_pdf(new_length, apple_length_mean, apple_length_var)
                   + log_gaussian_pdf(new_thick, apple_thick_mean, apple_thick_var))

print("\n---- LOG SCORES ----")
print("banana log score      :", round(banana_log_score, 4))    # -6.6126
print("apple  log score      :", round(apple_log_score, 4))     # -13.3626

# Consistency check: the log of the plain score must equal the log score. If
# the two lines print the same number, the log formula is derived correctly.
print("log of plain banana   :", round(np.log(banana_score), 4))   # -6.6126

# The gap between log scores is the log of the ratio of the plain scores, so
# exp(gap) turns it back into "how many times bigger". Expect about 854.
gap = banana_log_score - apple_log_score
print("gap, exp(gap)         :", round(gap, 2), round(np.exp(gap), 1))   # 6.75 854.1

# Decision: bigger (less negative) log score wins, same winner as before.
print("winner (log)          :", "banana" if banana_log_score > apple_log_score else "apple")


# -----------------------------------------------------------------------------
# SECTION G: UNDERFLOW DEMO (Step 7), WHY LOGS ARE NEEDED
# -----------------------------------------------------------------------------

# 0.01 ** 200 means 0.01 multiplied by itself 200 times (like 200 features each
# with likelihood 0.01). The true answer is 1e-400, below what a computer can
# store, so Python rounds it to exactly 0.0. If both classes did this we could
# not tell them apart.
print("\n---- UNDERFLOW DEMO ----")
print("0.01 ** 200           :", 0.01 ** 200)                    # 0.0

# The same quantity in logs: 200 x log(0.01) is an ordinary number, so it is
# stored without trouble.
print("200 * log(0.01)       :", round(200 * np.log(0.01), 2))   # -921.03


# -----------------------------------------------------------------------------
# SECTION H: ZERO VARIANCE AND EPSILON DEMO (Step 8)
# -----------------------------------------------------------------------------

# A made-up feature where every training value is identical (all 4.0). Its
# variance is exactly 0, which breaks the Gaussian formula.
constant_values = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
c_mean, c_var = estimate_mean_var(constant_values)
print("\n---- ZERO VARIANCE DEMO ----")
print("mean, variance        :", c_mean, c_var)                  # 4.0 0.0

# np.errstate silences numpy's RuntimeWarning for division by zero so the
# printout stays clean. The nan results are still produced. Without the
# silencing you would see warnings and the same nan values.
with np.errstate(divide="ignore", invalid="ignore"):
    # 0 / 0 and inf x 0 are undefined, so both heights come out as nan
    # ("not a number"): the classifier cannot use them.
    print("height at 4.0, no eps :", gaussian_pdf(4.0, c_mean, c_var))   # nan
    print("height at 4.1, no eps :", gaussian_pdf(4.1, c_mean, c_var))   # nan

# The fix: variance used = variance + epsilon. It is now a tiny positive
# number, so nothing divides by zero.
smoothed_var = c_var + epsilon
print("height at 4.0, with eps:", round(gaussian_pdf(4.0, c_mean, smoothed_var), 1))   # about 1970
print("height at 4.1, with eps:", gaussian_pdf(4.1, c_mean, smoothed_var))             # 0.0

# The result is a very tall, very narrow spike at 4.0 and essentially zero
# anywhere else: "this class always has thickness 4.0". That is not a clever
# model, only a way to avoid a crash. With the simplified epsilon = 1e-9 used in
# the Part D explanation, the spike height at 4.0 would be about 12,616:
print("spike with eps = 1e-9 :", round(gaussian_pdf(4.0, c_mean, 1e-9), 0))            # about 12616

# For normal features epsilon changes nothing visible. Check: the banana length
# variance with and without epsilon (8.0 versus 8.000000041).
print("banana length var     :", banana_length_var, banana_length_var + epsilon)


# -----------------------------------------------------------------------------
# SECTION I: DRAW THE BELLS AND THE NEW FRUIT
# -----------------------------------------------------------------------------

# One figure with two plots side by side: left for the length feature, right
# for the thickness feature. figsize=(12, 4) is the size in inches.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ---------------- LEFT PLOT: LENGTH ----------------

# 300 evenly spaced lengths from 0 to 30. A smooth curve is really many tiny
# straight segments, so we need many x positions.
x_len = np.linspace(0, 30, 300)

# One call to gaussian_pdf gives the bell height at all 300 positions at once.
axes[0].plot(x_len, gaussian_pdf(x_len, banana_length_mean, banana_length_var), label="banana bell")
axes[0].plot(x_len, gaussian_pdf(x_len, apple_length_mean, apple_length_var), label="apple bell")

# The real training measurements as dots on the floor (height 0). np.zeros(5)
# gives five zeros: one height per dot. They should sit under their own bell.
axes[0].scatter(banana_length, np.zeros(len(banana_length)), label="banana data")
axes[0].scatter(apple_length, np.zeros(len(apple_length)), label="apple data")

# A dashed vertical line at the new fruit's length (14). Where it meets each
# bell is the height we read off in Section E.
axes[0].axvline(new_length, linestyle="--", color="gray", label="new fruit (14 cm)")

# Black dots at the two heights where the dashed line meets the bells. These
# are h_length_banana and h_length_apple. zorder=3 draws them on top.
axes[0].scatter([new_length, new_length], [h_length_banana, h_length_apple],
                color="black", zorder=3, label="heights read off")

axes[0].set_xlabel("length (cm)")               # horizontal axis: the feature
axes[0].set_ylabel("density (curve height)")    # vertical axis: how typical
axes[0].set_title("Length bells")
axes[0].legend()                                # key built from label=... texts

# ---------------- RIGHT PLOT: THICKNESS ----------------

# Thickness values are smaller, so use a smaller range: 0 to 12 cm.
x_thick = np.linspace(0, 12, 300)

# Same drawing steps with the thickness means and variances.
axes[1].plot(x_thick, gaussian_pdf(x_thick, banana_thick_mean, banana_thick_var), label="banana bell")
axes[1].plot(x_thick, gaussian_pdf(x_thick, apple_thick_mean, apple_thick_var), label="apple bell")
axes[1].scatter(banana_thick, np.zeros(len(banana_thick)), label="banana data")
axes[1].scatter(apple_thick, np.zeros(len(apple_thick)), label="apple data")
axes[1].axvline(new_thick, linestyle="--", color="gray", label="new fruit (5 cm)")
axes[1].scatter([new_thick, new_thick], [h_thick_banana, h_thick_apple],
                color="black", zorder=3, label="heights read off")
axes[1].set_xlabel("thickness (cm)")
axes[1].set_ylabel("density (curve height)")
axes[1].set_title("Thickness bells")
axes[1].legend()

# Adjust spacing so labels do not overlap, then open the window. Without
# plt.show() many setups display nothing.
plt.tight_layout()
plt.show()

# WHAT TO LOOK FOR IN THE PICTURE:
#   * Left: the banana length bell is wide and low, the apple one narrow and
#     tall. The dashed line at 14 meets the banana bell clearly above the
#     floor (0.0149) and the apple bell almost ON the floor (0.0000348). That
#     big gap (about 427 times) is why length strongly favours banana.
#   * Right: at thickness 5 the banana bell is about twice as high as the
#     apple bell (0.1807 versus 0.0904): a mild vote for banana.
#   * Multiply the two votes and the priors (0.5 each) and you get the scores
#     in Section E. Banana wins by about 854 times.

# NOTE: UNDERSTANDING THE OUTPUT GRAPH OF THIS CODE:
#       Read the file - learning_gaussian_naive_bayes.md in this folder.