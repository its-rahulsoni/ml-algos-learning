# =============================================================================
# GAUSSIAN BELL CURVES: ONE PER CLASS (BANANA vs APPLE)
# =============================================================================
#
# -----------------------------------------------------------------------------
# 1. THE PROBLEM STATEMENT (in plain words)
# -----------------------------------------------------------------------------
# We have a small basket of fruit. For every fruit we measured ONE number:
# its length in cm. We know which fruit is a banana and which is an apple.
#
#     Banana lengths (cm): 16, 18, 20, 22, 24
#     Apple  lengths (cm):  6,  7,  8,  9, 10
#
# Later (Part D) a NEW fruit of unknown type will arrive, e.g. length = 18 cm,
# and we will have to decide: banana or apple?
#
# To decide, we need to answer, for EACH class:
#     "How typical is a length of 18 cm for this class?"
#
# A single number (like 0.1098) cannot be read off a table, because lengths are
# continuous (18.0, 18.01, 18.0001 ...). So we describe each class with a smooth
# BELL CURVE (a Gaussian / Normal distribution). The height of the curve at a
# length says how typical that length is for the class.
#
# -----------------------------------------------------------------------------
# 2. THE OBJECTIVE OF THIS CODE
# -----------------------------------------------------------------------------
#   (a) From the raw measurements, ESTIMATE two numbers for each class:
#         the mean (centre of the bell)  and  the variance (width of the bell).
#   (b) Turn those two numbers into a full bell curve using the Gaussian formula.
#   (c) Check the formula in code against our hand calculations.
#   (d) DRAW both bell curves with the real data points underneath, so we can
#       see that each curve sits where its own class's data sits.
#
# This is exactly the "fit" step of Gaussian Naive Bayes (estimating mean and
# variance per class), plus the likelihood function used by "predict".
#
# -----------------------------------------------------------------------------
# 3. THE INITIAL VALUES (the input)
# -----------------------------------------------------------------------------
#   banana_len = [16, 18, 20, 22, 24]     n = 5 bananas
#   apple_len  = [ 6,  7,  8,  9, 10]     n = 5 apples
#
# -----------------------------------------------------------------------------
# 4. THE MATHEMATICAL SOLUTION, STEP BY STEP (what the code reproduces)
# -----------------------------------------------------------------------------
# STEP 1: MEAN  (the centre).  μ = (x1 + x2 + ... + xn) / n
#   Banana: (16 + 18 + 20 + 22 + 24) / 5 = 100 / 5 = 20
#   Apple : ( 6 +  7 +  8 +  9 + 10) / 5 =  40 / 5 =  8
#
# STEP 2: DISTANCE of each value from its mean (x - μ)
#   Banana: 16-20, 18-20, 20-20, 22-20, 24-20 = -4, -2, 0, 2, 4
#   Apple :  6-8,   7-8,   8-8,   9-8,  10-8  = -2, -1, 0, 1, 2
#
# STEP 3: SQUARE each distance (so negatives do not cancel positives)
#   Banana: 16, 4, 0, 4, 16        Apple: 4, 1, 0, 1, 4
#
# STEP 4: VARIANCE = average of the squared distances.  σ² = Σ (x - μ)² / n
#   Banana: (16 + 4 + 0 + 4 + 16) / 5 = 40 / 5 = 8
#   Apple : ( 4 + 1 + 0 + 1 +  4) / 5 = 10 / 5 = 2
#   (Standard deviation σ = √variance: banana √8 ≈ 2.83, apple √2 ≈ 1.41)
#
# STEP 5: PLUG INTO THE GAUSSIAN FORMULA to get the curve height at any x.
#   Banana at x = 18 (μ = 20, σ² = 8):
#       coefficient = 1 / √(2 × 3.14159 × 8) = 1 / √50.27 = 1 / 7.09 ≈ 0.1411
#       exponent    = -(18 - 20)² / (2 × 8) = -4 / 16 = -0.25
#       exp(-0.25)  ≈ 0.7788
#       height      = 0.1411 × 0.7788 ≈ 0.1098
#   Apple at x = 10 (μ = 8, σ² = 2):
#       coefficient = 1 / √(2 × 3.14159 × 2) = 1 / √12.566 = 1 / 3.545 ≈ 0.2821
#       exponent    = -(10 - 8)² / (2 × 2) = -4 / 4 = -1
#       exp(-1)     ≈ 0.3679
#       height      = 0.2821 × 0.3679 ≈ 0.1038
#
# -----------------------------------------------------------------------------
# 5. THE GAUSSIAN FORMULA AND ITS COMPONENTS
# -----------------------------------------------------------------------------
#   f(x) = 1 / √(2π σ²)  ×  exp( -(x - μ)² / (2σ²) )
#
#   x      the value we ask about (a fruit's length in cm).
#   μ      the class mean: where the middle of the bell sits.
#   σ²     the class variance: how wide the bell is.
#   π      3.14159... (circle constant). It appears because the total area under
#          the bell has to come out as exactly 1.
#   exp(.) "e to the power of .", where e ≈ 2.718.
#
#   (x - μ)²   squared distance from the centre. Squaring makes the curve
#              symmetric (5 cm too short is as unusual as 5 cm too long).
#   the minus  far from the centre -> big squared distance -> very negative
#   sign       exponent -> exp(very negative) is close to 0 -> low height.
#   2σ²        the "width control": larger variance shrinks the fraction, so the
#              curve falls off more slowly -> wider bell.
#   exp(.)     turns that falling number into the smooth bell shape. At the
#              centre the exponent is 0, and exp(0) = 1 (the top of the bell).
#   1/√(2πσ²)  a rescaling constant that makes the total area under the curve
#              equal to 1. A wide bell must be lower to keep the same area.
#
#   IMPORTANT: f(x) is a DENSITY (a height), not a probability. Probability is
#   the AREA under the curve over a range. For classification we only COMPARE
#   heights of different classes at the same x, which is valid.
#
# -----------------------------------------------------------------------------
# 6. HOW THE CODE IS ORGANISED
# -----------------------------------------------------------------------------
#   Section A  imports
#   Section B  the input data (the initial values)
#   Section C  function 1: estimate_mean_var   (Steps 1 to 4 above)
#   Section D  function 2: gaussian_pdf        (Step 5 above)
#   Section E  run the functions and verify against the hand calculations
#   Section F  draw the curves
# =============================================================================


# -----------------------------------------------------------------------------
# SECTION A: IMPORTS
# -----------------------------------------------------------------------------

# numpy ("numerical python") gives us arrays (lists of numbers that support
# maths on the whole list at once) and functions like sqrt and exp.
# "as np" is a nickname, so we can write np.sqrt instead of numpy.sqrt.
# If removed: every line that starts with np. would fail with a NameError.
import numpy as np

# matplotlib.pyplot is the plotting library. "as plt" is the usual nickname.
# If removed: we could compute everything but could not draw anything.
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# SECTION B: THE INPUT DATA (initial values)
# -----------------------------------------------------------------------------

# np.array(...) turns a normal Python list into a numpy array. We want an array
# (not a plain list) because arrays allow whole-array maths such as
# "banana_len - 20", which subtracts 20 from every element in one go.
# These five numbers are the training data for the banana class.
banana_len = np.array([16, 18, 20, 22, 24])

# The same for the apple class. Notice apples are shorter AND their values are
# closer together; we expect a narrower bell centred lower down.
apple_len = np.array([6, 7, 8, 9, 10])


# -----------------------------------------------------------------------------
# SECTION C: FUNCTION 1, ESTIMATE MEAN AND VARIANCE
# -----------------------------------------------------------------------------

# estimate_mean_var(values)
#
# PURPOSE: Given the measurements of ONE class, work out the two numbers that
# completely describe its bell curve: the mean (where the middle is) and the
# variance (how spread out the values are). This is the "learning" step of
# Gaussian Naive Bayes: the model does not memorise the fruit, it only keeps
# these two numbers per class per feature. We write the arithmetic out by hand
# (instead of just calling numpy's shortcuts) so that you can see each maths
# step from the solution above turn into one line of code.
#
# INPUT : values, a numpy array of measurements for one class.
# OUTPUT: a pair (mean, variance), two plain numbers.
def estimate_mean_var(values):
    # len(values) is how many measurements there are. We call it n.
    # It is needed as the divisor in both the mean and the variance.
    # If removed: we would not know what to divide by.
    n = len(values)

    # STEP 1 (mean): values.sum() adds all measurements together, then we divide
    # by n. For bananas: 100 / 5 = 20.0
    # If removed: we would have no centre for the bell.
    mean = values.sum() / n

    # STEP 2 (distances): subtracting a single number from an array subtracts it
    # from EVERY element (numpy calls this "broadcasting").
    # For bananas this gives [-4, -2, 0, 2, 4].
    distances = values - mean

    # STEP 3 (squares): ** 2 means "to the power of 2", applied to every element.
    # Squaring makes every distance positive, and makes big distances count more.
    # For bananas this gives [16, 4, 0, 4, 16].
    # If removed (and we averaged raw distances): positives and negatives cancel
    # and the answer would always be 0, which says nothing about spread.
    squared_distances = distances ** 2

    # STEP 4 (variance): the average squared distance. Sum = 40, n = 5 -> 8.0
    # We divide by n (not n - 1) to match scikit-learn's GaussianNB.
    # If removed: we would have no width for the bell.
    variance = squared_distances.sum() / n

    # Send both numbers back to the caller as a pair.
    # If removed: the function would give back nothing (None).
    return mean, variance


# -----------------------------------------------------------------------------
# SECTION D: FUNCTION 2, THE GAUSSIAN FORMULA
# -----------------------------------------------------------------------------

# gaussian_pdf(x, mean, var)
#
# PURPOSE: Implements the Gaussian formula
#       f(x) = 1 / √(2π σ²) × exp( -(x - μ)² / (2σ²) )
# For a class with a given mean and variance, it returns the HEIGHT of that
# class's bell curve at the value x. A big number means "x is typical for this
# class", a tiny number means "x would be very unusual for this class". The
# function works whether x is a single number (18) or a whole array of numbers
# (300 points to draw a curve), because numpy applies each operation to every
# element. In Part D, this same function will give the likelihood of a new
# fruit's length under each class. The abbreviation PDF stands for "probability
# density function".
#
# INPUT : x    a number or array of numbers where we want the height
#         mean the class mean (μ)
#         var  the class variance (σ²)
# OUTPUT: the curve height(s) at x, same shape as x.
def gaussian_pdf(x, mean, var):
    # The front constant 1 / √(2π σ²). np.pi is π, np.sqrt is the square root.
    # Its job is to make the total area under the bell equal to 1.
    # If removed: the bell shape would be right but its height would be wrong,
    # and comparing classes with different variances would be unfair.
    coefficient = 1 / np.sqrt(2 * np.pi * var)

    # The exponent -(x - μ)² / (2σ²).
    # (x - mean) is the distance from the centre, ** 2 squares it, and the
    # leading minus sign makes the exponent negative, so the curve falls as we
    # move away from the centre. Dividing by 2 * var sets the width.
    # If the minus sign is removed: the curve would explode upward away from the
    # centre instead of shrinking.
    exponent = -((x - mean) ** 2) / (2 * var)

    # np.exp(exponent) is e raised to the exponent: exp(0) = 1 at the centre,
    # and values close to 0 far away. Multiplying by the coefficient gives the
    # final height, which we return.
    return coefficient * np.exp(exponent)


# -----------------------------------------------------------------------------
# SECTION E: RUN THE FUNCTIONS AND VERIFY AGAINST THE HAND CALCULATIONS
# -----------------------------------------------------------------------------

# Calling the function once per class. The function returns a pair, and Python
# lets us unpack the pair into two variable names in one line.
# Expected (from Steps 1 to 4): banana -> 20.0 and 8.0, apple -> 8.0 and 2.0.
banana_mean, banana_var = estimate_mean_var(banana_len)
apple_mean, apple_var = estimate_mean_var(apple_len)

# print shows values on screen. The goal is to compare them with the hand
# calculation. If they differ, there is a bug somewhere.
print("Banana mean and variance:", banana_mean, banana_var)   # expect 20.0 8.0
print("Apple  mean and variance:", apple_mean, apple_var)     # expect 8.0 2.0

# Cross-check against numpy's built-in shortcuts. .mean() and .var() do the same
# maths as our function. np.var divides by n by default, as we did. If these
# two lines print the same numbers, our hand-written function is correct.
print("numpy check, banana:", banana_len.mean(), banana_len.var())   # 20.0 8.0
print("numpy check, apple :", apple_len.mean(), apple_len.var())     # 8.0 2.0

# Standard deviation σ = √variance. It is in cm (the same unit as the data),
# so it is easier to picture than variance. Expect about 2.83 and 1.41.
print("Banana std dev:", round(np.sqrt(banana_var), 2))   # 2.83
print("Apple  std dev:", round(np.sqrt(apple_var), 2))    # 1.41

# Check the Gaussian function against Step 5 of the hand solution.
# round(value, 4) keeps 4 decimals. Without it, you might see a long tail of
# digits such as 0.10977..., which is harder to compare by eye.
print("Banana curve height at 18 cm:", round(gaussian_pdf(18, banana_mean, banana_var), 4))  # 0.1098
print("Apple  curve height at 10 cm:", round(gaussian_pdf(10, apple_mean, apple_var), 4))    # 0.1038

# The peak of each bell is at its own mean (exponent = 0, so exp(0) = 1, leaving
# only the coefficient). Expect banana about 0.141 and apple about 0.2821.
# Notice: the narrow apple bell has the TALLER peak, since area must stay 1.
print("Banana peak height:", round(gaussian_pdf(banana_mean, banana_mean, banana_var), 4))  # 0.141
print("Apple  peak height:", round(gaussian_pdf(apple_mean, apple_mean, apple_var), 4))      # 0.2821

# How typical is a 14 cm fruit for each class? (Preview of Part D.)
# Banana is only about 2.1 standard deviations away; apple is about 4.2 away.
# So the banana height should be much larger than the apple height.
print("Height at 14 cm, banana:", gaussian_pdf(14, banana_mean, banana_var))   # about 0.0149 (exp(-2.25) x 0.1411)
print("Height at 14 cm, apple :", gaussian_pdf(14, apple_mean, apple_var))     # about 0.0000348 (exp(-9) x 0.2821), about 430 times smaller


# -----------------------------------------------------------------------------
# SECTION F: DRAW THE CURVES
# -----------------------------------------------------------------------------

# np.linspace(start, stop, count) makes `count` evenly spaced numbers from start
# to stop. Here it gives 300 lengths from 0 to 30 cm. A smooth-looking curve is
# really a lot of tiny straight segments, so we need many x positions.
# If count were small (say 5): the curve would look like a jagged polygon.
x_values = np.linspace(0, 30, 300)

# Because gaussian_pdf works on whole arrays, ONE call gives the curve height at
# all 300 x positions. plt.plot(x, y) then joins the (x, y) points with a line.
# label="..." is the text that will appear in the legend.
plt.plot(x_values, gaussian_pdf(x_values, banana_mean, banana_var), label="banana curve")
plt.plot(x_values, gaussian_pdf(x_values, apple_mean, apple_var), label="apple curve")

# plt.scatter(xs, ys) draws a dot at each (x, y). We put each real measurement
# on the x-axis at height 0 (np.zeros(5) makes [0, 0, 0, 0, 0] for the heights),
# so you can see the data sitting under the middle of its own bell.
# If removed: you would see the curves but not the real data they came from.
plt.scatter(banana_len, np.zeros(len(banana_len)), label="banana data")
plt.scatter(apple_len, np.zeros(len(apple_len)), label="apple data")

# Axis labels and title tell the reader what the picture means.
plt.xlabel("length (cm)")                  # horizontal axis: the feature
plt.ylabel("density (height of curve)")    # vertical axis: how typical a length is
plt.title("One bell curve per class")

# Builds the legend from the label=... texts given above.
# If removed: you could not tell which colour is which.
plt.legend()

# Opens the window with the finished figure. Without this line, many setups
# show nothing at all.
plt.show()

# WHAT TO LOOK FOR IN THE PICTURE:
#   * The banana bell is centred at 20 and is wide and low (peak about 0.141).
#   * The apple bell is centred at 8 and is narrow and tall (peak about 0.282).
#   * Each set of dots sits under the middle of its own bell.
#   * The bells barely overlap. A length near 14 sits where the apple curve is
#     almost flat on the floor and the banana curve is still clearly above it.
#     That comparison of heights is what Naive Bayes uses to classify.