import numpy as np                # numpy: library for arrays of numbers and fast maths on them
import matplotlib.pyplot as plt   # matplotlib: library for drawing charts

# ---------------- STEP 1: THE DATA ----------------

# The four boxes an email can fall into. Used later as labels on the x-axis.
cells = ["both", "free only", "money only", "neither"]

# 20 spam emails sorted into the four boxes (same order as `cells`).
# Real data from the B4 table, where the words are independent inside spam.
indep_counts = np.array([6, 9, 2, 3])

# Made-up spam data where the words travel together. Also 20 emails in total.
dep_counts = np.array([8, 2, 0, 10])

# ---------------- STEP 2: THE FUNCTION ----------------

# PURPOSE: take the four box counts for one class and return two lists of
# four probabilities:
#   actual = what the counts really say (box count / total)
#   naive  = what you'd predict if the two words were independent
# A function is a reusable recipe: write it once, run it for both datasets.
def actual_and_naive(counts):
    # Total emails in the class (20). Every probability needs this as the denominator.
    n = counts.sum()

    # ACTUAL: divide each box count by the total. numpy divides all four at once.
    # Example: [6, 9, 2, 3] / 20 -> [0.30, 0.45, 0.10, 0.15]
    actual = counts / n

    # P(free) = emails that contain "free" / total. These are the "both" box
    # (position 0) plus the "free only" box (position 1).
    p_free = (counts[0] + counts[1]) / n

    # P(money) = "both" box (position 0) plus "money only" box (position 2), / total.
    p_money = (counts[0] + counts[2]) / n

    # NAIVE: build each box probability from the single-word probabilities only.
    # (1 - p) means "the word is absent": probabilities of yes and no add to 1.
    p_both       = p_free * p_money              # has free AND has money
    p_free_only  = p_free * (1 - p_money)        # has free, no money
    p_money_only = (1 - p_free) * p_money        # no free, has money
    p_neither    = (1 - p_free) * (1 - p_money)  # no free, no money

    # Pack the four naive numbers into an array in the SAME order as `cells`.
    # If the order differed, the bars would be compared with the wrong boxes.
    naive = np.array([p_both, p_free_only, p_money_only, p_neither])

    # Send both arrays back to whoever called the function.
    return actual, naive

# ---------------- STEP 3: RUN IT AND PRINT ----------------

# Call the function on each dataset. It returns two things, so we catch two names.
actual_indep, naive_indep = actual_and_naive(indep_counts)
actual_dep, naive_dep = actual_and_naive(dep_counts)

# np.round(..., 3) keeps 3 decimals, hiding tiny computer rounding noise
# like 0.44999999999999996. Compare these printouts with the bar table above.
print("Independent case  actual:", np.round(actual_indep, 3))
print("Independent case  naive :", np.round(naive_indep, 3))
print("Dependent case    actual:", np.round(actual_dep, 3))
print("Dependent case    naive :", np.round(naive_dep, 3))

# ---------------- STEP 4: THE PICTURE ----------------

# One figure with 1 row and 2 plots side by side, 11 x 4 inches.
# sharey=True gives both plots the same vertical scale so heights are comparable.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

# Centre positions for the four groups of bars: 0, 1, 2, 3.
positions = np.arange(4)

# Width of one bar. Two bars of 0.35 fit side by side in one unit of space.
bar_width = 0.35

# ----- Left plot (axes[0]): independent case -----
# Blue actual bars, shifted left by half a bar so they don't sit on the orange ones.
axes[0].bar(positions - bar_width / 2, actual_indep, bar_width, label="actual (counted)")
# Orange naive bars, shifted right by half a bar.
axes[0].bar(positions + bar_width / 2, naive_indep, bar_width, label="naive (product)")
axes[0].set_title("Independent given spam")   # plot title
axes[0].set_xticks(positions)                 # put tick marks at the group centres
axes[0].set_xticklabels(cells)                # name the groups using `cells`
axes[0].set_ylabel("probability inside spam") # y-axis label (shared, so only here)
axes[0].legend()                              # key: which colour is actual / naive

# ----- Right plot (axes[1]): dependent case. Same steps, different data. -----
axes[1].bar(positions - bar_width / 2, actual_dep, bar_width, label="actual (counted)")
axes[1].bar(positions + bar_width / 2, naive_dep, bar_width, label="naive (product)")
axes[1].set_title("Words appear together in spam")
axes[1].set_xticks(positions)
axes[1].set_xticklabels(cells)
axes[1].legend()

plt.tight_layout()   # adjust spacing so labels don't overlap
plt.show()           # display the figure (without this, often nothing appears)