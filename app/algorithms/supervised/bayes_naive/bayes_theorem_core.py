import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

is_spam = np.array([True] * 20 + [False] * 80)
has_free = np.zeros(100, dtype=bool)
has_free[0:15] = True
has_free[20:28] = True

code = is_spam.astype(int) * 2 + has_free.astype(int)
grid = code.reshape(10, 10)

cmap = ListedColormap(["lightgray", "orange", "lightblue", "red"])
plt.imshow(grid, cmap=cmap, vmin=0, vmax=3)
plt.xticks(np.arange(-0.5, 10, 1), [])
plt.yticks(np.arange(-0.5, 10, 1), [])
plt.grid(color="white", linewidth=2)

legend = [Patch(color="lightgray", label="ham, no 'free'"),
          Patch(color="orange", label="ham, has 'free'"),
          Patch(color="lightblue", label="spam, no 'free'"),
          Patch(color="red", label="spam, has 'free'")]
plt.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
plt.title("100 emails: 20 spam, 23 contain 'free'")
plt.show()
"""
What to look for:

Prior, P(spam) = 0.20. The top two rows are spam (red and light blue). That is 20 squares out of 100.
Evidence, P(free) = 0.23. Count the red and orange squares together: 15 + 8 = 23 out of 100.
Likelihood, P(free | spam) = 0.75. Look only at the spam squares (red and light blue). 15 of those 20 are red.
Posterior, P(spam | free) = 15/23. Ignore every square without “free”, so only red and orange remain. The red share of that group is the posterior. 
This is the “shrink the world” idea from A2, and you can see the world shrink.
"""


n_total = 100
n_spam = 20
n_ham = 80
n_free_and_spam = 15
n_free_and_ham = 8

prior_spam = n_spam / n_total
prior_ham = n_ham / n_total

likelihood_spam = n_free_and_spam / n_spam
likelihood_ham = n_free_and_ham / n_ham

evidence = (n_free_and_spam + n_free_and_ham) / n_total

posterior_spam = likelihood_spam * prior_spam / evidence
posterior_ham = likelihood_ham * prior_ham / evidence

print(posterior_spam, posterior_ham, posterior_spam + posterior_ham)


"""
A4. The four names

For classification, rename A as “class” and B as “features” (the measurements we observe):

P(class | features) = P(features | class) * P(class) ÷ P(features)

Name	        Symbol	                Plain meaning	                                                                        In our example
Posterior	    P(class | features)	    What we believe about the class after seeing the features. This is what we want.	    P(spam | free)
Likelihood	    P(features | class)	    How likely these features are if the class were true.	                                P(free | spam) = 0.75
Prior	        P(class)	            What we believed about the class before seeing any features. 
                                        It is just how common the class is.	P(spam) = 0.20
Evidence	    P(features)	            How common these features are overall, across all classes.	                            P(free) = 0.23

Intuition: start with a prior guess (spam is rare, 20%). Then the evidence arrives (the email contains “free”). If “free” is much more common in spam than in ham, 
the likelihood pushes your belief up. The posterior is your updated belief.
"""