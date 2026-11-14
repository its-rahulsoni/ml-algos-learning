# Understanding the Length and Thickness Bell-Curve Graphs (Gaussian Naive Bayes, Part D)

> Context: banana vs apple classification with two features (length and thickness).
> A new fruit with length 14 cm and thickness 5 cm is classified by reading heights off each class's bell curve.
> Code that produces this figure: `gaussian_naive_bayes.py`

---

# Part 1: Explaining the two graphs

**Question:**

> Explain this graph (both) to me, especially the vertical line in the middle of each graph and the dark-coloured dots on this vertical line. Your explanation should let me figure out what is being showcased in the graphs, and connect immediately to your explanations so far.

![Length bells (left) and thickness bells (right) with the new fruit marked by a dashed vertical line and black dots](images/length_thickness_bells.png)

**Answer:**

## First, a colour correction

The dots on the dashed lines are **black**, not blue. The legend calls them "heights read off". There are three kinds of dots in each graph, and mixing them up is the main source of confusion:

| Dot | Where it sits | What it is |
|---|---|---|
| Blue dots on the floor (height 0) | along the x-axis | the 5 training bananas |
| Orange dots on the floor (height 0) | along the x-axis | the 5 training apples |
| **Black dots** | on the dashed line, up on the curves | the heights read off for the new fruit |

Only the black dots are new information about the new fruit. The blue and orange floor dots are the training data from Part C. They are there to show that each bell sits where its own data sits.

## What each graph is about

Each graph is about **one feature**. Each has two bells, one per class, built from the means and variances we trained. This is the table from D1 drawn as pictures:

| | Left graph (length) | Right graph (thickness) |
|---|---|---|
| Banana bell (blue line) | centre 20, variance 8 | centre 4, variance 0.4 |
| Apple bell (orange line) | centre 8, variance 2 | centre 7, variance 1.6 |

You can check this against your picture. In the left graph the orange bell is centred at 8 and is narrow and tall (peak about 0.28), and the blue one is centred at 20 and is wide and low (peak about 0.14). In the right graph it flips: the blue banana bell is narrow and tall (peak about 0.63) and the orange apple bell is wider and lower (peak about 0.32).

## The vertical dashed line = the new fruit

The dashed line marks the new fruit's measurement on that graph's x-axis:
- Left graph: the line is at **14**, the new fruit's length.
- Right graph: the line is at **5**, the new fruit's thickness.

Think of it as the question we are asking each bell: **"how typical is this value for you?"**

## The black dots = the answers

Where the dashed line meets a bell, the black dot sits on the curve at that point. The **height of the dot** is the bell's answer. A higher dot means a more typical value for that class.

This is exactly what I called f(length | class) and f(thickness | class) in Part D, and what `gaussian_pdf` returns. Each black dot is one of the four numbers from Step 4 of D3:

**Left graph (length = 14):**
- Black dot on the blue banana curve: height about **0.0149**. You can see it slightly above the floor. This is f(length 14 | banana).
- Black dot on the orange apple curve: height about **0.0000348**. On this scale that is indistinguishable from zero, so it sits on the floor. This is f(length 14 | apple).

**Right graph (thickness = 5):**
- Upper black dot on the blue banana curve: height about **0.18**. This is f(thickness 5 | banana).
- Lower black dot on the orange apple curve: height about **0.09**. This is f(thickness 5 | apple).

## How to read each graph

**In each graph, which curve is higher at the dashed line?** That class gets the vote for that feature. Judge the curves at the dashed line, not by the overall peaks.

**Left graph.** The apple bell has the taller peak overall, but at 14 cm it has already fallen to the floor. The new fruit's length is in the far right tail of the apple bell. How far? The apple mean is 8 and σ = √2 ≈ 1.41, so 14 is (14 − 8) ÷ 1.41 ≈ **4.2σ** away, which is far past the 3σ edge from our 68-95-99.7 discussion. The banana bell is still clearly above the floor at 14. It is (14 − 20) ÷ 2.83 ≈ 2.1σ away, at the edge of the 95% range but plausible. The vote is **banana**, and strongly: 0.01487 ÷ 0.0000348 ≈ **427 times**.

**Right graph.** Both bells are above the floor at 5. The banana dot is about twice as high as the apple dot, 0.1807 ÷ 0.0904 = **2 times**. The vote is **banana**, mildly.

A detail you may find interesting: the banana bell is narrow and sits 1 cm from 5, while the apple bell is wide and sits 2 cm from 5. In σ units both distances are the same, about 1.58σ. That is why both exponents were −1.25 in D3. The 2 times difference comes only from the narrow bell being taller, that is, from the front constant 1/√(2πσ²).

## The two dots at x = 5 on the floor

In the right graph, an orange dot sits on the floor right at the dashed line. That is a **training** apple whose thickness happens to be exactly 5. Banana thickness also includes a 5, hidden underneath it, and the three bananas at 4 are stacked on top of each other. Dots overlap because values repeat. It is a coincidence of the data and not part of the new-fruit reading.

## Connecting back to the score

The two graphs are the two features of the naive product:

score(class) = prior × (left black dot) × (right black dot)

| | prior | length height | thickness height | score |
|---|---|---|---|---|
| banana | 0.5 | 0.01487 | 0.1807 | 0.001343 |
| apple | 0.5 | 0.0000348 | 0.0904 | 0.00000157 |

The combined advantage is 427 × 2 = **854 times**, with the priors tying. We never ask how length and thickness relate to each other. We just read each graph on its own and multiply, which is the naive assumption in picture form.

## Where this sits in the code

Three kinds of lines in Section I of the file produce what you see:

```python
axes[0].axvline(new_length, linestyle="--", color="gray", label="new fruit (14 cm)")
axes[0].scatter([new_length, new_length], [h_length_banana, h_length_apple],
                color="black", zorder=3, label="heights read off")
```

- `axvline(new_length, ...)` draws the dashed line at the new fruit's value.
- `scatter([14, 14], [h_length_banana, h_length_apple], color="black")` puts two black dots at x = 14. Their heights are the variables `h_length_banana` and `h_length_apple`, which Section E computed with `gaussian_pdf(...)`. The right graph does the same with `h_thick_banana` and `h_thick_apple`.

So the picture and the printed numbers are the same quantities. The graph is just the printout drawn.

## Quick check

1. In the left graph, if the new fruit's length were 20 instead of 14, which curve would the dashed line hit highest, and roughly how high would the black dot be on it?
2. In the right graph, what would you expect to see if the new fruit's thickness were 7 instead of 5?

Tell me your answers, or say **"next"** to go on to Part E (building `fit` and `predict` as a proper class).

---

# Part 2: The maths behind the graph explanation (z-scores, ratios, and exponents)

**Question (polished):**

> In your explanation of the left and right graphs, I did not understand the maths behind the following calculations. Please explain each one step by step, so that I can relate them to the underlying concepts:
>
> 1. (14 − 8) ÷ 1.41 ≈ 4.2σ away, which is far past the 3σ edge from the 68-95-99.7 rule
> 2. (14 − 20) ÷ 2.83 ≈ 2.1σ away, at the edge of the 95% range but plausible
> 3. 0.01487 ÷ 0.0000348 ≈ 427 times
> 4. In σ units both distances are the same, about 1.58σ

**Answer:**

All four calculations use one idea, so I'll explain that idea first and then go through each calculation.

## The one idea: measure distance in σ steps

The raw distance between two numbers doesn't tell you much about a bell. Six centimetres away from the centre is huge for a narrow bell and small for a wide bell. So we measure distance using the bell's own ruler, the standard deviation σ (one "step"):

**distance in σ units = (value − mean) ÷ σ**

This number has a name, the **z-score**. It answers: how many steps from the centre is this value? A sign only says which side of the mean (minus means below it). We care about the size, so I'll drop the sign.

**Tiny analogy.** Say a class has an average height of 160 cm with σ = 5 cm. A student of 175 cm is 15 cm above the mean, and 15 ÷ 5 = 3 steps above. Another class has σ = 10 cm. There, the same 15 cm is only 1.5 steps. Same raw distance, but a very different meaning.

We need σ itself, not the variance. In our table we stored variances, so first take square roots:

| Bell | Variance σ² | σ = √variance |
|---|---|---|
| banana length | 8 | √8 = 2.83 (since 2.83 × 2.83 ≈ 8) |
| apple length | 2 | √2 = 1.41 |
| banana thickness | 0.4 | √0.4 = 0.632 |
| apple thickness | 1.6 | √1.6 = 1.265 |

## Calculation 1: apple length, (14 − 8) ÷ 1.41 ≈ 4.2σ

1. The new fruit's length is 14. The apple length bell has centre μ = 8 and σ = 1.41.
2. Distance from the centre: 14 − 8 = 6 cm.
3. Convert to steps: 6 ÷ 1.41 = **4.24**. So the new fruit is about 4.2 steps above the apple centre.

**What does 4.2 steps mean?** Remember the rule from earlier:
- within 1 step of the centre: about 68% of apples
- within 2 steps: about 95%
- within 3 steps: about 99.7%

```
apple length bell (σ = 1.41, steps drawn as a ruler)

  mean-3σ  mean-2σ  mean-1σ  mean  mean+1σ  mean+2σ  mean+3σ        14 cm
   3.77     5.18     6.59     8     9.41     10.82    12.24            |
                                                                  4.2σ away
```

The 3σ mark is at 8 + 3 × 1.41 = 12.24 cm, and 14 cm is well beyond it. The rule says only about 0.3% of apples (3 in 1000) are beyond 3σ on either side. At 4.2σ it is far rarer, roughly 2 in 100,000. So an apple of length 14 would be extremely unusual, which is why the apple curve is on the floor there.

## Calculation 2: banana length, (14 − 20) ÷ 2.83 ≈ 2.1σ

1. The banana length bell has μ = 20 and σ = 2.83.
2. Distance: 14 − 20 = −6 cm. The minus sign just says "below the mean". The size is 6 cm.
3. Steps: 6 ÷ 2.83 = **2.12**.

The 2σ mark below the mean is 20 − 2 × 2.83 = 14.34 cm. The new fruit at 14 is just beyond it. A correction to my earlier wording: I said "at the edge of the 95% range", but 2.12 is slightly outside it. The honest reading is: a bit unusual for a banana (about 5% of bananas are beyond 2σ on either side, combined), but not rare. It is nothing like the apple's 4.2.

**Comparing the two:** the same 6 cm raw distance is 2.1 steps for bananas (wide bell) and 4.2 steps for apples (narrow bell). The narrow apple bell punishes 6 cm twice as hard.

One caution: the 68-95-99.7 rule describes a true Gaussian. Our apple and banana bells are built from only 5 fruits each, so treat these as illustrations of the idea, not exact claims.

## Calculation 3: 0.01487 ÷ 0.0000348 ≈ 427 times

These are the two black dots on the left graph: banana height 0.01487 and apple height 0.0000348.

**Why divide?** To ask "how many times bigger is one than the other?", you divide. For comparison, 6 ÷ 2 = 3 means 6 is three times 2.

1. Banana height ÷ apple height = 0.01487 ÷ 0.0000348.
2. Estimate by hand: 0.0000348 × 400 = 0.01392. That leaves 0.01487 − 0.01392 = 0.00095. And 0.00095 ÷ 0.0000348 ≈ 27. So the total is about 400 + 27 = **427**.

So the banana bell's answer to "how typical is 14 cm?" is about 427 times larger than the apple bell's answer. This ratio is called a **likelihood ratio**, and it is the strength of the length clue's vote for banana. We use a ratio because Naive Bayes only needs to know which class wins and by how much. The final scores multiply such ratios (427 × 2 = 854).

*(The tiny rounding in 0.01487 and 0.0000348 can move this by a point or two, so "about 427" is the right way to say it.)*

## Calculation 4: thickness, both distances are 1.58σ

1. **Banana thickness bell:** μ = 4, σ = 0.632. New fruit thickness is 5. Distance: 5 − 4 = 1 cm. Steps: 1 ÷ 0.632 = **1.58**.
2. **Apple thickness bell:** μ = 7, σ = 1.265. Distance: 5 − 7 = −2 cm (size 2 cm). Steps: 2 ÷ 1.265 = **1.58**.

The raw distances differ (1 cm versus 2 cm), but the apple bell is twice as wide, so in step units both are equal: 1.58 steps from the centre.

**Why this matters for the height.** The bell formula's exponent is exactly half the square of the step count. Here is the short proof:

(x − μ)² ÷ (2σ²) = ((x − μ) ÷ σ)² ÷ 2 = z² ÷ 2

So the exponent in the Gaussian formula is −z² ÷ 2. For all three cases we have seen:
- apple length: z = 4.24, so z² ÷ 2 = 18 ÷ 2 = 9 (the −9 in D3)
- banana length: z = 2.12, so z² ÷ 2 = 4.5 ÷ 2 = 2.25 (the −2.25 in D3)
- both thickness bells: z = 1.58, so z² ÷ 2 = 2.5 ÷ 2 = 1.25 (the −1.25 in D3)

So the exponent is just "how many steps from the centre", squared and halved. That is the link between the 68-95-99.7 ruler and the formula.

**Why do the two thickness heights still differ by 2 times?** Same z gives the same exp(−1.25) = 0.2865 for both bells. What differs is the front constant 1/√(2πσ²):
- banana: 0.6308
- apple: 0.3154

The ratio is 0.6308 ÷ 0.3154 = 2. This comes from the variances: the apple variance is 1.6 ÷ 0.4 = 4 times the banana variance, and √4 = 2. A bell that is twice as wide in σ must be half as tall to keep the area at 1. So the 2 times is the narrower banana bell being taller, not the new fruit being closer to the banana centre in step units.

## Summary table

| Calculation | Distance (cm) | σ (cm) | Steps from centre | Meaning |
|---|---|---|---|---|
| apple length | 6 | 1.41 | 4.24 | far beyond 3σ: extremely unusual |
| banana length | 6 | 2.83 | 2.12 | just beyond 2σ: a bit unusual |
| banana thickness | 1 | 0.632 | 1.58 | well within 2σ: typical |
| apple thickness | 2 | 1.265 | 1.58 | well within 2σ: typical |

## Verify in code

```python
import numpy as np

print((14 - 8) / np.sqrt(2))
print((14 - 20) / np.sqrt(8))
print((5 - 4) / np.sqrt(0.4))
print((5 - 7) / np.sqrt(1.6))
```

- `(14 - 8) / np.sqrt(2)`: value minus mean, divided by the standard deviation (`np.sqrt` turns the variance 2 into σ). Without the square root we would divide by the variance, which is the wrong ruler.
- The four lines are the four calculations above.
- Expect `4.2426...`, `-2.1213...`, `1.5811...`, `-1.5811...`. The minus signs are the "below the mean" direction.

## Quick check

A new fruit has length 10.
1. How many steps (z) is it from the apple length bell? From the banana length bell?
2. Which bell fits it better, and is either of them past 3σ?

Tell me your answers, or say **"next"** to go on to Part E.