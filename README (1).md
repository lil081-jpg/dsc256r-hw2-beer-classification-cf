# Homework 2 — Beer Classification & Collaborative Filtering

Coursework for **UCSD DSC 256R (Recommender Systems & Web Mining)**, Fall 2026.
Submitted as a single file (`homework2.py`) to the course autograder.

## What this assignment covers

**Part 1 — Classification: predicting high-ABV beers from review data**
(Q1–Q4; balanced logistic regression, evaluated with Balanced Error Rate)

1. Classify beers as high-ABV (`beer/ABV > 7`) using beer-style one-hot features only (`C = 10`).
2. Add all features — one-hot beer style + five sensory ratings
   (`review/aroma`, `review/appearance`, `review/palate`, `review/taste`, `review/overall`)
   + normalized review-text length — and compare against Q1.
3. Hyperparameter sweep: pick `C ∈ {0.001, 0.01, 0.1, 1, 10}` by lowest validation BER.
4. Feature ablation: test BER with category features / review ratings / text length removed.

**Part 2 — Item similarity and collaborative filtering**
(Q5–Q7; predicting `star_rating`)

5. Jaccard similarity between the user sets of two items, plus `mostSimilar(i, N)` —
   the top-N most similar items to item `i` by Jaccard overlap.
6. Baseline predictors: global mean, per-user averages, per-item averages, and an
   item-based collaborative-filtering predictor — the item's average adjusted by the
   user's deviation on other items, weighted by Jaccard similarity between item user sets.
7. Bias-model predictor with shrinkage: baseline plus user bias shrunk as `n / (n + 5)`
   and item bias shrunk as `n / (n + 10)`, clamped to `[1.0, 5.0]`.

## Files

| File | Description |
|---|---|
| `homework2.py` | Complete solution — all feature functions and model code (the autograder submission) |

## Reproducing

The code is written against the course's data loaders and test runner, which are course
materials and not included here. With the beer-review dataset loaded as a list of dicts,
e.g.:

```python
from homework2 import Q1, Q2, Q3, Jaccard, predictRating

mod, validBER, testBER = Q1(catID, dataTrain, dataValid, dataTest)
mod, validBER, testBER = Q2(catID, dataTrain, dataValid, dataTest)
bestMod, bestValid, bestTest = Q3(catID, dataTrain, dataValid, dataTest)
```

requires `scikit-learn` and `numpy`.

## Notes

- `feat()` builds the feature vector modularly — one-hot beer style + sensory ratings +
  normalized review length + bias term — and the Q1/Q2/Q4 calls select feature subsets through it.
- Class labels are `beer/ABV > 7`, and `class_weight='balanced'` handles the minority class.
- BER (Balanced Error Rate) = (false positive rate + false negative rate) / 2.
- The solution was validated against the course's test runner before submission.
