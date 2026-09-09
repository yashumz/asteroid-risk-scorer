# Asteroid Risk Scorer

An end-to-end ML service that ingests NASA near-Earth object data, engineers
physics-based features, trains an XGBoost classifier to predict "potentially
hazardous asteroid" (PHA) status, and serves predictions through a FastAPI
backend with a React dashboard.

Reported test AUC is 0.953. **This README explains why that number is
misleading, and what the model is actually worth.**

---

## TL;DR

The prediction target is not an observed outcome. It is a threshold rule that
NASA/CNEOS wrote down:

> An asteroid is a PHA if its absolute magnitude (H) is 22.0 or brighter **and**
> its minimum orbit intersection distance (MOID) with Earth is 0.05 au or less.

Absolute magnitude was included directly as an input feature. Half the rule that
*defines* the label was handed to the model as a predictor.

The result is a headline AUC of 0.953 where roughly **0.67** is genuine signal.
Meanwhile precision on the hazardous class is **0.52** - of 27 asteroids flagged,
13 were false alarms.

---

## Setup

- Source: NASA NeoWs API, 6 months of close-approach windows
- Dataset: 828 unique asteroids after de-duplication by ID, 90 hazardous (10.9%)
- Features (10): `diameter_km`, `velocity_kps`, `miss_dist_km`, `magnitude_h`,
  `kinetic_energy_proxy`, `threat_score`, `is_fast`, `size_category`,
  `log_miss_dist`, `log_diameter`
- Model: XGBoost, 300 trees, depth 4, SMOTE on the training split only
- Split: 80/20 stratified (662 train / 166 test, 18 hazardous in test)

---

## Finding 1: three quarters of the model's signal is one variable

XGBoost feature importance by gain:

| feature | gain | share |
|---|---|---|
| `diameter_km` | 25.9 | 43.6% |
| `log_diameter` | 15.8 | 26.6% |
| `magnitude_h` | 3.5 | 5.9% |
| everything else | 14.2 | 23.9% |

The top three are **the same quantity**. NeoWs does not measure asteroid
diameter - it derives `estimated_diameter` from absolute magnitude using an
assumed albedo. `log_diameter` is then a log transform of that.

The audit makes the identity explicit. Single-feature AUC for each:

```
diameter_km    0.9071
magnitude_h    0.9071
log_diameter   0.9071
```

Identical to four decimal places. These are not three features - they are one
feature counted three times, contributing 76% of total gain. Including the other
magnitude-derived features (`kinetic_energy_proxy`, `size_category`,
`threat_score`) brings it to 87%.

## Finding 2: 74% of the dataset is settled by definition

NASA's rule requires H <= 22.0 for PHA status, so any asteroid above that
threshold is non-hazardous **with certainty, by definition**.

| | count | share |
|---|---|---|
| H > 22.0 | 614 | 74.2% |
| of those, hazardous | **0** | - |

Those rows are not predicted. They are looked up.

## Finding 3: dropping one feature fixes nothing

Because the leak is distributed across six correlated features, removing any
single one changes held-out AUC by exactly zero:

```
baseline, all features       0.9002
without diameter_km          0.9002  (+0.0000)
without magnitude_h          0.9002  (+0.0000)
without log_diameter         0.9002  (+0.0000)
without kinetic_energy_proxy 0.9002  (+0.0000)
without size_category        0.9002  (+0.0000)
without threat_score         0.9002  (+0.0000)
```

Drop `magnitude_h` and the model reads `diameter_km` instead. Drop that and it
reads `log_diameter`. The initial hypothesis was that a single engineered feature
(`threat_score`) was responsible; the audit disproved it.

## Finding 4: what the model is actually worth

Removing every magnitude-derived feature and retraining on what remains
(`velocity_kps`, `miss_dist_km`, `is_fast`, `log_miss_dist`):

```
AUC = 0.6712
```

That is the honest number. It is above random (0.50), so there is real signal:
close-approach distance is correlated with, but not identical to, the MOID in
NASA's rule. The gap between 0.6712 and 0.953 is the definition talking.

**The model is not merely rediscovering the rule.** It scores 0.953 rather than
1.0 precisely because `miss_distance` (the distance at one close approach) is not
MOID (the minimum across the entire orbit). Approximating that second condition
is genuine, if narrow, learning. Consistent with this, applying the full NASA
rule with `miss_distance` substituted for MOID reproduces the label on only 89.5%
of rows.

## Finding 5: the model is bad at the job anyway

Setting leakage aside entirely, the classifier performs poorly on the class that
matters:

```
              precision  recall  f1-score  support
Safe               0.97    0.91      0.94      148
Hazardous          0.52    0.78      0.62       18

Confusion matrix
                  Pred Safe   Pred Hazardous
Actual Safe             135               13
Actual Hazardous          4               14
```

Average precision is 0.63. Of 27 asteroids flagged as hazardous, 13 were false
alarms - a precision barely better than a coin flip. AUC-ROC is optimistic on
imbalanced data (10.9% positives); the precision-recall curve is the honest view
and it is far less flattering.

So the model has 87% of its signal handed to it by the label's own definition and
*still* gets half its positive predictions wrong.

---

## The deeper mistake

Feature selection was the symptom. The real error was upstream:

**Predicting a definitional label is not a machine learning problem.**

PHA status is not a property of the universe waiting to be discovered. It is an
administrative category defined by two thresholds on two measured quantities.
Given both quantities, the correct engineering solution is an `if` statement.
Given one of them plus a proxy for the other - which is this setup - a model can
only interpolate the missing threshold, and will report a flattering score for
mostly reciting the definition.

### What would have made this a real problem

- **Predict impact probability** from JPL's Sentry risk table - a computed
  physical quantity, not a category
- **Predict MOID itself** from orbital elements, where the relationship is
  genuinely non-trivial
- **Keep the PHA target but exclude all magnitude-derived inputs**, and report
  against the 0.67 ceiling honestly

---

## What holds up

Checked during the audit and found correct:

- **SMOTE placement.** `train_test_split` runs first; SMOTE is fitted on
  `X_train` only (662 -> 1180 rows). The 166-row test set is untouched real data.
  This is the most common resampling mistake and it is not present here.
- **No fit-before-split leakage in feature engineering.** All transforms use
  hardcoded constants (size bins at 0.3/1.0 km, velocity threshold at 20 km/s),
  so nothing learned from the full dataset crosses the split.
- **De-duplication by asteroid ID**, so objects appearing in multiple
  close-approach windows do not inflate the row count.
- The FastAPI service, Pydantic schemas, and React dashboard work as built. The
  serving layer is sound; the target is what is wrong.

---

## Reproducing the audit

```bash
python leakage_audit.py
```

Reads `asteroids.db`, rebuilds the exact feature set from `train_model.py`, and
runs six checks: the one-sided magnitude rule, full-rule agreement, per-feature
AUC, a depth-2 tree, leave-one-feature-out, and drop-the-whole-family.

Implemented with numpy and pandas only - no scikit-learn - so it runs in
environments where compiled scientific packages are unavailable.

---

## Limitations of this audit

- The audit's decision trees are a simple greedy CART implementation, not
  scikit-learn's. Absolute values differ from the XGBoost run; the comparisons
  between conditions are what matter, not the absolute figures.
- In leave-one-out, removing `velocity_kps` *improved* audit AUC
  (0.9002 -> 0.9403), and depth-4 scored below depth-2. Both are artifacts of the
  greedy tree overfitting at this sample size, not findings about the data.
- `miss_distance` stands in for MOID throughout, because NeoWs does not expose
  MOID.
- An earlier version of this project reported AUC 1.0. That was measured on a
  34-asteroid dataset with a 7-row test set and was not meaningful. The 0.953
  figure above is from the current 828-asteroid run.

---

## Takeaway

The most valuable output here was not the model. It was asking why a strong score
demanded an explanation, and following it back through the feature importances to
a mis-specified target.

Leakage of this kind is easy to ship and hard to spot, because every individual
step looks reasonable: absolute magnitude is a legitimate physical measurement,
diameter is a legitimate feature, and log-scaling skewed variables is standard
practice. Nothing in the code is obviously wrong. The problem is in the
relationship between the features and the definition of the label - which no
amount of cross-validation would have surfaced.
