import sqlite3
import pandas as pd
import numpy as np

# ── 1. Load data from SQLite ───────────────────────────────────────
# Same database we created in fetch_neos.py
conn = sqlite3.connect("asteroids.db")
df = pd.read_sql("SELECT * FROM neo_raw", conn)
conn.close()

# SQLite stores booleans as 0/1 — convert back to True/False
df['is_hazardous'] = df['is_hazardous'].astype(bool)

print(f"Loaded {len(df)} asteroids from DB")
print(f"Columns: {df.columns.tolist()}")


# ── 2. Feature Engineering ─────────────────────────────────────────
# Raw columns alone aren't always the best inputs for ML
# We create NEW columns that combine raw data into more meaningful signals
# Think of it as giving the model "smarter" inputs to learn from

# Feature 1: Kinetic Energy Proxy
# In physics: KE = 0.5 * mass * velocity²
# We don't know exact mass, but volume ∝ diameter³, so mass ∝ diameter³
# This single number combines SIZE and SPEED into one danger measure
df['kinetic_energy_proxy'] = 0.5 * (df['diameter_km'] ** 3) * (df['velocity_kps'] ** 2)

# Feature 2: Threat Score
# Combines two key risk factors:
# - larger diameter = more dangerous (numerator goes up)
# - larger miss distance = less dangerous (denominator goes up)
# Result: high threat score = big rock passing close to Earth
df['threat_score'] = df['diameter_km'] / (df['miss_dist_km'] / 1e6)

# Feature 3: Is it a Fast Mover?
# Asteroids above 20 km/s are unusually fast — binary flag (0 or 1)
# astype(int) converts True → 1, False → 0 so ML model can use it
df['is_fast'] = (df['velocity_kps'] > 20).astype(int)

# Feature 4: Size Category
# Bins diameter into 3 categories: small / medium / large
# pd.cut() divides a range into buckets — like grouping exam scores into grades
# labels=[0,1,2] gives numeric values the model can use
df['size_category'] = pd.cut(
    df['diameter_km'],
    bins=[0, 0.3, 1.0, float('inf')],  # 0-0.3km=small, 0.3-1km=medium, >1km=large
    labels=[0, 1, 2]                    # 0=small, 1=medium, 2=large
).astype(int)

# Feature 5: Log of miss distance
# miss_dist_km ranges from millions to hundreds of millions
# Taking log compresses this huge range into a smaller, more manageable scale
# Models learn better when features don't have extreme ranges
df['log_miss_dist'] = np.log1p(df['miss_dist_km'])  # log1p = log(1+x), safe when x=0

# Feature 6: Log of diameter
# Same reason — diameter ranges from 0.01 to 10+ km, log scale helps
df['log_diameter'] = np.log1p(df['diameter_km'])


# ── 3. Preview the new features ───────────────────────────────────
# Show original + new columns side by side to verify they look right
feature_cols = [
    'name', 'diameter_km', 'velocity_kps', 'miss_dist_km',  # originals
    'kinetic_energy_proxy', 'threat_score',                  # new combined
    'is_fast', 'size_category',                              # new categorical
    'log_miss_dist', 'log_diameter',                         # new log-scaled
    'is_hazardous'                                           # our label/target
]

print("\nFeature preview (first 5 rows):")
print(df[feature_cols].head())

print(f"\nNew feature stats:")
print(df[['kinetic_energy_proxy','threat_score','is_fast','size_category']].describe().round(4))


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ── 4. Define features (X) and target (y) ─────────────────────────
# X = the inputs the model learns from (all our engineered features)
# y = what we're trying to predict (is this asteroid hazardous?)
# This separation is fundamental to all supervised ML

X = df[[
    'diameter_km',           # raw size
    'velocity_kps',          # raw speed
    'miss_dist_km',          # raw distance
    'magnitude_h',           # brightness (proxy for size)
    'kinetic_energy_proxy',  # size + speed combined
    'threat_score',          # size / distance combined
    'is_fast',               # speed flag
    'size_category',         # size bucket
    'log_miss_dist',         # distance on log scale
    'log_diameter'           # diameter on log scale
]]

# y is our label column — True (hazardous) or False (safe)
y = df['is_hazardous']

print(f"Features shape: {X.shape}")       # should be (34, 10)
print(f"Label distribution:\n{y.value_counts()}")


# ── 5. Train/Test Split ────────────────────────────────────────────
# We split data into two sets:
#   train set (80%) → model learns patterns from this
#   test set  (20%) → we hide this from the model, use it to check real performance
#
# WHY: if we tested on the same data we trained on, the model would look perfect
# but fail on new asteroids — like memorising exam answers vs actually understanding
#
# stratify=y ensures BOTH splits have the same % of hazardous asteroids
# Without this, by bad luck the test set could have zero hazardous ones
# random_state=42 makes the split reproducible — same split every run
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% goes to test set
    stratify=y,          # keep class ratio the same in both splits
    random_state=42      # fixed seed for reproducibility
)

print(f"\nAfter split:")
print(f"  Training set : {X_train.shape[0]} asteroids")
print(f"  Test set     : {X_test.shape[0]} asteroids")
print(f"  Train hazardous: {y_train.sum()} / {len(y_train)}")
print(f"  Test hazardous : {y_test.sum()} / {len(y_test)}")


# ── 6. Handle Class Imbalance with SMOTE ──────────────────────────
# Problem: ~20% hazardous, ~80% safe
# A lazy model that ALWAYS predicts "safe" would be 80% accurate but useless
# We need the model to actually learn what makes an asteroid hazardous
#
# SMOTE (Synthetic Minority Over-sampling TEchnique) fixes this by:
# 1. Looking at existing hazardous asteroid examples
# 2. Creating NEW synthetic hazardous examples by interpolating between real ones
#    e.g. takes two real hazardous asteroids and creates a "blend" of their features
# 3. Result: balanced training set where hazardous = safe in count
#
# IMPORTANT: SMOTE only on TRAINING data — never touch the test set
# The test set must stay real/unmodified to reflect the real world
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE balancing:")
print(f"  Training set size : {len(X_train_balanced)} (was {len(X_train)})")
print(f"  Hazardous count   : {y_train_balanced.sum()}")
print(f"  Safe count        : {(~y_train_balanced).sum()}")
print(f"  Balance ratio     : {y_train_balanced.mean():.1%} hazardous")

import xgboost as xgb


# ── 7. Train XGBoost Classifier ───────────────────────────────────
# XGBoost = Extreme Gradient Boosting
# It builds decision trees one at a time, each tree learning from
# the mistakes (residuals) of the previous tree
# After 300 trees, it combines all their predictions — very accurate
#
# Key parameters explained:
#   n_estimators   : how many trees to build (more = better but slower)
#   max_depth      : how deep each tree can grow (deeper = learns more complex patterns)
#   learning_rate  : how much each tree contributes (lower = more careful learning)
#   subsample      : use 80% of rows per tree (prevents overfitting)
#   colsample_bytree: use 80% of features per tree (prevents overfitting)
#   use_label_encoder: suppress a deprecation warning
#   eval_metric    : what to optimise for — 'logloss' is standard for binary classification
#   random_state   : fixed seed so results are same every run

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',   # use_label_encoder line removed
    random_state=42
)

# .fit() is where actual learning happens
# We pass the BALANCED training data (after SMOTE)
# NOT the original X_train — we want the model to learn from balanced classes
print("Training XGBoost model...")
model.fit(X_train_balanced, y_train_balanced)
print("Training complete!")


# ── 8. Make predictions on the test set ───────────────────────────
# predict()      → gives hard labels: True or False
# predict_proba()→ gives probability scores: [prob_safe, prob_hazardous]
#                  e.g. [0.15, 0.85] means 85% chance hazardous
#
# We use probabilities for AUC scoring (more informative than just True/False)
# We use hard predictions for precision/recall/F1

y_pred = model.predict(X_test)                    # hard predictions: True/False
y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of being hazardous
                                                   # [:, 1] takes the second column
                                                   # (column 0 = prob safe, column 1 = prob hazardous)

print(f"\nPredictions on {len(y_test)} test asteroids:")
print(f"  Actual    : {y_test.values.tolist()}")
print(f"  Predicted : {y_pred.tolist()}")
print(f"  Hazard probabilities: {[round(float(p), 3) for p in y_pred_proba]}")




from sklearn.metrics import (
    classification_report,   # precision, recall, F1 all in one
    roc_auc_score,           # AUC-ROC score
    confusion_matrix,        # actual vs predicted counts
    RocCurveDisplay,         # plots the ROC curve
    PrecisionRecallDisplay   # plots precision-recall curve
)
import matplotlib.pyplot as plt


# ── 9. Evaluation Metrics ──────────────────────────────────────────
# We never judge a model by accuracy alone — especially with imbalanced data
# A model that always says "safe" would be 80% accurate but dangerously wrong
# We use 4 metrics that together give the full picture:
#
# PRECISION: of all asteroids we flagged as hazardous, how many actually were?
#            High precision = fewer false alarms
#
# RECALL:    of all actually hazardous asteroids, how many did we catch?
#            High recall = fewer missed threats (critical for safety!)
#
# F1 SCORE:  harmonic mean of precision and recall — single balanced number
#            F1=1.0 is perfect, F1=0.0 is worst
#
# AUC-ROC:   how well does the model RANK hazardous above safe?
#            0.5 = random guessing, 1.0 = perfect separation
#            Good models target AUC > 0.85

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# AUC-ROC — uses probabilities, not hard predictions
# Higher = model is better at separating hazardous from safe
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC Score: {auc:.3f}")
print("(1.0 = perfect | 0.5 = random guessing)")

# Classification report — uses hard predictions (True/False)
# Shows precision, recall, F1 for each class separately
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Hazardous']))

# Confusion Matrix — shows exactly what was right and wrong
# Format:
#          Predicted Safe  Predicted Hazardous
# Actual Safe     [TN]         [FP]
# Actual Hazardous[FN]         [TP]
#
# TN = True Negative  (correctly said safe)
# FP = False Positive (wrongly flagged as hazardous — false alarm)
# FN = False Negative (missed a real hazardous one — dangerous!)
# TP = True Positive  (correctly caught hazardous)
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"                 Predicted Safe  Predicted Hazardous")
print(f"  Actual Safe         {cm[0][0]}               {cm[0][1]}")
print(f"  Actual Hazardous    {cm[1][0]}               {cm[1][1]}")


# ── 10. Plot ROC + Precision-Recall curves ─────────────────────────
# ROC curve: plots True Positive Rate vs False Positive Rate
#            the more it hugs the top-left corner, the better
# PR curve:  plots Precision vs Recall
#            more useful than ROC when classes are imbalanced (like ours)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ROC Curve
RocCurveDisplay.from_predictions(
    y_test, y_pred_proba,
    name="XGBoost",
    ax=axes[0]
)
axes[0].set_title(f'ROC Curve (AUC = {auc:.3f})')
axes[0].plot([0,1],[0,1],'k--', alpha=0.4, label='Random')  # diagonal = random model

# Precision-Recall Curve
PrecisionRecallDisplay.from_predictions(
    y_test, y_pred_proba,
    name="XGBoost",
    ax=axes[1]
)
axes[1].set_title('Precision-Recall Curve')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=120, bbox_inches='tight')
plt.show()
print("\nSaved model_evaluation.png")
# Save trained model to disk
import pickle
import os

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

size = os.path.getsize('model.pkl')
print(f"Model saved to model.pkl ({size:,} bytes)")


# ── 12. Feature Importance ─────────────────────────────────────────
# XGBoost internally tracks how much each feature contributed to
# improving predictions across all 300 trees
#
# importance_type='gain' = how much a feature IMPROVED accuracy when used
# This is the most meaningful measure — higher gain = more useful feature
# Other options: 'weight' (how often used), 'cover' (how many samples affected)

import matplotlib.pyplot as plt

# get_booster() accesses the underlying XGBoost engine
# get_score() returns a dict: {feature_name: importance_score}
# Features with zero importance are excluded automatically
importance_dict = model.get_booster().get_score(importance_type='gain')

# Sort by importance score descending so most important appears first
importance_sorted = dict(sorted(importance_dict.items(),
                                key=lambda x: x[1],
                                reverse=True))

print("\n" + "="*50)
print("FEATURE IMPORTANCE (by gain)")
print("="*50)

# Print a simple text bar chart so you can see it in terminal too
max_score = max(importance_sorted.values())
for feature, score in importance_sorted.items():
    # Scale bar length relative to the highest scoring feature
    bar_len = int((score / max_score) * 30)
    bar = "█" * bar_len
    print(f"  {feature:<25} {bar} {score:.1f}")


# ── 13. Plot feature importance as a bar chart ─────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

features = list(importance_sorted.keys())
scores   = list(importance_sorted.values())

# Horizontal bar chart — easier to read feature names than vertical
bars = ax.barh(features, scores, color='steelblue', edgecolor='white')

# Add score labels at the end of each bar
for bar, score in zip(bars, scores):
    ax.text(bar.get_width() + max_score*0.01,  # slightly right of bar end
            bar.get_y() + bar.get_height()/2,   # vertically centred on bar
            f'{score:.1f}',                      # the score value
            va='center', fontsize=10)

ax.set_xlabel('Importance Score (Gain)')
ax.set_title('XGBoost Feature Importance — Asteroid Risk Scorer')
ax.invert_yaxis()   # highest importance at top, lowest at bottom
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved feature_importance.png")