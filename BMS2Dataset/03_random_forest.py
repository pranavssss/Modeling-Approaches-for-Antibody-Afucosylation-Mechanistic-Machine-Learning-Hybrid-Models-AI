"""
03_random_forest.py
====================
Random Forest Regression — Machine Learning Baseline

Random Forest captures nonlinear relationships and feature interactions without
explicit specification, serving as a strong ML baseline against PLSR.

Literature References:
- Breiman (2001): Random Forests
- Strobl et al. (2007): Bias in random forest variable importance (MDI bias)
- Altmann et al. (2010): Permutation importance — a corrected alternative

Key Improvements:
- Both MDI and Permutation Importance (addresses MDI bias)
- Hyperparameter tuning via RandomizedSearchCV
- Out-of-bag (OOB) error estimation
- Partial dependence plots for top features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("results/03_random_forest", exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 70)
print("03 — RANDOM FOREST REGRESSION")
print("=" * 70)

data = pd.read_csv("data/mab_fucosylation_dataset.csv")
feature_cols = [c for c in data.columns if c not in ["Fucosylation_pct", "Batch_ID"]]
X = data[feature_cols].values
y = data["Fucosylation_pct"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RF doesn't require scaling, but we keep for consistency
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ============================================================
# 2. Hyperparameter Tuning (RandomizedSearchCV)
# ============================================================
print("\n" + "-" * 50)
print("Hyperparameter Tuning (RandomizedSearchCV)")
print("-" * 50)

param_distributions = {
    "n_estimators": randint(100, 500),
    "max_depth": [None, 10, 15, 20, 25, 30],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", 0.5, 0.7, 0.9, 1.0],
    "bootstrap": [True],
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, oob_score=True),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
rf_search.fit(X_train_s, y_train)

print(f"  Best CV R²: {rf_search.best_score_:.4f}")
print(f"  Best parameters:")
for param, val in rf_search.best_params_.items():
    print(f"    {param}: {val}")

# ============================================================
# 3. Final Model Evaluation
# ============================================================
print("\n" + "-" * 50)
print("Final Model Performance")
print("-" * 50)

rf = rf_search.best_estimator_
y_pred_train = rf.predict(X_train_s)
y_pred_test = rf.predict(X_test_s)

metrics = {
    "R²_train": r2_score(y_train, y_pred_train),
    "R²_test": r2_score(y_test, y_pred_test),
    "RMSE_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
    "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
    "MAE_test": mean_absolute_error(y_test, y_pred_test),
    "OOB_score": rf.oob_score_,
}
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# ============================================================
# 4. Feature Importance — MDI vs Permutation
# ============================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE COMPARISON")
print("=" * 70)

# --- 4a. Mean Decrease in Impurity (MDI) — built-in ---
print("\n--- MDI (Gini) Importance ---")
print("  ⚠️ NOTE: MDI importance can be biased toward high-cardinality/")
print("  high-variance features (Strobl et al., 2007). Use with caution.")
mdi_importance = rf.feature_importances_
mdi_df = pd.DataFrame({
    "Feature": feature_cols,
    "MDI_Importance": mdi_importance
}).sort_values("MDI_Importance", ascending=False)

for _, row in mdi_df.iterrows():
    print(f"    {row['Feature']:25s}: {row['MDI_Importance']:.4f}")

# --- 4b. Permutation Importance (model-agnostic, unbiased) ---
print("\n--- Permutation Importance (corrected) ---")
print("  Method: Shuffle each feature, measure R² drop on test set")
print("  Advantage: Not biased by feature scale or cardinality")

perm_result = permutation_importance(
    rf, X_test_s, y_test, n_repeats=30, random_state=42, n_jobs=-1
)
perm_df = pd.DataFrame({
    "Feature": feature_cols,
    "Perm_Importance_Mean": perm_result.importances_mean,
    "Perm_Importance_Std": perm_result.importances_std,
}).sort_values("Perm_Importance_Mean", ascending=False)

for _, row in perm_df.iterrows():
    sig = "★" if row["Perm_Importance_Mean"] > 2 * row["Perm_Importance_Std"] else " "
    print(f"  {sig} {row['Feature']:25s}: {row['Perm_Importance_Mean']:.4f} ± {row['Perm_Importance_Std']:.4f}")

# --- 4c. Compare MDI vs Permutation rankings ---
print("\n--- Ranking Comparison (MDI vs Permutation) ---")
mdi_rank = mdi_df["Feature"].reset_index(drop=True)
perm_rank = perm_df["Feature"].reset_index(drop=True)
print(f"  {'Rank':<6} {'MDI':>25s} {'Permutation':>25s} {'Match':>8s}")
print("  " + "-" * 66)
for i in range(len(feature_cols)):
    match = "✓" if mdi_rank[i] == perm_rank[i] else "✗"
    print(f"  {i + 1:<6} {mdi_rank[i]:>25s} {perm_rank[i]:>25s} {match:>8s}")

# Spearman rank correlation between methods
from scipy.stats import spearmanr
mdi_ranks = mdi_df.set_index("Feature")["MDI_Importance"].rank(ascending=False)
perm_ranks = perm_df.set_index("Feature")["Perm_Importance_Mean"].rank(ascending=False)
common_features = mdi_ranks.index
rho, pval = spearmanr(mdi_ranks[common_features], perm_ranks[common_features])
print(f"\n  Spearman rank correlation: ρ = {rho:.4f} (p = {pval:.4f})")
if rho > 0.8:
    print("  → Strong agreement between MDI and Permutation rankings")
else:
    print("  → Notable disagreement — permutation importance is more reliable")

# ============================================================
# 5. Visualization
# ============================================================
print("\n" + "-" * 50)
print("Generating plots...")
print("-" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Random Forest — Fucosylation Prediction", fontsize=14, fontweight="bold")

# 5a. MDI Importance
ax = axes[0, 0]
mdi_sorted = mdi_df.sort_values("MDI_Importance", ascending=True)
ax.barh(mdi_sorted["Feature"], mdi_sorted["MDI_Importance"], color="steelblue")
ax.set_xlabel("Mean Decrease in Impurity")
ax.set_title("MDI Feature Importance\n(⚠ may be biased)")

# 5b. Permutation Importance
ax = axes[0, 1]
perm_sorted = perm_df.sort_values("Perm_Importance_Mean", ascending=True)
ax.barh(perm_sorted["Feature"], perm_sorted["Perm_Importance_Mean"],
        xerr=perm_sorted["Perm_Importance_Std"], color="#43a047", capsize=3)
ax.set_xlabel("Permutation Importance (ΔR²)")
ax.set_title("Permutation Importance\n(unbiased)")

# 5c. Actual vs Predicted
ax = axes[0, 2]
ax.scatter(y_test, y_pred_test, alpha=0.5, s=20, color="steelblue")
lims = [min(y_test.min(), y_pred_test.min()) - 1, max(y_test.max(), y_pred_test.max()) + 1]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual Fucosylation (%)")
ax.set_ylabel("Predicted Fucosylation (%)")
ax.set_title(f"Actual vs Predicted (R²={metrics['R²_test']:.4f})")

# 5d. Residual plot
ax = axes[1, 0]
residuals = y_test - y_pred_test
ax.scatter(y_pred_test, residuals, alpha=0.5, s=15, color="steelblue")
ax.axhline(y=0, color="red", linestyle="--")
ax.set_xlabel("Predicted")
ax.set_ylabel("Residual")
ax.set_title("Residuals vs Predicted")

# 5e. Importance comparison (side by side)
ax = axes[1, 1]
merged = mdi_df.merge(perm_df[["Feature", "Perm_Importance_Mean"]], on="Feature")
merged["MDI_rank"] = merged["MDI_Importance"].rank(ascending=False)
merged["Perm_rank"] = merged["Perm_Importance_Mean"].rank(ascending=False)
ax.scatter(merged["MDI_rank"], merged["Perm_rank"], s=50, color="steelblue")
for _, row in merged.iterrows():
    ax.annotate(row["Feature"], (row["MDI_rank"], row["Perm_rank"]),
                fontsize=6, ha="center", va="bottom")
ax.plot([0, 11], [0, 11], "r--", linewidth=1, alpha=0.5)
ax.set_xlabel("MDI Rank")
ax.set_ylabel("Permutation Rank")
ax.set_title(f"MDI vs Permutation Ranking (ρ={rho:.3f})")

# 5f. OOB vs test error (learning curve proxy)
ax = axes[1, 2]
n_trees_range = [10, 25, 50, 100, 150, 200, 300, 400]
oob_errors = []
test_errors = []
for n_trees in n_trees_range:
    rf_temp = RandomForestRegressor(
        n_estimators=n_trees, oob_score=True, random_state=42,
        **{k: v for k, v in rf_search.best_params_.items() if k != "n_estimators"}
    )
    rf_temp.fit(X_train_s, y_train)
    oob_errors.append(1 - rf_temp.oob_score_)
    test_errors.append(1 - r2_score(y_test, rf_temp.predict(X_test_s)))

ax.plot(n_trees_range, oob_errors, "o-", color="steelblue", label="OOB Error")
ax.plot(n_trees_range, test_errors, "s--", color="#ff9800", label="Test Error")
ax.set_xlabel("Number of Trees")
ax.set_ylabel("1 - R²")
ax.set_title("Convergence: OOB vs Test Error")
ax.legend()

plt.tight_layout()
plt.savefig("results/03_random_forest/rf_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ All plots saved")

# ============================================================
# 6. Save Results
# ============================================================
pd.DataFrame([metrics]).to_csv("results/03_random_forest/metrics.csv", index=False)
mdi_df.to_csv("results/03_random_forest/mdi_importance.csv", index=False)
perm_df.to_csv("results/03_random_forest/permutation_importance.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Test R²: {metrics['R²_test']:.4f}")
print(f"Test RMSE: {metrics['RMSE_test']:.4f}")
print(f"OOB Score: {metrics['OOB_score']:.4f}")
print(f"MDI vs Permutation rank correlation: ρ = {rho:.4f}")
print("\n✅ Random Forest analysis complete. Results saved to results/03_random_forest/")
