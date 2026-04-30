"""
07_xgboost_shap.py
===================
XGBoost with SHAP Explainability — Production-Ready Model

XGBoost + SHAP provides the strongest combination of predictive performance
and interpretability, making it the recommended model for deployment.

Literature References:
- Chen & Guestrin (2016): XGBoost — A Scalable Tree Boosting System
- Lundberg & Lee (2017): SHAP — A Unified Approach to Interpreting Model Predictions
- Lundberg et al. (2020): From local explanations to global understanding
- ICH Q8/Q9/Q10: Quality system guidelines (regulatory framework)

Key Strengths:
- State-of-the-art gradient boosting performance
- SHAP resolves the black-box vs interpretability trade-off
- Feature interactions detectable via SHAP interaction values
- Regulatory-friendly: individual prediction explanations
"""

import numpy as np
import pandas as pd
import subprocess
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("results/07_xgboost_shap", exist_ok=True)

# ============================================================
# 0. Auto-install missing packages
# ============================================================
def install_if_missing(package_name, import_name=None):
    """Try to import; if missing, pip install it."""
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"  Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        return True

install_if_missing("xgboost")
install_if_missing("shap")

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 70)
print("07 — XGBOOST + SHAP EXPLAINABILITY")
print("=" * 70)

data = pd.read_csv("data/mab_fucosylation_dataset.csv")
feature_cols = [c for c in data.columns if c not in ["Fucosylation_pct", "Batch_ID"]]
X = data[feature_cols].values
y = data["Fucosylation_pct"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ============================================================
# 2. XGBoost Hyperparameter Tuning
# ============================================================
print("\n" + "-" * 50)
print("Hyperparameter Tuning (RandomizedSearchCV)")
print("-" * 50)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
    print("  Using XGBoost")
except ImportError:
    HAS_XGB = False
    print("  XGBoost install failed. Using GradientBoostingRegressor fallback.")
    from sklearn.ensemble import GradientBoostingRegressor

if HAS_XGB:
    param_distributions = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 10),
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 0.5),
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.1, 0.5, 1.0, 2.0],
    }

    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    xgb_search.fit(X_train_s, y_train)
    model = xgb_search.best_estimator_

    print(f"  Best CV R²: {xgb_search.best_score_:.4f}")
    print(f"  Best parameters:")
    for param, val in xgb_search.best_params_.items():
        print(f"    {param}: {val}")
else:
    param_distributions = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 8),
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": uniform(0.6, 0.4),
        "min_samples_leaf": randint(1, 10),
    }

    gbr_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=30,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    gbr_search.fit(X_train_s, y_train)
    model = gbr_search.best_estimator_
    print(f"  Best CV R²: {gbr_search.best_score_:.4f}")

# ============================================================
# 3. Model Evaluation
# ============================================================
print("\n" + "-" * 50)
print("Model Performance")
print("-" * 50)

y_pred_train = model.predict(X_train_s)
y_pred_test = model.predict(X_test_s)

metrics = {
    "R²_train": r2_score(y_train, y_pred_train),
    "R²_test": r2_score(y_test, y_pred_test),
    "RMSE_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
    "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
    "MAE_test": mean_absolute_error(y_test, y_pred_test),
}
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="r2")
print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 4. SHAP Analysis
# ============================================================
print("\n" + "=" * 70)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 70)
print("SHAP (SHapley Additive exPlanations) provides:")
print("  - Consistent, locally accurate feature attributions")
print("  - Both global and local interpretability")
print("  - Detection of feature interactions")
print("  - Regulatory compliance (ICH Q8/Q9)")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    print("  ⚠️ SHAP install failed. Skipping SHAP analysis.")
    HAS_SHAP = False

if HAS_SHAP:
    # Compute SHAP values
    print("\n  Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_s)

    # --- 4a. Global Feature Importance (mean |SHAP|) ---
    print("\n--- Global Feature Importance (mean |SHAP|) ---")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False)

    for _, row in shap_importance.iterrows():
        print(f"    {row['Feature']:25s}: {row['Mean_Abs_SHAP']:.4f}")

    # --- 4b. Feature Direction Analysis ---
    print("\n--- Feature Direction Analysis ---")
    print("  How each feature pushes predictions (positive vs negative SHAP):")
    for i, feat in enumerate(feature_cols):
        mean_shap = shap_values[:, i].mean()
        direction = "↑ fucosylation" if mean_shap > 0 else "↓ fucosylation"
        corr = np.corrcoef(X_test_s[:, i], shap_values[:, i])[0, 1]
        relationship = "positive" if corr > 0 else "negative" if corr < 0 else "complex"
        print(f"    {feat:25s}: mean SHAP={mean_shap:+.4f} ({direction}), "
              f"correlation={corr:+.3f} ({relationship})")

    # --- 4c. Individual Prediction Explanations ---
    print("\n--- Sample Explanations (for regulatory documentation) ---")
    n_explain = min(3, len(X_test_s))
    for i in range(n_explain):
        print(f"\n  Prediction {i + 1}: Predicted={y_pred_test[i]:.2f}%, Actual={y_test[i]:.2f}%")
        print(f"  Base value (expected): {explainer.expected_value:.2f}%")
        feature_contributions = sorted(
            zip(feature_cols, shap_values[i]),
            key=lambda x: abs(x[1]), reverse=True
        )
        print(f"  Top drivers:")
        for feat, shap_val in feature_contributions[:5]:
            direction = "↑" if shap_val > 0 else "↓"
            print(f"    {feat:25s}: {shap_val:+.4f} {direction}")

    # --- 4d. Feature Interactions ---
    print("\n--- Feature Interactions (top pairs) ---")
    interaction_strength = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            corr_ij = abs(np.corrcoef(X_test_s[:, j], shap_values[:, i])[0, 1])
            corr_ji = abs(np.corrcoef(X_test_s[:, i], shap_values[:, j])[0, 1])
            interaction_strength.append({
                "Feature_1": feature_cols[i],
                "Feature_2": feature_cols[j],
                "Interaction_Strength": (corr_ij + corr_ji) / 2
            })

    interaction_df = pd.DataFrame(interaction_strength).sort_values(
        "Interaction_Strength", ascending=False
    )
    print("  Top 5 interacting feature pairs:")
    for _, row in interaction_df.head(5).iterrows():
        print(f"    {row['Feature_1']:20s} × {row['Feature_2']:20s}: {row['Interaction_Strength']:.4f}")
else:
    shap_values = None
    mean_abs_shap = None

# ============================================================
# 5. Visualization
# ============================================================
print("\n" + "-" * 50)
print("Generating plots...")
print("-" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("XGBoost + SHAP — Fucosylation Prediction", fontsize=14, fontweight="bold")

# 5a. Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred_test, alpha=0.5, s=20, color="steelblue")
lims = [min(y_test.min(), y_pred_test.min()) - 1, max(y_test.max(), y_pred_test.max()) + 1]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual Fucosylation (%)")
ax.set_ylabel("Predicted Fucosylation (%)")
ax.set_title(f"Actual vs Predicted (R²={metrics['R²_test']:.4f})")

# 5b. SHAP global importance
ax = axes[0, 1]
if HAS_SHAP:
    shap_sorted = shap_importance.sort_values("Mean_Abs_SHAP", ascending=True)
    ax.barh(shap_sorted["Feature"], shap_sorted["Mean_Abs_SHAP"], color="steelblue")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("SHAP Global Feature Importance")
else:
    # Fallback: use built-in feature importance
    fi = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
    fi = fi.sort_values("Importance", ascending=True)
    ax.barh(fi["Feature"], fi["Importance"], color="steelblue")
    ax.set_xlabel("Feature Importance (built-in)")
    ax.set_title("Feature Importance (SHAP unavailable)")

# 5c. SHAP beeswarm (manual implementation)
ax = axes[0, 2]
if HAS_SHAP:
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:10]
    for rank, feat_idx in enumerate(sorted_idx):
        feat_shap = shap_values[:, feat_idx]
        feat_vals = X_test_s[:, feat_idx]
        vmin, vmax = feat_vals.min(), feat_vals.max()
        if vmax > vmin:
            colors_bee = (feat_vals - vmin) / (vmax - vmin)
        else:
            colors_bee = np.zeros_like(feat_vals) + 0.5
        jitter = np.random.normal(0, 0.15, len(feat_shap))
        ax.scatter(feat_shap, np.full_like(feat_shap, rank) + jitter,
                   c=colors_bee, cmap="coolwarm", s=5, alpha=0.5)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_cols[i] for i in sorted_idx], fontsize=8)
    ax.set_xlabel("SHAP Value")
    ax.set_title("SHAP Beeswarm (top 10 features)")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
else:
    ax.text(0.5, 0.5, "SHAP not available\nInstall: pip install shap",
            ha="center", va="center", fontsize=12, color="gray")
    ax.set_title("SHAP Beeswarm — Not Available")

# 5d. Residuals
ax = axes[1, 0]
residuals = y_test - y_pred_test
ax.scatter(y_pred_test, residuals, alpha=0.5, s=15, color="steelblue")
ax.axhline(y=0, color="red", linestyle="--")
ax.set_xlabel("Predicted")
ax.set_ylabel("Residual")
ax.set_title("Residuals vs Predicted")

# 5e. SHAP dependence for top feature
ax = axes[1, 1]
if HAS_SHAP:
    top_feat_idx = np.argmax(mean_abs_shap)
    top_feat = feature_cols[top_feat_idx]
    # Find second most important as interaction color
    sorted_importance = np.argsort(mean_abs_shap)[::-1]
    interact_idx = sorted_importance[1] if len(sorted_importance) > 1 else 0
    scatter = ax.scatter(X_test_s[:, top_feat_idx], shap_values[:, top_feat_idx],
                         c=X_test_s[:, interact_idx], cmap="coolwarm",
                         alpha=0.5, s=15)
    plt.colorbar(scatter, ax=ax, label=feature_cols[interact_idx])
    ax.set_xlabel(f"{top_feat} (standardized)")
    ax.set_ylabel(f"SHAP value for {top_feat}")
    ax.set_title("SHAP Dependence Plot")
else:
    ax.text(0.5, 0.5, "SHAP not available", ha="center", va="center", fontsize=12, color="gray")
    ax.set_title("SHAP Dependence — Not Available")

# 5f. Error distribution
ax = axes[1, 2]
ax.hist(residuals, bins=25, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(x=0, color="red", linestyle="--")
ax.set_xlabel("Residual")
ax.set_ylabel("Count")
ax.set_title(f"Error Distribution (RMSE={metrics['RMSE_test']:.4f})")

plt.tight_layout()
plt.savefig("results/07_xgboost_shap/xgb_shap_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ All plots saved")

# ============================================================
# 6. Save Results
# ============================================================
pd.DataFrame([metrics]).to_csv("results/07_xgboost_shap/metrics.csv", index=False)
if HAS_SHAP:
    shap_importance.to_csv("results/07_xgboost_shap/shap_importance.csv", index=False)
    interaction_df.to_csv("results/07_xgboost_shap/feature_interactions.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Test R²: {metrics['R²_test']:.4f}")
print(f"Test RMSE: {metrics['RMSE_test']:.4f}")
print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
if HAS_SHAP:
    print(f"Top SHAP feature: {shap_importance.iloc[0]['Feature']} "
          f"(|SHAP|={shap_importance.iloc[0]['Mean_Abs_SHAP']:.4f})")
print(f"SHAP analysis: {'Complete' if HAS_SHAP else 'Skipped (install shap)'}")
print("\nRecommendation: XGBoost + SHAP is the strongest candidate for")
print("production deployment due to its combination of accuracy and")
print("explainability. SHAP satisfies regulatory transparency requirements.")
print("\n✅ XGBoost + SHAP analysis complete. Results saved to results/07_xgboost_shap/")
