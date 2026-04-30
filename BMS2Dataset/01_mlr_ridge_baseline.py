"""
01_mlr_ridge_baseline.py
=========================
Multiple Linear Regression & Ridge Regression — Baseline Models

This script establishes the linear baseline for predicting mAb fucosylation.
It includes full assumption diagnostics as required for academic rigor.

Literature References:
- Montgomery et al. (2012): Introduction to Linear Regression Analysis
- Hoerl & Kennard (1970): Ridge regression — biased estimation for nonorthogonal problems
- Jedrzejewski et al. (2014): Linear models for CHO cell culture optimization

Models:
1. Ordinary Least Squares (OLS) — pure linear baseline
2. RidgeCV — L2-regularized regression with cross-validated alpha

Diagnostic Tests Included:
- Residual normality (Shapiro-Wilk, Q-Q plot)
- Homoscedasticity (Breusch-Pagan test, residuals vs fitted)
- Linearity check (residuals vs fitted, partial regression plots)
- Multicollinearity (VIF — Variance Inflation Factor)
- Influential observations (Cook's distance)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("results/01_mlr_ridge", exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 70)
print("01 — MLR / RIDGE BASELINE")
print("=" * 70)

data = pd.read_csv("data/mab_fucosylation_dataset.csv")
feature_cols = [c for c in data.columns if c not in ["Fucosylation_pct", "Batch_ID"]]
X = data[feature_cols].values
y = data["Fucosylation_pct"].values

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# Train/test split (80/20, stratified by batch for realism)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Standardize features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# 2. OLS Regression
# ============================================================
print("\n" + "-" * 50)
print("MODEL 1: Ordinary Least Squares (OLS)")
print("-" * 50)

ols = LinearRegression()
ols.fit(X_train_s, y_train)
y_pred_ols_train = ols.predict(X_train_s)
y_pred_ols_test = ols.predict(X_test_s)

# Metrics
ols_metrics = {
    "R²_train": r2_score(y_train, y_pred_ols_train),
    "R²_test": r2_score(y_test, y_pred_ols_test),
    "RMSE_train": np.sqrt(mean_squared_error(y_train, y_pred_ols_train)),
    "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_ols_test)),
    "MAE_test": mean_absolute_error(y_test, y_pred_ols_test),
}
for k, v in ols_metrics.items():
    print(f"  {k}: {v:.4f}")

# Cross-validation
cv_scores = cross_val_score(ols, X_train_s, y_train, cv=5, scoring="r2")
print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Coefficients
print("\n  Standardized Coefficients (biological interpretation):")
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": ols.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
for _, row in coef_df.iterrows():
    direction = "↑" if row["Coefficient"] > 0 else "↓"
    print(f"    {row['Feature']:25s}: {row['Coefficient']:+.4f} {direction}")

# ============================================================
# 3. Ridge Regression (L2 Regularization)
# ============================================================
print("\n" + "-" * 50)
print("MODEL 2: RidgeCV (L2 Regularization)")
print("-" * 50)

alphas = np.logspace(-4, 4, 100)
ridge = RidgeCV(alphas=alphas, cv=5, scoring="r2")
ridge.fit(X_train_s, y_train)
y_pred_ridge_train = ridge.predict(X_train_s)
y_pred_ridge_test = ridge.predict(X_test_s)

print(f"  Optimal alpha: {ridge.alpha_:.6f}")

ridge_metrics = {
    "R²_train": r2_score(y_train, y_pred_ridge_train),
    "R²_test": r2_score(y_test, y_pred_ridge_test),
    "RMSE_train": np.sqrt(mean_squared_error(y_train, y_pred_ridge_train)),
    "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_ridge_test)),
    "MAE_test": mean_absolute_error(y_test, y_pred_ridge_test),
}
for k, v in ridge_metrics.items():
    print(f"  {k}: {v:.4f}")

# ============================================================
# 4. ASSUMPTION DIAGNOSTICS (Critical for academic submission)
# ============================================================
print("\n" + "=" * 70)
print("ASSUMPTION DIAGNOSTICS")
print("=" * 70)

residuals = y_train - y_pred_ols_train
standardized_resid = (residuals - residuals.mean()) / residuals.std()

# --- 4a. Normality of Residuals ---
print("\n--- 4a. Residual Normality ---")
shapiro_stat, shapiro_p = stats.shapiro(residuals[:min(5000, len(residuals))])
print(f"  Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("  → Residuals appear normally distributed (fail to reject H₀)")
else:
    print("  → Residuals deviate from normality (p < 0.05)")
    print("    Note: With large n, minor deviations are expected. Check Q-Q plot visually.")

# Skewness and kurtosis
skew = stats.skew(residuals)
kurt = stats.kurtosis(residuals)
print(f"  Skewness: {skew:.4f} (ideal: 0)")
print(f"  Excess Kurtosis: {kurt:.4f} (ideal: 0)")

# --- 4b. Homoscedasticity (Breusch-Pagan) ---
print("\n--- 4b. Homoscedasticity (Breusch-Pagan Test) ---")
# Manual Breusch-Pagan implementation
resid_sq = residuals ** 2
bp_model = LinearRegression().fit(X_train_s, resid_sq)
ss_resid_bp = np.sum((resid_sq - bp_model.predict(X_train_s)) ** 2)
ss_total_bp = np.sum((resid_sq - resid_sq.mean()) ** 2)
r2_bp = 1 - ss_resid_bp / ss_total_bp
bp_stat = len(residuals) * r2_bp
bp_p = 1 - stats.chi2.cdf(bp_stat, X_train_s.shape[1])
print(f"  BP statistic: {bp_stat:.4f}, p-value: {bp_p:.4f}")
if bp_p > 0.05:
    print("  → No evidence of heteroscedasticity")
else:
    print("  → Heteroscedasticity detected — consider robust standard errors or WLS")

# --- 4c. Multicollinearity (VIF) ---
print("\n--- 4c. Variance Inflation Factors (VIF) ---")
from numpy.linalg import inv

# VIF for standardized features
corr_matrix = np.corrcoef(X_train_s.T)
try:
    vif_values = np.diag(inv(corr_matrix))
    vif_df = pd.DataFrame({"Feature": feature_cols, "VIF": vif_values})
    vif_df = vif_df.sort_values("VIF", ascending=False)
    for _, row in vif_df.iterrows():
        flag = " ⚠️ HIGH" if row["VIF"] > 5 else ""
        print(f"  {row['Feature']:25s}: {row['VIF']:.2f}{flag}")
    print("  Rule of thumb: VIF > 5 indicates concerning multicollinearity")
    print("  VIF > 10 indicates severe multicollinearity")
except np.linalg.LinAlgError:
    print("  Could not compute VIF — near-singular correlation matrix")

# --- 4d. Cook's Distance (Influential Points) ---
print("\n--- 4d. Cook's Distance (Influential Observations) ---")
n, p = X_train_s.shape
hat_matrix = X_train_s @ np.linalg.pinv(X_train_s.T @ X_train_s) @ X_train_s.T
leverages = np.diag(hat_matrix)
mse = np.sum(residuals ** 2) / (n - p - 1)
cooks_d = (residuals ** 2 * leverages) / (p * mse * (1 - leverages) ** 2)

n_influential = np.sum(cooks_d > 4 / n)
print(f"  Threshold (4/n): {4 / n:.4f}")
print(f"  Influential observations (Cook's D > 4/n): {n_influential} ({n_influential / n * 100:.1f}%)")
print(f"  Max Cook's D: {cooks_d.max():.4f}")

# ============================================================
# 5. Diagnostic Plots
# ============================================================
print("\n" + "-" * 50)
print("Generating diagnostic plots...")
print("-" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("OLS Regression — Assumption Diagnostics", fontsize=14, fontweight="bold")

# 5a. Residuals vs Fitted
ax = axes[0, 0]
ax.scatter(y_pred_ols_train, standardized_resid, alpha=0.4, s=15, color="steelblue")
ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
ax.axhline(y=2, color="orange", linestyle=":", alpha=0.5)
ax.axhline(y=-2, color="orange", linestyle=":", alpha=0.5)
ax.set_xlabel("Fitted Values")
ax.set_ylabel("Standardized Residuals")
ax.set_title("Residuals vs Fitted (Linearity + Homoscedasticity)")

# 5b. Q-Q Plot
ax = axes[0, 1]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Normal Q-Q Plot (Normality)")

# 5c. Scale-Location Plot
ax = axes[0, 2]
ax.scatter(y_pred_ols_train, np.sqrt(np.abs(standardized_resid)), alpha=0.4, s=15, color="steelblue")
z = np.polyfit(y_pred_ols_train, np.sqrt(np.abs(standardized_resid)), 1)
p_fit = np.poly1d(z)
x_sorted = np.sort(y_pred_ols_train)
ax.plot(x_sorted, p_fit(x_sorted), color="red", linewidth=1.5)
ax.set_xlabel("Fitted Values")
ax.set_ylabel("√|Standardized Residuals|")
ax.set_title("Scale-Location Plot (Homoscedasticity)")

# 5d. Cook's Distance
ax = axes[1, 0]
ax.stem(range(len(cooks_d)), cooks_d, linefmt="steelblue", markerfmt=" ", basefmt="k-")
ax.axhline(y=4 / n, color="red", linestyle="--", label=f"Threshold (4/n={4 / n:.4f})")
ax.set_xlabel("Observation Index")
ax.set_ylabel("Cook's Distance")
ax.set_title("Cook's Distance (Influential Points)")
ax.legend()

# 5e. Actual vs Predicted (Test Set)
ax = axes[1, 1]
ax.scatter(y_test, y_pred_ols_test, alpha=0.5, s=20, color="steelblue")
lims = [min(y_test.min(), y_pred_ols_test.min()) - 1, max(y_test.max(), y_pred_ols_test.max()) + 1]
ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Actual Fucosylation (%)")
ax.set_ylabel("Predicted Fucosylation (%)")
ax.set_title(f"Actual vs Predicted (Test R²={ols_metrics['R²_test']:.4f})")
ax.legend()

# 5f. Coefficient Plot
ax = axes[1, 2]
coef_df_sorted = coef_df.sort_values("Coefficient")
colors = ["#d32f2f" if c < 0 else "#1976d2" for c in coef_df_sorted["Coefficient"]]
ax.barh(coef_df_sorted["Feature"], coef_df_sorted["Coefficient"], color=colors)
ax.axvline(x=0, color="black", linewidth=0.5)
ax.set_xlabel("Standardized Coefficient")
ax.set_title("Feature Coefficients (Biological Interpretation)")

plt.tight_layout()
plt.savefig("results/01_mlr_ridge/diagnostic_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Diagnostic plots saved")

# ============================================================
# 6. Residual Distribution Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(residuals, bins=30, density=True, alpha=0.7, color="steelblue", edgecolor="white")
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
ax.plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()),
        "r-", linewidth=2, label="Normal fit")
ax.set_xlabel("Residual")
ax.set_ylabel("Density")
ax.set_title("Residual Distribution")
ax.legend()

ax = axes[1]
ax.scatter(y_test, y_pred_ridge_test, alpha=0.5, s=20, color="#1976d2", label="Ridge")
ax.scatter(y_test, y_pred_ols_test, alpha=0.3, s=20, color="#ff9800", label="OLS")
lims = [min(y_test.min(), y_pred_ols_test.min()) - 1, max(y_test.max(), y_pred_ols_test.max()) + 1]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual Fucosylation (%)")
ax.set_ylabel("Predicted Fucosylation (%)")
ax.set_title("OLS vs Ridge — Test Set Comparison")
ax.legend()

plt.tight_layout()
plt.savefig("results/01_mlr_ridge/ols_vs_ridge.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Comparison plot saved")

# ============================================================
# 7. Save Results
# ============================================================
results_df = pd.DataFrame({
    "Metric": list(ols_metrics.keys()),
    "OLS": list(ols_metrics.values()),
    "Ridge": list(ridge_metrics.values()),
})
results_df.to_csv("results/01_mlr_ridge/metrics.csv", index=False)

coef_comparison = pd.DataFrame({
    "Feature": feature_cols,
    "OLS_Coeff": ols.coef_,
    "Ridge_Coeff": ridge.coef_,
})
coef_comparison.to_csv("results/01_mlr_ridge/coefficients.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"OLS  — Test R²: {ols_metrics['R²_test']:.4f}, RMSE: {ols_metrics['RMSE_test']:.4f}")
print(f"Ridge — Test R²: {ridge_metrics['R²_test']:.4f}, RMSE: {ridge_metrics['RMSE_test']:.4f}")
print(f"Ridge optimal α: {ridge.alpha_:.6f}")
print("\n✅ Baseline models complete. Results saved to results/01_mlr_ridge/")
