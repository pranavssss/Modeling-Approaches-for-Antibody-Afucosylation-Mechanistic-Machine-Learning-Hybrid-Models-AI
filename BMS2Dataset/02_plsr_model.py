"""
02_plsr_model.py
================
Partial Least Squares Regression (PLSR) — Primary Academic Model

PLSR is the most academically relevant model for bioprocessing applications,
directly aligned with chemometrics and PAT (Process Analytical Technology) literature.

Literature References:
- Wold et al. (2001): PLS-regression: a basic tool of chemometrics
- Geladi & Kowalski (1986): Partial least-squares regression — a tutorial
- Kirdar et al. (2008): Application of multivariate analysis to bioprocessing data
- BMS/Biogen internal reports on PLSR for CQA prediction

Key Features:
- VIP (Variable Importance in Projection) scores for interpretability
- Latent variable / component analysis
- Comparison with PCA to justify PLSR over unsupervised decomposition
- Cross-validated component selection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("results/02_plsr", exist_ok=True)

# ============================================================
# 1. Load and Prepare Data
# ============================================================
print("=" * 70)
print("02 — PARTIAL LEAST SQUARES REGRESSION (PLSR)")
print("=" * 70)

data = pd.read_csv("data/mab_fucosylation_dataset.csv")
feature_cols = [c for c in data.columns if c not in ["Fucosylation_pct", "Batch_ID"]]
X = data[feature_cols].values
y = data["Fucosylation_pct"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ============================================================
# 2. Optimal Component Selection via Cross-Validation
# ============================================================
print("\n" + "-" * 50)
print("Component Selection (CV)")
print("-" * 50)

max_components = min(X_train_s.shape[1], 10)
cv_results = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for n_comp in range(1, max_components + 1):
    pls_temp = PLSRegression(n_components=n_comp)
    scores = cross_val_score(pls_temp, X_train_s, y_train, cv=kf, scoring="r2")
    rmse_scores = cross_val_score(
        pls_temp, X_train_s, y_train, cv=kf,
        scoring="neg_root_mean_squared_error"
    )
    cv_results.append({
        "n_components": n_comp,
        "cv_r2_mean": scores.mean(),
        "cv_r2_std": scores.std(),
        "cv_rmse_mean": -rmse_scores.mean(),
        "cv_rmse_std": rmse_scores.std(),
    })
    print(f"  {n_comp} components: R²={scores.mean():.4f} ± {scores.std():.4f}, "
          f"RMSE={-rmse_scores.mean():.4f}")

cv_df = pd.DataFrame(cv_results)
optimal_n = cv_df.loc[cv_df["cv_r2_mean"].idxmax(), "n_components"]
print(f"\n  → Optimal components: {int(optimal_n)}")

# ============================================================
# 3. Fit Final PLSR Model
# ============================================================
print("\n" + "-" * 50)
print(f"Final PLSR Model ({int(optimal_n)} components)")
print("-" * 50)

pls = PLSRegression(n_components=int(optimal_n))
pls.fit(X_train_s, y_train)
y_pred_train = pls.predict(X_train_s).ravel()
y_pred_test = pls.predict(X_test_s).ravel()

metrics = {
    "R²_train": r2_score(y_train, y_pred_train),
    "R²_test": r2_score(y_test, y_pred_test),
    "RMSE_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
    "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
    "MAE_test": mean_absolute_error(y_test, y_pred_test),
}
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# ============================================================
# 4. VIP Scores (Variable Importance in Projection)
# ============================================================
print("\n" + "-" * 50)
print("VIP Scores — Feature Importance")
print("-" * 50)


def compute_vip(pls_model, X_s, y_vals):
    """
    Compute VIP scores following Wold et al. (2001).

    VIP_j = sqrt(p * Σ_h [SS_h * (w_jh / ||w_h||)²] / Σ_h SS_h)

    where:
    - p = number of predictors
    - h = component index
    - SS_h = sum of squares explained by component h
    - w_jh = weight of variable j in component h
    """
    t = pls_model.x_scores_       # Scores (n × n_comp)
    w = pls_model.x_weights_      # Weights (p × n_comp)
    q = pls_model.y_loadings_     # Y-loadings (1 × n_comp)

    p_vars, n_comp = w.shape
    ss = np.zeros(n_comp)

    for h in range(n_comp):
        ss[h] = (q[0, h] ** 2) * np.dot(t[:, h], t[:, h])

    total_ss = np.sum(ss)
    vip = np.zeros(p_vars)

    for j in range(p_vars):
        weighted_sum = 0
        for h in range(n_comp):
            weighted_sum += ss[h] * (w[j, h] / np.linalg.norm(w[:, h])) ** 2
        vip[j] = np.sqrt(p_vars * weighted_sum / total_ss)

    return vip


vip_scores = compute_vip(pls, X_train_s, y_train)
vip_df = pd.DataFrame({
    "Feature": feature_cols,
    "VIP": vip_scores
}).sort_values("VIP", ascending=False)

print("\n  Variable Importance in Projection:")
for _, row in vip_df.iterrows():
    significance = "★ SIGNIFICANT" if row["VIP"] > 1.0 else "  moderate" if row["VIP"] > 0.8 else "  low"
    print(f"    {row['Feature']:25s}: {row['VIP']:.4f} {significance}")

print("\n  Rule: VIP > 1.0 = significant influence on response")
print(f"  Significant features: {(vip_scores > 1.0).sum()} / {len(feature_cols)}")

# ============================================================
# 5. Latent Space Interpretation
# ============================================================
print("\n" + "-" * 50)
print("Latent Space Analysis")
print("-" * 50)

# X-loadings: how original variables map to latent components
x_loadings = pls.x_loadings_  # (p × n_comp)
print("\n  X-Loadings (feature contributions to each latent variable):")
loadings_df = pd.DataFrame(
    x_loadings,
    index=feature_cols,
    columns=[f"LV{i + 1}" for i in range(int(optimal_n))]
)
print(loadings_df.round(4).to_string())

# Variance explained by each component
x_scores = pls.x_scores_
total_var = np.sum(np.var(X_train_s, axis=0))
var_explained = []
for i in range(int(optimal_n)):
    # Reconstruct X from component i
    comp_var = np.var(x_scores[:, i]) * np.sum(x_loadings[:, i] ** 2)
    var_explained.append(comp_var / total_var * 100)

print(f"\n  Variance explained per component:")
for i, ve in enumerate(var_explained):
    print(f"    LV{i + 1}: {ve:.2f}%")
print(f"    Cumulative: {sum(var_explained):.2f}%")

# ============================================================
# 6. PCA Comparison (Why PLSR > PCA+Regression)
# ============================================================
print("\n" + "=" * 70)
print("PCA vs PLSR COMPARISON")
print("=" * 70)
print("Justification: PLSR maximizes covariance with Y,")
print("while PCA only maximizes variance in X (ignoring Y).")

pca_results = []
for n_comp_pca in range(1, max_components + 1):
    pca = PCA(n_components=n_comp_pca)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    lr = LinearRegression()
    lr.fit(X_train_pca, y_train)
    y_pred_pca = lr.predict(X_test_pca)

    pca_results.append({
        "n_components": n_comp_pca,
        "R²_test": r2_score(y_test, y_pred_pca),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_pca)),
        "var_explained_pct": pca.explained_variance_ratio_.sum() * 100,
    })

pca_df = pd.DataFrame(pca_results)

print("\n  Components | PLSR R² (test) | PCR R² (test) | PCA Var Explained")
print("  " + "-" * 65)
for i in range(max_components):
    plsr_r2 = cv_df.iloc[i]["cv_r2_mean"] if i < len(cv_df) else np.nan
    pcr_r2 = pca_df.iloc[i]["R²_test"]
    pca_var = pca_df.iloc[i]["var_explained_pct"]
    winner = "← PLSR" if plsr_r2 > pcr_r2 else "← PCR"
    print(f"  {i + 1:10d} | {plsr_r2:14.4f} | {pcr_r2:13.4f} | {pca_var:16.1f}% {winner}")

print("\n  Key Insight: PLSR typically outperforms PCR because it finds")
print("  latent variables that are relevant to the response variable (Y),")
print("  not just directions of maximum variance in X.")

# ============================================================
# 7. Visualization
# ============================================================
print("\n" + "-" * 50)
print("Generating plots...")
print("-" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("PLSR Analysis — Fucosylation Prediction", fontsize=14, fontweight="bold")

# 7a. Component selection
ax = axes[0, 0]
ax.errorbar(cv_df["n_components"], cv_df["cv_r2_mean"], yerr=cv_df["cv_r2_std"],
            marker="o", color="steelblue", capsize=3, linewidth=1.5)
ax.axvline(x=optimal_n, color="red", linestyle="--", alpha=0.7, label=f"Optimal: {int(optimal_n)}")
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cross-Validated R²")
ax.set_title("Component Selection")
ax.legend()

# 7b. VIP scores
ax = axes[0, 1]
vip_sorted = vip_df.sort_values("VIP", ascending=True)
colors = ["#d32f2f" if v > 1.0 else "#90a4ae" for v in vip_sorted["VIP"]]
ax.barh(vip_sorted["Feature"], vip_sorted["VIP"], color=colors)
ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1, label="VIP = 1.0 threshold")
ax.set_xlabel("VIP Score")
ax.set_title("Variable Importance in Projection (VIP)")
ax.legend()

# 7c. Actual vs Predicted
ax = axes[0, 2]
ax.scatter(y_test, y_pred_test, alpha=0.5, s=20, color="steelblue")
lims = [min(y_test.min(), y_pred_test.min()) - 1, max(y_test.max(), y_pred_test.max()) + 1]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual Fucosylation (%)")
ax.set_ylabel("Predicted Fucosylation (%)")
ax.set_title(f"Actual vs Predicted (R²={metrics['R²_test']:.4f})")

# 7d. Latent space (scores plot LV1 vs LV2)
ax = axes[1, 0]
if int(optimal_n) >= 2:
    scatter = ax.scatter(x_scores[:, 0], x_scores[:, 1], c=y_train, cmap="coolwarm",
                         alpha=0.6, s=15)
    plt.colorbar(scatter, ax=ax, label="Fucosylation (%)")
    ax.set_xlabel("Latent Variable 1")
    ax.set_ylabel("Latent Variable 2")
    ax.set_title("Score Plot (colored by target)")
else:
    ax.text(0.5, 0.5, "Only 1 component selected", ha="center", va="center", fontsize=12)

# 7e. Loading plot (biplot of features on LV1 vs LV2)
ax = axes[1, 1]
if int(optimal_n) >= 2:
    for i, feat in enumerate(feature_cols):
        ax.annotate(feat, (x_loadings[i, 0], x_loadings[i, 1]),
                    fontsize=7, ha="center")
        ax.arrow(0, 0, x_loadings[i, 0] * 0.95, x_loadings[i, 1] * 0.95,
                 head_width=0.01, head_length=0.005, fc="steelblue", ec="steelblue", alpha=0.7)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Loading on LV1")
    ax.set_ylabel("Loading on LV2")
    ax.set_title("Loading Plot (Feature Space)")
else:
    ax.text(0.5, 0.5, "Only 1 component selected", ha="center", va="center", fontsize=12)

# 7f. PLSR vs PCR comparison
ax = axes[1, 2]
ax.plot(cv_df["n_components"], cv_df["cv_r2_mean"], "o-", color="steelblue",
        label="PLSR", linewidth=1.5)
ax.plot(pca_df["n_components"], pca_df["R²_test"], "s--", color="#ff9800",
        label="PCR (PCA+OLS)", linewidth=1.5)
ax.set_xlabel("Number of Components")
ax.set_ylabel("R² (Test)")
ax.set_title("PLSR vs PCR Comparison")
ax.legend()

plt.tight_layout()
plt.savefig("results/02_plsr/plsr_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ All plots saved")

# ============================================================
# 8. Save Results
# ============================================================
pd.DataFrame([metrics]).to_csv("results/02_plsr/metrics.csv", index=False)
vip_df.to_csv("results/02_plsr/vip_scores.csv", index=False)
loadings_df.to_csv("results/02_plsr/x_loadings.csv")
cv_df.to_csv("results/02_plsr/cv_component_selection.csv", index=False)

comparison_df = pd.merge(
    cv_df[["n_components", "cv_r2_mean"]].rename(columns={"cv_r2_mean": "PLSR_R2"}),
    pca_df[["n_components", "R²_test"]].rename(columns={"R²_test": "PCR_R2"}),
    on="n_components"
)
comparison_df.to_csv("results/02_plsr/plsr_vs_pcr.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Optimal components: {int(optimal_n)}")
print(f"Test R²: {metrics['R²_test']:.4f}")
print(f"Test RMSE: {metrics['RMSE_test']:.4f}")
print(f"Significant features (VIP > 1): {(vip_scores > 1.0).sum()}")
print("\n✅ PLSR analysis complete. Results saved to results/02_plsr/")
