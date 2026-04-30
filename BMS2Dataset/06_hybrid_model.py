"""
06_hybrid_model.py
===================
Physics-Informed Hybrid Model — Mechanistic + ML Integration

This is the differentiator model that combines domain knowledge (biochemistry)
with machine learning. Each engineered feature is explicitly linked to its
biological mechanism, satisfying both academic and industrial requirements.

Literature References:
- Von Stosch et al. (2014): Hybrid semi-parametric modeling in process systems engineering
- Raju et al. (2000): Species-specific variation in glycosylation of IgG
- Hossler et al. (2009): Optimal and consistent protein glycosylation in mammalian cell culture
- Jedrzejewski et al. (2014): Towards a mechanistic model of CHO cell metabolism
- Sha et al. (2016): Mechanistic modeling and simulation of glycosylation

Engineered Features (with biological justification):
1. FPI (Fucosylation Potential Index) — substrate availability for FUT8
2. Energy Charge — cellular energy status affecting Golgi function
3. Golgi Proxy — approximation of Golgi apparatus processing capacity
4. Metabolic Stress Index — aggregate stress affecting glycosylation fidelity
5. Enzyme Activity Factor — environmental conditions for glycosyltransferase activity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("results/06_hybrid", exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 70)
print("06 — PHYSICS-INFORMED HYBRID MODEL")
print("=" * 70)

data = pd.read_csv("data/mab_fucosylation_dataset.csv")
feature_cols = [c for c in data.columns if c not in ["Fucosylation_pct", "Batch_ID"]]
X_raw = data[feature_cols].copy()
y = data["Fucosylation_pct"].values

print(f"Dataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} raw features")

# ============================================================
# 2. Physics-Informed Feature Engineering
# ============================================================
print("\n" + "=" * 70)
print("PHYSICS-INFORMED FEATURE ENGINEERING")
print("=" * 70)

features_eng = X_raw.copy()

# --- Feature 1: Fucosylation Potential Index (FPI) ---
# Biological Mechanism:
#   FUT8 (α-1,6-fucosyltransferase) catalyzes the transfer of fucose from
#   GDP-fucose to the core GlcNAc of N-glycans. The reaction rate depends on:
#   - GDP-Fucose concentration (substrate)
#   - Mn²⁺ concentration (divalent cation cofactor required by FUT8)
#   - The relationship follows Michaelis-Menten kinetics with cofactor activation
#
# Reference: Raju et al. (2000); Sha et al. (2016)
print("\n--- Feature 1: Fucosylation Potential Index (FPI) ---")
print("  Biology: FUT8 activity depends on GDP-Fucose (substrate) and Mn²⁺ (cofactor)")
print("  Model: FPI = GDP_Fucose × log(1 + Mn/Km) — saturating cofactor effect")

Km_Mn = 0.01  # Approximate Km for Mn²⁺ activation (µM)
features_eng["FPI"] = X_raw["GDP_Fucose_uM"] * np.log1p(X_raw["Mn_uM"] / Km_Mn)
print(f"  Range: [{features_eng['FPI'].min():.4f}, {features_eng['FPI'].max():.4f}]")

# --- Feature 2: Energy Charge ---
# Biological Mechanism:
#   Glycosylation in the Golgi requires energy (GTP/ATP) for:
#   - Nucleotide sugar transport into Golgi lumen
#   - Maintenance of Golgi pH gradient
#   - Protein trafficking
#   Energy charge approximated from metabolic indicators:
#   - High DO → oxidative phosphorylation → high ATP
#   - High lactate → glycolytic shift → energy stress
#   - pCO₂ affects intracellular pH → disrupts proton motive force
#
# Reference: Jedrzejewski et al. (2014); Hossler et al. (2009)
print("\n--- Feature 2: Energy Charge Proxy ---")
print("  Biology: Golgi glycosylation requires ATP/GTP. High DO promotes oxidative")
print("  phosphorylation; high lactate/pCO₂ indicate metabolic stress.")
print("  Model: Energy = DO / (1 + Lactate) / (1 + pCO2/100)")

features_eng["Energy_Charge"] = (
    X_raw["DO_pct"] /
    (1 + X_raw["Lactate_g_per_L"]) /
    (1 + X_raw["pCO2_mmHg"] / 100)
)
print(f"  Range: [{features_eng['Energy_Charge'].min():.4f}, {features_eng['Energy_Charge'].max():.4f}]")

# --- Feature 3: Golgi Processing Proxy ---
# Biological Mechanism:
#   The Golgi apparatus pH gradient is critical for glycosyltransferase activity.
#   - Optimal Golgi lumen pH ~ 6.0-6.5
#   - Culture pH affects intracellular pH homeostasis
#   - Uridine provides UDP-sugars (nucleotide sugar donors in Golgi)
#   - Temperature affects enzyme kinetics (Arrhenius) and membrane fluidity
#
# Reference: Sha et al. (2016); Von Stosch et al. (2014)
print("\n--- Feature 3: Golgi Processing Proxy ---")
print("  Biology: Golgi glycosylation depends on pH optimality, nucleotide sugar")
print("  availability (uridine), and temperature (enzyme kinetics).")
print("  Model: Golgi = Uridine × exp(-|pH-7.0|²/0.32) × exp(-|T-36.5|²/12.5)")

# Gaussian penalty for pH deviation from optimum
ph_penalty = np.exp(-((X_raw["pH"] - 7.0) ** 2) / (2 * 0.4 ** 2))
# Gaussian penalty for temperature deviation
temp_penalty = np.exp(-((X_raw["Temperature_C"] - 36.5) ** 2) / (2 * 2.5 ** 2))

features_eng["Golgi_Proxy"] = X_raw["Uridine_mM"] * ph_penalty * temp_penalty
print(f"  Range: [{features_eng['Golgi_Proxy'].min():.4f}, {features_eng['Golgi_Proxy'].max():.4f}]")

# --- Feature 4: Metabolic Stress Index ---
# Biological Mechanism:
#   Multiple stressors compromise glycosylation fidelity:
#   - High osmolality → cellular shrinkage → ER/Golgi disruption
#   - High lactate → metabolic acidosis → intracellular pH drop
#   - High pCO₂ → carbonate buffering stress → pH instability
#   The combined effect is synergistic, not additive.
#
# Reference: Hossler et al. (2009)
print("\n--- Feature 4: Metabolic Stress Index ---")
print("  Biology: Osmolality, lactate, and pCO₂ synergistically compromise")
print("  glycosylation by disrupting ER/Golgi homeostasis.")
print("  Model: Stress = (Osm/400 + Lactate/4 + pCO2/120)² — synergistic penalty")

osm_norm = X_raw["Osmolality_mOsm"] / 400
lac_norm = X_raw["Lactate_g_per_L"] / 4.0
pco2_norm = X_raw["pCO2_mmHg"] / 120

features_eng["Stress_Index"] = (osm_norm + lac_norm + pco2_norm) ** 2
print(f"  Range: [{features_eng['Stress_Index'].min():.4f}, {features_eng['Stress_Index'].max():.4f}]")

# --- Feature 5: Enzyme Activity Factor ---
# Biological Mechanism:
#   Glycosyltransferase (FUT8, GnTII, etc.) activity depends on:
#   - pH near enzyme optimum (~6.5-7.0 for FUT8)
#   - Temperature near physiological optimum
#   - Mn²⁺ availability (cofactor)
#   Combined as a pseudo-Arrhenius × pH-bell-curve model.
#
# Reference: Raju et al. (2000)
print("\n--- Feature 5: Enzyme Activity Factor ---")
print("  Biology: FUT8 and related glycosyltransferases have pH and temperature")
print("  optima. Combined pseudo-Arrhenius × pH-bell-curve model.")
print("  Model: Activity = Mn × pH_bell × Temp_bell")

features_eng["Enzyme_Activity"] = (
    X_raw["Mn_uM"] * ph_penalty * temp_penalty
)
print(f"  Range: [{features_eng['Enzyme_Activity'].min():.6f}, {features_eng['Enzyme_Activity'].max():.6f}]")

# Summary of engineered features
engineered_feature_names = ["FPI", "Energy_Charge", "Golgi_Proxy", "Stress_Index", "Enzyme_Activity"]
print(f"\n  Total features: {features_eng.shape[1]} ({len(feature_cols)} raw + {len(engineered_feature_names)} engineered)")

# ============================================================
# 3. Model Comparison: Raw vs Engineered vs Hybrid
# ============================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON: Raw vs Engineered vs Hybrid Features")
print("=" * 70)

# Split data — use consistent split for all variants
X_hybrid = features_eng.values
X_eng_only = features_eng[engineered_feature_names].values

# FIX: Use same random_state and do splits properly
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw.values, y, test_size=0.2, random_state=42
)

# Split hybrid and engineered using same indices
X_train_hyb, X_test_hyb, _, _ = train_test_split(
    X_hybrid, y, test_size=0.2, random_state=42
)
X_train_eng, X_test_eng, _, _ = train_test_split(
    X_eng_only, y, test_size=0.2, random_state=42
)

# Standardize
scaler_raw = StandardScaler().fit(X_train_raw)
scaler_hyb = StandardScaler().fit(X_train_hyb)
scaler_eng = StandardScaler().fit(X_train_eng)

all_feature_names = list(features_eng.columns)

configs = {
    "Raw Features Only": (scaler_raw.transform(X_train_raw), scaler_raw.transform(X_test_raw), feature_cols),
    "Engineered Only": (scaler_eng.transform(X_train_eng), scaler_eng.transform(X_test_eng), engineered_feature_names),
    "Hybrid (Raw + Engineered)": (scaler_hyb.transform(X_train_hyb), scaler_hyb.transform(X_test_hyb), all_feature_names),
}

comparison_results = {}

for name, (Xtr, Xte, feat_names) in configs.items():
    print(f"\n--- {name} ({len(feat_names)} features) ---")

    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    gb.fit(Xtr, y_train)
    y_pred = gb.predict(Xte)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(gb, Xtr, y_train, cv=5, scoring="r2")

    comparison_results[name] = {
        "R²_test": r2,
        "RMSE_test": rmse,
        "MAE_test": mae,
        "CV_R²": cv_scores.mean(),
        "CV_R²_std": cv_scores.std(),
        "n_features": len(feat_names),
        "model": gb,
        "feature_names": feat_names,
    }

    print(f"  R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    imp = pd.DataFrame({
        "Feature": feat_names,
        "Importance": gb.feature_importances_
    }).sort_values("Importance", ascending=False)
    print(f"  Top 5 features:")
    for _, row in imp.head(5).iterrows():
        print(f"    {row['Feature']:25s}: {row['Importance']:.4f}")

# ============================================================
# 4. Quantify Physics Contribution
# ============================================================
print("\n" + "=" * 70)
print("PHYSICS CONTRIBUTION ANALYSIS")
print("=" * 70)

raw_r2 = comparison_results["Raw Features Only"]["R²_test"]
hybrid_r2 = comparison_results["Hybrid (Raw + Engineered)"]["R²_test"]
eng_r2 = comparison_results["Engineered Only"]["R²_test"]

improvement = (hybrid_r2 - raw_r2) / raw_r2 * 100
print(f"  Raw features R²:     {raw_r2:.4f}")
print(f"  Engineered only R²:  {eng_r2:.4f}")
print(f"  Hybrid R²:           {hybrid_r2:.4f}")
print(f"  Improvement from physics: {improvement:+.2f}%")
print(f"\n  Interpretation: Engineered features alone capture {eng_r2 / hybrid_r2 * 100:.1f}%")
print(f"  of the hybrid model's performance using only {len(engineered_feature_names)} features")
print(f"  vs {len(feature_cols)} raw features — demonstrating the value of domain knowledge.")

# ============================================================
# 5. Biological Mechanism Summary Table
# ============================================================
print("\n" + "=" * 70)
print("BIOLOGICAL MECHANISM SUMMARY")
print("=" * 70)

mechanism_table = pd.DataFrame([
    {
        "Feature": "FPI",
        "Full_Name": "Fucosylation Potential Index",
        "Components": "GDP-Fucose, Mn²⁺",
        "Mechanism": "FUT8 substrate × cofactor activation (Michaelis-Menten)",
        "Reference": "Raju et al. 2000; Sha et al. 2016",
        "Expected_Effect": "Strong positive — directly governs fucose transfer rate"
    },
    {
        "Feature": "Energy_Charge",
        "Full_Name": "Cellular Energy Proxy",
        "Components": "DO, Lactate, pCO₂",
        "Mechanism": "Oxidative phosphorylation vs glycolytic stress balance",
        "Reference": "Jedrzejewski et al. 2014; Hossler et al. 2009",
        "Expected_Effect": "Positive — Golgi function requires ATP/GTP"
    },
    {
        "Feature": "Golgi_Proxy",
        "Full_Name": "Golgi Processing Capacity",
        "Components": "Uridine, pH, Temperature",
        "Mechanism": "Nucleotide sugar supply × pH/temp optima for Golgi enzymes",
        "Reference": "Sha et al. 2016; Von Stosch et al. 2014",
        "Expected_Effect": "Positive — optimal Golgi conditions enhance glycosylation"
    },
    {
        "Feature": "Stress_Index",
        "Full_Name": "Metabolic Stress Index",
        "Components": "Osmolality, Lactate, pCO₂",
        "Mechanism": "Synergistic cellular stress disrupting ER/Golgi homeostasis",
        "Reference": "Hossler et al. 2009",
        "Expected_Effect": "Negative — stress compromises glycosylation fidelity"
    },
    {
        "Feature": "Enzyme_Activity",
        "Full_Name": "Glycosyltransferase Activity Factor",
        "Components": "Mn²⁺, pH, Temperature",
        "Mechanism": "Pseudo-Arrhenius × pH-bell-curve for FUT8 activity",
        "Reference": "Raju et al. 2000",
        "Expected_Effect": "Positive — higher enzyme activity → more fucosylation"
    },
])

for _, row in mechanism_table.iterrows():
    print(f"\n  {row['Feature']} — {row['Full_Name']}")
    print(f"    Components: {row['Components']}")
    print(f"    Mechanism: {row['Mechanism']}")
    print(f"    Expected: {row['Expected_Effect']}")
    print(f"    Reference: {row['Reference']}")

mechanism_table.to_csv("results/06_hybrid/mechanism_table.csv", index=False)

# ============================================================
# 6. Visualization
# ============================================================
print("\n" + "-" * 50)
print("Generating plots...")
print("-" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Hybrid Physics-Informed Model — Fucosylation", fontsize=14, fontweight="bold")

# 6a. Model comparison bar chart
ax = axes[0, 0]
names = list(comparison_results.keys())
r2s = [comparison_results[n]["R²_test"] for n in names]
colors = ["#90a4ae", "#43a047", "#1976d2"]
bars = ax.bar(range(len(names)), r2s, color=colors)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(["Raw", "Engineered\nOnly", "Hybrid"], fontsize=9)
ax.set_ylabel("R² (Test)")
ax.set_title("Feature Set Comparison")
for i, v in enumerate(r2s):
    ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

# 6b. Hybrid model actual vs predicted
ax = axes[0, 1]
y_pred_hyb = comparison_results["Hybrid (Raw + Engineered)"]["model"].predict(
    scaler_hyb.transform(X_test_hyb)
)
ax.scatter(y_test, y_pred_hyb, alpha=0.5, s=20, color="steelblue")
lims = [min(y_test.min(), y_pred_hyb.min()) - 1, max(y_test.max(), y_pred_hyb.max()) + 1]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual Fucosylation (%)")
ax.set_ylabel("Predicted Fucosylation (%)")
ax.set_title(f"Hybrid Model (R²={hybrid_r2:.4f})")

# 6c. Feature importance (hybrid model)
ax = axes[0, 2]
hyb_model = comparison_results["Hybrid (Raw + Engineered)"]["model"]
hyb_feat_names = comparison_results["Hybrid (Raw + Engineered)"]["feature_names"]
imp_df = pd.DataFrame({
    "Feature": hyb_feat_names,
    "Importance": hyb_model.feature_importances_
}).sort_values("Importance", ascending=True)
colors_imp = ["#d32f2f" if f in engineered_feature_names else "#90a4ae"
              for f in imp_df["Feature"]]
ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors_imp)
ax.set_xlabel("Feature Importance")
ax.set_title("Feature Importance\n(red = engineered, gray = raw)")

# 6d. FPI vs Fucosylation
ax = axes[1, 0]
ax.scatter(features_eng["FPI"], y, alpha=0.3, s=10, color="steelblue")
ax.set_xlabel("FPI (Fucosylation Potential Index)")
ax.set_ylabel("Fucosylation (%)")
ax.set_title("FPI vs Fucosylation\n(substrate × cofactor)")

# 6e. Stress Index vs Fucosylation
ax = axes[1, 1]
ax.scatter(features_eng["Stress_Index"], y, alpha=0.3, s=10, color="#d32f2f")
ax.set_xlabel("Metabolic Stress Index")
ax.set_ylabel("Fucosylation (%)")
ax.set_title("Stress Index vs Fucosylation\n(↑ stress → ↓ fucosylation)")

# 6f. Engineered feature correlations (FIX: build correlation DataFrame properly)
ax = axes[1, 2]
corr_data = features_eng[engineered_feature_names].copy()
corr_data["Fucosylation_pct"] = y  # Add target as a new column
eng_corr = corr_data.corr()

im = ax.imshow(eng_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(eng_corr.columns)))
ax.set_xticklabels(eng_corr.columns, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(eng_corr.columns)))
ax.set_yticklabels(eng_corr.columns, fontsize=7)
# Add correlation values as text
for i in range(len(eng_corr)):
    for j in range(len(eng_corr)):
        ax.text(j, i, f"{eng_corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("Engineered Feature Correlations")

plt.tight_layout()
plt.savefig("results/06_hybrid/hybrid_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ All plots saved")

# ============================================================
# 7. Save Results
# ============================================================
comp_df = pd.DataFrame([
    {"Model": k, **{kk: vv for kk, vv in v.items() if kk not in ["model", "feature_names"]}}
    for k, v in comparison_results.items()
])
comp_df.to_csv("results/06_hybrid/model_comparison.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Raw features R²:    {raw_r2:.4f}")
print(f"Engineered only R²: {eng_r2:.4f}")
print(f"Hybrid R²:          {hybrid_r2:.4f}")
print(f"Physics improvement: {improvement:+.2f}%")
print("\n✅ Hybrid model analysis complete. Results saved to results/06_hybrid/")
