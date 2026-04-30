"""
00_dataset_generator.py
========================
Synthetic Dataset Generator for Monoclonal Antibody (mAb) Fucosylation Prediction

This script generates a biologically grounded synthetic dataset simulating CHO cell
bioprocessing conditions and their effect on mAb fucosylation levels.

Literature References:
- Jedrzejewski et al. (2014): Mechanistic modeling of CHO cell metabolism
- Raju et al. (2000): GDP-fucose pathway and fucosylation control
- Hossler et al. (2009): Cell culture media optimization for glycosylation

Ground truth relationships modeled:
- GDP-Fucose: Primary substrate for fucosyltransferase (FUT8) — strong positive effect
- Manganese (Mn²⁺): Cofactor for glycosyltransferases — moderate positive effect
- Uridine: Nucleotide sugar precursor — mild positive effect
- pH: Optimal near 7.0, quadratic penalty at extremes (enzyme denaturation)
- Dissolved Oxygen (DO): Moderate positive (supports metabolic flux)
- pCO₂: Negative effect at high levels (intracellular pH disruption)
- Temperature: Optimal near 36.5°C, quadratic penalty at extremes
- Viable Cell Density (VCD): Weak positive (more cells → more product)
- Osmolality: Mild negative at high levels (cellular stress)
- Lactate: Negative (metabolic byproduct indicating inefficiency)

Interaction terms:
- GDP-Fucose × Mn²⁺ synergy (cofactor enhances enzyme utilizing substrate)
"""

import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
SEED = 42
N_SAMPLES = 500
N_BATCHES = 10  # For batch effects
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)

# ============================================================
# 1. Generate Base Process Parameters (biologically plausible ranges)
# ============================================================
print("=" * 70)
print("STEP 1: Generating base process parameters")
print("=" * 70)

data = pd.DataFrame({
    # GDP-Fucose concentration (µM) — primary driver
    "GDP_Fucose_uM": np.random.uniform(5, 80, N_SAMPLES),

    # Manganese concentration (µM) — glycosyltransferase cofactor
    "Mn_uM": np.random.uniform(0.001, 0.05, N_SAMPLES),

    # Uridine concentration (mM) — nucleotide sugar precursor
    "Uridine_mM": np.random.uniform(0.5, 5.0, N_SAMPLES),

    # Culture pH — tightly controlled in bioreactors
    "pH": np.random.uniform(6.6, 7.4, N_SAMPLES),

    # Dissolved oxygen (% air saturation)
    "DO_pct": np.random.uniform(20, 80, N_SAMPLES),

    # Dissolved CO₂ (mmHg)
    "pCO2_mmHg": np.random.uniform(30, 120, N_SAMPLES),

    # Temperature (°C) — typically 36–37°C with some variation
    "Temperature_C": np.random.uniform(33, 38, N_SAMPLES),

    # Viable cell density (×10⁶ cells/mL)
    "VCD_1e6_per_mL": np.random.uniform(2, 25, N_SAMPLES),

    # Osmolality (mOsm/kg)
    "Osmolality_mOsm": np.random.uniform(280, 400, N_SAMPLES),

    # Lactate concentration (g/L)
    "Lactate_g_per_L": np.random.uniform(0.1, 4.0, N_SAMPLES),
})

print(f"Generated {N_SAMPLES} samples with {data.shape[1]} features")
print(f"Feature ranges:")
for col in data.columns:
    print(f"  {col}: [{data[col].min():.4f}, {data[col].max():.4f}]")

# ============================================================
# 2. Define Ground Truth Fucosylation Model
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Computing ground truth fucosylation via mechanistic model")
print("=" * 70)


def compute_fucosylation(df):
    """
    Compute fucosylation percentage using biologically grounded relationships.

    The model captures:
    - Michaelis-Menten-like saturation for GDP-Fucose
    - Linear cofactor/precursor effects
    - Quadratic pH and temperature optima
    - Interaction between GDP-Fucose and Mn²⁺
    - Negative effects of metabolic stress indicators
    """
    # Normalize inputs to [0, 1] for coefficient interpretability
    gdp_fuc_norm = (df["GDP_Fucose_uM"] - 5) / (80 - 5)
    mn_norm = (df["Mn_uM"] - 0.001) / (0.05 - 0.001)
    uridine_norm = (df["Uridine_mM"] - 0.5) / (5.0 - 0.5)
    do_norm = (df["DO_pct"] - 20) / (80 - 20)
    pco2_norm = (df["pCO2_mmHg"] - 30) / (120 - 30)
    vcd_norm = (df["VCD_1e6_per_mL"] - 2) / (25 - 2)
    osm_norm = (df["Osmolality_mOsm"] - 280) / (400 - 280)
    lactate_norm = (df["Lactate_g_per_L"] - 0.1) / (4.0 - 0.1)
    temp_norm = (df["Temperature_C"] - 33) / (38 - 33)

    # pH deviation from optimum (7.0)
    ph_dev = (df["pH"] - 7.0) / 0.4  # normalized deviation

    # Temperature deviation from optimum (36.5°C)
    temp_dev = (df["Temperature_C"] - 36.5) / 2.5

    # --- Ground truth equation ---
    # Intercept (baseline fucosylation ~85% under ideal conditions)
    y = 85.0

    # Primary substrate effect (saturating / Michaelis-Menten-like)
    y += 12.0 * (gdp_fuc_norm / (gdp_fuc_norm + 0.3))

    # Cofactor and precursor effects (linear)
    y += 4.0 * mn_norm       # Mn²⁺ cofactor
    y += 2.0 * uridine_norm  # Uridine precursor

    # Quadratic pH effect (optimum at 7.0)
    y -= 8.0 * (ph_dev ** 2)

    # Quadratic temperature effect (optimum at 36.5°C)
    y -= 5.0 * (temp_dev ** 2)

    # Dissolved oxygen (moderate positive)
    y += 3.0 * do_norm

    # pCO₂ (negative — disrupts intracellular pH)
    y -= 4.0 * pco2_norm

    # VCD (weak positive)
    y += 1.5 * vcd_norm

    # Osmolality (mild negative at high levels — cellular stress)
    y -= 2.5 * osm_norm

    # Lactate (negative — metabolic inefficiency marker)
    y -= 3.0 * lactate_norm

    # Interaction: GDP-Fucose × Mn²⁺ synergy
    y += 3.5 * gdp_fuc_norm * mn_norm

    return y


fucosylation_clean = compute_fucosylation(data)
print(f"Clean fucosylation range: [{fucosylation_clean.min():.2f}%, {fucosylation_clean.max():.2f}%]")

# ============================================================
# 3. Add Batch Effects (Random Intercept Model)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Adding batch effects (random intercept per batch)")
print("=" * 70)

batch_ids = np.random.randint(0, N_BATCHES, N_SAMPLES)
# Each batch has a random shift (simulating equipment/operator variation)
batch_effects = np.random.normal(0, 1.5, N_BATCHES)  # ±1.5% shift per batch
batch_shift = np.array([batch_effects[b] for b in batch_ids])

data["Batch_ID"] = batch_ids
print(f"Batch effects (% shift): {dict(enumerate(np.round(batch_effects, 2)))}")

# ============================================================
# 4. Add Realistic Measurement Noise
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Injecting measurement noise")
print("=" * 70)

# Heteroscedastic noise: more noise at extreme fucosylation values
base_noise_std = 1.8
noise = np.random.normal(0, base_noise_std, N_SAMPLES)

# Add heteroscedastic component (higher noise at extremes)
deviation_from_mean = np.abs(fucosylation_clean - fucosylation_clean.mean())
heteroscedastic_noise = 0.02 * deviation_from_mean * np.random.normal(0, 1, N_SAMPLES)

total_noise = noise + heteroscedastic_noise
print(f"Base noise std: {base_noise_std}")
print(f"Total noise std: {total_noise.std():.3f}")

# Combine: clean signal + batch effects + noise
data["Fucosylation_pct"] = np.clip(
    fucosylation_clean + batch_shift + total_noise,
    0, 100  # Physical bounds
)

print(f"Final fucosylation range: [{data['Fucosylation_pct'].min():.2f}%, {data['Fucosylation_pct'].max():.2f}%]")
print(f"Mean: {data['Fucosylation_pct'].mean():.2f}%, Std: {data['Fucosylation_pct'].std():.2f}%")

# ============================================================
# 5. Generate Missing Data Scenarios
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Creating missing data variants")
print("=" * 70)

# --- 5a. Missing Completely at Random (MCAR) ---
data_mcar = data.copy()
mcar_rate = 0.05  # 5% missing
features_for_missing = [c for c in data.columns if c not in ["Fucosylation_pct", "Batch_ID"]]
mask_mcar = np.random.random((N_SAMPLES, len(features_for_missing))) < mcar_rate
for i, col in enumerate(features_for_missing):
    data_mcar.loc[mask_mcar[:, i], col] = np.nan

n_missing_mcar = data_mcar[features_for_missing].isna().sum().sum()
print(f"MCAR: {n_missing_mcar} values missing ({n_missing_mcar / (N_SAMPLES * len(features_for_missing)) * 100:.1f}%)")

# --- 5b. Missing Not at Random (MNAR) — sensor failure at extremes ---
data_mnar = data.copy()
# DO sensor fails when DO > 70% (common real-world issue)
do_high_mask = data_mnar["DO_pct"] > 70
do_fail_mask = do_high_mask & (np.random.random(N_SAMPLES) < 0.3)  # 30% failure rate when high
data_mnar.loc[do_fail_mask, "DO_pct"] = np.nan
# pCO₂ sensor drift at high values
pco2_high_mask = data_mnar["pCO2_mmHg"] > 100
pco2_fail_mask = pco2_high_mask & (np.random.random(N_SAMPLES) < 0.25)
data_mnar.loc[pco2_fail_mask, "pCO2_mmHg"] = np.nan

n_missing_mnar = data_mnar[features_for_missing].isna().sum().sum()
print(f"MNAR: {n_missing_mnar} values missing (sensor-failure pattern)")

# ============================================================
# 6. Generate Noise Corruption Scenarios
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Creating noise-corrupted variants for sensitivity analysis")
print("=" * 70)

noise_levels = {"low": 0.5, "medium": 1.0, "high": 2.0, "extreme": 4.0}
noise_datasets = {}

for level_name, noise_multiplier in noise_levels.items():
    data_noisy = data.copy()
    extra_noise = np.random.normal(0, noise_multiplier * base_noise_std, N_SAMPLES)
    data_noisy["Fucosylation_pct"] = np.clip(
        data_noisy["Fucosylation_pct"] + extra_noise, 0, 100
    )
    noise_datasets[level_name] = data_noisy
    print(f"  {level_name} noise (σ={noise_multiplier * base_noise_std:.1f}): "
          f"target std = {data_noisy['Fucosylation_pct'].std():.2f}%")

# ============================================================
# 7. Add Time-Series / Culture Day Dimension
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Generating time-series variant (culture day progression)")
print("=" * 70)

N_BATCHES_TS = 20
DAYS_PER_BATCH = 14
N_TS = N_BATCHES_TS * DAYS_PER_BATCH

ts_records = []
for batch in range(N_BATCHES_TS):
    # Each batch starts with slightly different conditions
    base_gdp_fuc = np.random.uniform(30, 60)
    base_mn = np.random.uniform(0.01, 0.04)
    base_ph = np.random.uniform(6.8, 7.2)

    for day in range(DAYS_PER_BATCH):
        t = day / DAYS_PER_BATCH  # normalized time [0, 1]

        record = {
            "Batch_ID": batch,
            "Culture_Day": day + 1,
            # GDP-Fucose depletes over culture (substrate consumption)
            "GDP_Fucose_uM": base_gdp_fuc * np.exp(-0.8 * t) + np.random.normal(0, 2),
            # Mn²⁺ slowly consumed
            "Mn_uM": base_mn * (1 - 0.3 * t) + np.random.normal(0, 0.002),
            "Uridine_mM": np.random.uniform(1, 4) * (1 - 0.2 * t),
            # pH drifts downward (lactate accumulation)
            "pH": base_ph - 0.15 * t + np.random.normal(0, 0.03),
            "DO_pct": np.random.uniform(30, 60),
            "pCO2_mmHg": 40 + 30 * t + np.random.normal(0, 5),  # CO₂ accumulates
            "Temperature_C": 36.5 + np.random.normal(0, 0.3),
            # VCD follows logistic growth
            "VCD_1e6_per_mL": 20 / (1 + 15 * np.exp(-0.5 * (day - 5))) + np.random.normal(0, 0.5),
            "Osmolality_mOsm": 300 + 40 * t + np.random.normal(0, 5),
            # Lactate accumulates then partially consumed
            "Lactate_g_per_L": 3.0 * t * np.exp(-0.3 * t) * 5 + np.random.normal(0, 0.1),
        }
        ts_records.append(record)

data_ts = pd.DataFrame(ts_records)

# Clip to plausible ranges
data_ts["GDP_Fucose_uM"] = data_ts["GDP_Fucose_uM"].clip(1, 80)
data_ts["Mn_uM"] = data_ts["Mn_uM"].clip(0.001, 0.05)
data_ts["Uridine_mM"] = data_ts["Uridine_mM"].clip(0.1, 5.0)
data_ts["pH"] = data_ts["pH"].clip(6.5, 7.5)
data_ts["VCD_1e6_per_mL"] = data_ts["VCD_1e6_per_mL"].clip(0.5, 25)
data_ts["Lactate_g_per_L"] = data_ts["Lactate_g_per_L"].clip(0.05, 5.0)

# Compute fucosylation for time-series data
data_ts["Fucosylation_pct"] = np.clip(
    compute_fucosylation(data_ts) + np.random.normal(0, 1.5, N_TS), 0, 100
)

print(f"Time-series dataset: {data_ts.shape[0]} observations ({N_BATCHES_TS} batches × {DAYS_PER_BATCH} days)")
print(f"Fucosylation range: [{data_ts['Fucosylation_pct'].min():.2f}%, {data_ts['Fucosylation_pct'].max():.2f}%]")

# ============================================================
# 8. Save All Datasets
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Saving datasets")
print("=" * 70)

# Primary dataset (with batch effects + noise)
data.to_csv(os.path.join(OUTPUT_DIR, "mab_fucosylation_dataset.csv"), index=False)
print(f"  ✓ Primary dataset: {data.shape}")

# Missing data variants
data_mcar.to_csv(os.path.join(OUTPUT_DIR, "mab_fucosylation_MCAR.csv"), index=False)
print(f"  ✓ MCAR dataset: {data_mcar.shape}")

data_mnar.to_csv(os.path.join(OUTPUT_DIR, "mab_fucosylation_MNAR.csv"), index=False)
print(f"  ✓ MNAR dataset: {data_mnar.shape}")

# Noise variants
for level_name, df_noisy in noise_datasets.items():
    fname = f"mab_fucosylation_noise_{level_name}.csv"
    df_noisy.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
    print(f"  ✓ Noise ({level_name}): {df_noisy.shape}")

# Time-series variant
data_ts.to_csv(os.path.join(OUTPUT_DIR, "mab_fucosylation_timeseries.csv"), index=False)
print(f"  ✓ Time-series dataset: {data_ts.shape}")

# Ground truth parameters (for validation)
ground_truth = {
    "intercept": 85.0,
    "GDP_Fucose_saturation_coeff": 12.0,
    "Mn_coeff": 4.0,
    "Uridine_coeff": 2.0,
    "pH_quadratic_penalty": -8.0,
    "Temperature_quadratic_penalty": -5.0,
    "DO_coeff": 3.0,
    "pCO2_coeff": -4.0,
    "VCD_coeff": 1.5,
    "Osmolality_coeff": -2.5,
    "Lactate_coeff": -3.0,
    "GDPFuc_x_Mn_interaction": 3.5,
    "batch_effect_std": 1.5,
    "measurement_noise_std": base_noise_std,
}
pd.DataFrame([ground_truth]).to_csv(os.path.join(OUTPUT_DIR, "ground_truth_params.csv"), index=False)
print(f"  ✓ Ground truth parameters saved")

# ============================================================
# 9. Summary Statistics
# ============================================================
print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)
print(f"\nPrimary dataset shape: {data.shape}")
print(f"\nDescriptive statistics:")
print(data.describe().round(3).to_string())
print(f"\nCorrelation with target (Fucosylation_pct):")
feature_cols = [c for c in data.columns if c not in ["Fucosylation_pct", "Batch_ID"]]
correlations = data[feature_cols].corrwith(data["Fucosylation_pct"]).sort_values(ascending=False)
for feat, corr in correlations.items():
    print(f"  {feat:25s}: {corr:+.4f}")

print("\n✅ All datasets generated successfully.")
print(f"   Files saved to: {OUTPUT_DIR}/")
