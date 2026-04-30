"""
04_ann_model.py
================
Artificial Neural Network — Deep Learning Model

ANN captures complex nonlinear relationships but historically lacks interpretability.
This implementation adds a LIME-based interpretability layer to address the
"black box" criticism raised in regulatory contexts.

Literature References:
- Goodfellow et al. (2016): Deep Learning
- Ribeiro et al. (2016): LIME — "Why Should I Trust You?"
- ICH Q8/Q9: Quality by Design framework (regulatory context)

Architecture:
- Input → 128 → 64 → 32 → 1 (ReLU activations, batch normalization)
- Dropout for regularization
- Early stopping + learning rate scheduling
- LIME for local interpretability (addresses regulatory concern)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("results/04_ann", exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 70)
print("04 — ARTIFICIAL NEURAL NETWORK (ANN)")
print("=" * 70)

data = pd.read_csv("data/mab_fucosylation_dataset.csv")
feature_cols = [c for c in data.columns if c not in ["Fucosylation_pct", "Batch_ID"]]
X = data[feature_cols].values
y = data["Fucosylation_pct"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Further split train into train/validation for early stopping
X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
    X_train_s, y_train, test_size=0.15, random_state=42
)

print(f"Train: {X_train_nn.shape[0]}, Validation: {X_val_nn.shape[0]}, Test: {X_test_s.shape[0]}")

# ============================================================
# 2. Build PyTorch Model
# ============================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    USE_TORCH = True
    print("Using PyTorch backend")
except ImportError:
    USE_TORCH = False
    print("PyTorch not available — using sklearn MLPRegressor fallback")

if USE_TORCH:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_nn).to(device)
    y_train_t = torch.FloatTensor(y_train_nn).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val_nn).to(device)
    y_val_t = torch.FloatTensor(y_val_nn).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test_s).to(device)

    # DataLoader
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # --- Define Architecture ---
    class FucosylationANN(nn.Module):
        """
        Feedforward ANN for fucosylation prediction.

        Architecture rationale:
        - 128→64→32 tapering captures hierarchical feature interactions
        - BatchNorm stabilizes training with small datasets
        - Dropout (0.2) prevents overfitting
        - ReLU activation for nonlinearity
        """
        def __init__(self, input_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.network(x)

    model = FucosylationANN(X_train_s.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # FIX: Removed 'verbose' parameter — it was removed in PyTorch 2.2+
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ============================================================
    # 3. Training Loop with Early Stopping
    # ============================================================
    print("\n" + "-" * 50)
    print("Training...")
    print("-" * 50)

    EPOCHS = 500
    PATIENCE = 30
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None

    for epoch in range(EPOCHS):
        # Training
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(X_batch)
        epoch_train_loss /= len(train_ds)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # FIX: Use proper deep copy of state_dict
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch + 1:4d}/{EPOCHS}: "
                  f"Train Loss={epoch_train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, LR={lr:.6f}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"  Best validation loss: {best_val_loss:.4f}")

    # ============================================================
    # 4. Evaluation
    # ============================================================
    print("\n" + "-" * 50)
    print("Model Evaluation")
    print("-" * 50)

    model.eval()
    with torch.no_grad():
        y_pred_train_all = model(torch.FloatTensor(X_train_s).to(device)).cpu().numpy().ravel()
        y_pred_test = model(X_test_t).cpu().numpy().ravel()

    # Prediction function for LIME
    def predict_fn(X_input):
        model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X_input).to(device)
            return model(t).cpu().numpy().ravel()

else:
    # Fallback: sklearn MLP with better convergence settings
    from sklearn.neural_network import MLPRegressor

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=1000,           # increased from 500
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,     # increased patience
        random_state=42,
        verbose=False,
    )
    model.fit(X_train_s, y_train)
    y_pred_train_all = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)
    train_losses = model.loss_curve_
    val_losses = model.validation_scores_ if hasattr(model, "validation_scores_") else []

    def predict_fn(X_input):
        return model.predict(X_input)

metrics = {
    "R²_train": r2_score(y_train, y_pred_train_all),
    "R²_test": r2_score(y_test, y_pred_test),
    "RMSE_train": np.sqrt(mean_squared_error(y_train, y_pred_train_all)),
    "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
    "MAE_test": mean_absolute_error(y_test, y_pred_test),
}
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# ============================================================
# 5. LIME Interpretability Layer
# ============================================================
print("\n" + "=" * 70)
print("LIME INTERPRETABILITY (Addressing Black-Box Limitation)")
print("=" * 70)
print("LIME provides local, human-readable explanations for individual predictions.")
print("This is critical for FDA/EMA regulatory discussions where model")
print("transparency is required (ICH Q8/Q9 framework).")

try:
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        X_train_s,
        feature_names=feature_cols,
        mode="regression",
        random_state=42,
    )

    # Explain 5 representative test samples
    n_explain = min(5, len(X_test_s))
    lime_results = []

    print(f"\n  Explaining {n_explain} sample predictions:")
    for i in range(n_explain):
        exp = explainer.explain_instance(
            X_test_s[i], predict_fn, num_features=len(feature_cols)
        )
        feature_weights = dict(exp.as_list())
        print(f"\n  Sample {i + 1}: Predicted={y_pred_test[i]:.2f}%, Actual={y_test[i]:.2f}%")
        for feat, weight in sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            direction = "↑" if weight > 0 else "↓"
            print(f"    {feat}: {weight:+.4f} {direction}")
        lime_results.append({
            "sample_idx": i,
            "predicted": y_pred_test[i],
            "actual": y_test[i],
            **feature_weights
        })

    lime_df = pd.DataFrame(lime_results)
    lime_df.to_csv("results/04_ann/lime_explanations.csv", index=False)
    print("\n  ✓ LIME explanations saved")

    # Aggregate LIME importance (global approximation via local explanations)
    print("\n  Global Feature Importance (aggregated LIME):")
    all_explanations = []
    n_global = min(50, len(X_test_s))
    for i in range(n_global):
        exp = explainer.explain_instance(X_test_s[i], predict_fn, num_features=len(feature_cols))
        all_explanations.append(dict(exp.as_list()))

    global_lime = pd.DataFrame(all_explanations)
    mean_abs_importance = global_lime.abs().mean().sort_values(ascending=False)
    for feat, imp in mean_abs_importance.head(10).items():
        print(f"    {feat}: {imp:.4f}")

    HAS_LIME = True
except ImportError:
    print("  ⚠️ LIME not installed. Run: pip install lime")
    print("  Skipping interpretability analysis.")
    HAS_LIME = False

# ============================================================
# 6. Visualization
# ============================================================
print("\n" + "-" * 50)
print("Generating plots...")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("ANN — Fucosylation Prediction", fontsize=14, fontweight="bold")

# 6a. Training curve
ax = axes[0, 0]
ax.plot(train_losses, label="Train Loss", color="steelblue", alpha=0.7)
if val_losses:
    ax.plot(val_losses, label="Validation Loss", color="#d32f2f", alpha=0.7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE)")
ax.set_title("Training Curve")
ax.legend()
ax.set_yscale("log")

# 6b. Actual vs Predicted
ax = axes[0, 1]
ax.scatter(y_test, y_pred_test, alpha=0.5, s=20, color="steelblue")
lims = [min(y_test.min(), y_pred_test.min()) - 1, max(y_test.max(), y_pred_test.max()) + 1]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual Fucosylation (%)")
ax.set_ylabel("Predicted Fucosylation (%)")
ax.set_title(f"Actual vs Predicted (R²={metrics['R²_test']:.4f})")

# 6c. Residual distribution
ax = axes[1, 0]
residuals = y_test - y_pred_test
ax.hist(residuals, bins=25, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(x=0, color="red", linestyle="--")
ax.set_xlabel("Residual")
ax.set_ylabel("Count")
ax.set_title(f"Residual Distribution (Mean={residuals.mean():.3f}, Std={residuals.std():.3f})")

# 6d. LIME importance (if available)
ax = axes[1, 1]
if HAS_LIME:
    top_features = mean_abs_importance.head(10)
    ax.barh(range(len(top_features)), top_features.values, color="#43a047")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=8)
    ax.set_xlabel("Mean |LIME Weight|")
    ax.set_title("LIME — Global Feature Importance\n(aggregated local explanations)")
else:
    ax.text(0.5, 0.5, "LIME not available\nInstall: pip install lime",
            ha="center", va="center", fontsize=12, color="gray")
    ax.set_title("LIME — Not Available")

plt.tight_layout()
plt.savefig("results/04_ann/ann_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ All plots saved")

# ============================================================
# 7. Save Results
# ============================================================
pd.DataFrame([metrics]).to_csv("results/04_ann/metrics.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Test R²: {metrics['R²_test']:.4f}")
print(f"Test RMSE: {metrics['RMSE_test']:.4f}")
print(f"LIME interpretability: {'Available' if HAS_LIME else 'Not installed'}")
print("\nRegulatory Note: While ANNs achieve high accuracy, their use in")
print("regulated environments requires interpretability tools like LIME")
print("or SHAP to satisfy ICH Q8/Q9 transparency requirements.")
print("\n✅ ANN analysis complete. Results saved to results/04_ann/")
