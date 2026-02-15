#!/usr/bin/env python3
"""
Evaluation script - loads all trained models and generates CSV outputs
Uses same stratified train/val/test split as training
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

print("Loading data...")
items = pd.read_csv("data/hnm/processed/articles_features.csv", dtype={"article_id": str})
users = pd.read_csv("data/hnm/processed/customers_features.csv", dtype={"customer_id": str})
tx = pd.read_csv("data/hnm/processed/transactions_sample.csv",
                 dtype={"customer_id": str, "article_id": str})

# Recreate the same stratified train/val/test split
purchase_counts = tx.groupby(["customer_id", "article_id"]).size().reset_index(name="purchase_count")
data = purchase_counts.merge(items, on="article_id", how="left")
data = data.merge(users, on="customer_id", how="left")

id_cols = ["customer_id", "article_id", "product_code"]
cols_to_drop = ["purchase_count"] + [col for col in id_cols if col in data.columns]

X = data.drop(columns=cols_to_drop)
y = data["purchase_count"]

# Keep the IDs for output
ids_df = data[["customer_id", "article_id"]]

# Stratified 3-way split matching training script
strata = pd.cut(y, bins=[-np.inf, 1, 2, 5, np.inf], labels=[0, 1, 2, 3])

X_train, X_temp, y_train, y_temp, ids_train, ids_temp, strata_train, strata_temp = train_test_split(
    X, y, ids_df, strata, test_size=0.4, random_state=42, stratify=strata
)
X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
    X_temp, y_temp, ids_temp, test_size=0.5, random_state=42, stratify=strata_temp
)

print(f"Train set size:      {len(X_train):,} samples")
print(f"Validation set size: {len(X_val):,} samples")
print(f"Test set size:       {len(X_test):,} samples")

# Find all model files
model_dir = Path("data/hnm/models")
model_files = list(model_dir.glob("*.pkl"))

if not model_files:
    print("\n❌ No model files found in data/hnm/models/")
    print("Please train models first using run_training.py or run_training_fast.py")
    exit(1)

print(f"\nFound {len(model_files)} model(s):")
for mf in model_files:
    print(f"  - {mf.name}")

# Create output directories
output_dir = Path("data/hnm/evaluation")
predictions_dir = output_dir / "predictions"
output_dir.mkdir(exist_ok=True)
predictions_dir.mkdir(exist_ok=True)

# Store all metrics
all_metrics = []

print("\n" + "="*60)
print("EVALUATING MODELS")
print("="*60)

# Evaluate each model
for model_file in sorted(model_files):
    model_name = model_file.stem  # filename without .pkl extension

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    # Load model
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Make predictions on all sets
    print("Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Calculate metrics for all sets
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)

    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_r2 = r2_score(y_val, y_pred_val)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    try:
        test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
    except:
        test_mape = np.nan

    print(f"\n  {'Set':<12} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print(f"  {'-'*42}")
    print(f"  {'Train':<12} {train_rmse:<10.4f} {train_mae:<10.4f} {train_r2:<10.4f}")
    print(f"  {'Validation':<12} {val_rmse:<10.4f} {val_mae:<10.4f} {val_r2:<10.4f}")
    print(f"  {'Test':<12} {test_rmse:<10.4f} {test_mae:<10.4f} {test_r2:<10.4f}")

    overfit_gap = train_rmse - val_rmse
    if abs(overfit_gap) > 0.1:
        print(f"  ** Overfitting detected: train-val RMSE gap = {overfit_gap:.4f}")

    # Save predictions to CSV (test set)
    predictions_df = ids_test.copy()
    predictions_df["actual"] = y_test.values
    predictions_df["predicted"] = y_pred_test
    predictions_df["error"] = y_test.values - y_pred_test
    predictions_df["abs_error"] = np.abs(predictions_df["error"])

    predictions_file = predictions_dir / f"{model_name}_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"\n✓ Predictions saved → {predictions_file}")

    # Store metrics
    all_metrics.append({
        "model": model_name,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_r2": val_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "test_mape": test_mape,
        "overfitting_gap": train_rmse - val_rmse,
    })

# Save metrics summary to CSV
print("\n" + "="*60)
print("SAVING METRICS SUMMARY")
print("="*60)

metrics_df = pd.DataFrame(all_metrics)
metrics_df = metrics_df.sort_values("val_rmse")  # Sort by best validation RMSE

metrics_file = output_dir / "model_metrics_summary.csv"
metrics_df.to_csv(metrics_file, index=False)
print(f"\n✓ Metrics summary saved → {metrics_file}")

# Print comparison table
print("\n" + "="*60)
print("MODEL COMPARISON (sorted by Validation RMSE)")
print("="*60)
print(f"\n{'Model':<30} {'Train RMSE':<12} {'Val RMSE':<12} {'Test RMSE':<12} {'Val R²':<10}")
print("-" * 80)
for _, row in metrics_df.iterrows():
    print(f"{row['model']:<30} {row['train_rmse']:<12.4f} {row['val_rmse']:<12.4f} {row['test_rmse']:<12.4f} {row['val_r2']:<10.4f}")

# Identify best model by validation RMSE
best_model = metrics_df.iloc[0]
print(f"\nBest Model (by Validation RMSE): {best_model['model']}")
print(f"   Val RMSE:  {best_model['val_rmse']:.4f}")
print(f"   Test RMSE: {best_model['test_rmse']:.4f}")
print(f"   Val R²:    {best_model['val_r2']:.4f}")

print("\n" + "="*60)
print("OUTPUTS GENERATED:")
print("="*60)
print(f"\n1. Metrics Summary CSV:")
print(f"   → {metrics_file}")
print(f"\n2. Individual Prediction CSVs:")
for model_file in sorted(model_files):
    pred_file = predictions_dir / f"{model_file.stem}_predictions.csv"
    print(f"   → {pred_file}")

print("\n✓ EVALUATION COMPLETE!")
