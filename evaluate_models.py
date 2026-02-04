#!/usr/bin/env python3
"""
Evaluation script - loads all trained models and generates CSV outputs
"""
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

print("Loading data...")
items = pd.read_csv("data/hnm/processed/articles_features.csv", dtype={"article_id": str})
users = pd.read_csv("data/hnm/processed/customers_features.csv", dtype={"customer_id": str})
tx = pd.read_csv("data/hnm/processed/transactions_sample.csv",
                 dtype={"customer_id": str, "article_id": str})

# Recreate the same train/test split
purchase_counts = tx.groupby(["customer_id", "article_id"]).size().reset_index(name="purchase_count")
data = purchase_counts.merge(items, on="article_id", how="left")
data = data.merge(users, on="customer_id", how="left")

id_cols = ["customer_id", "article_id", "product_code"]
cols_to_drop = ["purchase_count"] + [col for col in id_cols if col in data.columns]

X = data.drop(columns=cols_to_drop)
y = data["purchase_count"]

# Keep the IDs for output
ids_df = data[["customer_id", "article_id"]]

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids_df, test_size=0.2, random_state=42
)

print(f"Test set size: {len(X_test):,} samples")

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

    # Make predictions
    print("Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics for both train and test sets
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    # Calculate additional metrics
    try:
        test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
    except:
        test_mape = np.nan

    print(f"\nTraining Set Metrics:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R²:   {train_r2:.4f}")

    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    if not np.isnan(test_mape):
        print(f"  MAPE: {test_mape:.4f}")

    # Save predictions to CSV
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
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "test_mape": test_mape,
        "overfitting": train_rmse - test_rmse,  # negative means worse on test
    })

# Save metrics summary to CSV
print("\n" + "="*60)
print("SAVING METRICS SUMMARY")
print("="*60)

metrics_df = pd.DataFrame(all_metrics)
metrics_df = metrics_df.sort_values("test_rmse")  # Sort by best test RMSE

metrics_file = output_dir / "model_metrics_summary.csv"
metrics_df.to_csv(metrics_file, index=False)
print(f"\n✓ Metrics summary saved → {metrics_file}")

# Print comparison table
print("\n" + "="*60)
print("MODEL COMPARISON (sorted by Test RMSE)")
print("="*60)
print(f"\n{'Model':<30} {'Test RMSE':<12} {'Test MAE':<12} {'Test R²':<12}")
print("-" * 70)
for _, row in metrics_df.iterrows():
    print(f"{row['model']:<30} {row['test_rmse']:<12.4f} {row['test_mae']:<12.4f} {row['test_r2']:<12.4f}")

# Identify best model
best_model = metrics_df.iloc[0]
print(f"\n🏆 Best Model: {best_model['model']}")
print(f"   Test RMSE: {best_model['test_rmse']:.4f}")
print(f"   Test R²:   {best_model['test_r2']:.4f}")

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
