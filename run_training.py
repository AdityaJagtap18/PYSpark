#!/usr/bin/env python3
"""Complete fixed training script with stratified train/val/test split"""
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print("Loading data...")
items = pd.read_csv("data/hnm/processed/articles_features.csv", dtype={"article_id": str})
users = pd.read_csv("data/hnm/processed/customers_features.csv", dtype={"customer_id": str})
tx = pd.read_csv("data/hnm/processed/transactions_sample.csv",
                 dtype={"customer_id": str, "article_id": str})

# ── Behavioral features from transaction history ──
print("Creating behavioral features from transactions...")

# Customer-level behavioral features
cust_agg = tx.groupby("customer_id").agg(
    cust_total_purchases=("article_id", "count"),
    cust_unique_articles=("article_id", "nunique"),
    cust_avg_price=("price", "mean"),
    cust_total_spend=("price", "sum"),
).reset_index()
cust_agg["cust_purchase_diversity"] = cust_agg["cust_unique_articles"] / cust_agg["cust_total_purchases"]

# Article-level behavioral features
art_agg = tx.groupby("article_id").agg(
    art_total_purchases=("customer_id", "count"),
    art_unique_buyers=("customer_id", "nunique"),
    art_avg_price=("price", "mean"),
).reset_index()
art_agg["art_repeat_buyer_ratio"] = 1 - (art_agg["art_unique_buyers"] / art_agg["art_total_purchases"])

print(f"  ✓ Customer features: {len(cust_agg.columns)-1}")
print(f"  ✓ Article features:  {len(art_agg.columns)-1}")

purchase_counts = tx.groupby(["customer_id", "article_id"]).size().reset_index(name="purchase_count")
data = purchase_counts.merge(items, on="article_id", how="left")
data = data.merge(users, on="customer_id", how="left")
data = data.merge(cust_agg, on="customer_id", how="left")
data = data.merge(art_agg, on="article_id", how="left")

id_cols = ["customer_id", "article_id", "product_code"]
cols_to_drop = ["purchase_count"] + [col for col in id_cols if col in data.columns]
X = data.drop(columns=cols_to_drop)
y = data["purchase_count"]

# Stratified 3-way split: train (60%) / validation (20%) / test (20%)
# Bin purchase_count into strata for stratified splitting
strata = pd.cut(y, bins=[-np.inf, 1, 2, 5, np.inf], labels=[0, 1, 2, 3])

# First split: 60% train, 40% temp
X_train, X_temp, y_train, y_temp, strata_train, strata_temp = train_test_split(
    X, y, strata, test_size=0.4, random_state=42, stratify=strata
)
# Second split: 50/50 of temp -> 20% val, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=strata_temp
)

print(f"\nStratified split sizes:")
print(f"  Train:      {len(X_train):,} ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Validation: {len(X_val):,} ({len(X_val)/len(X)*100:.0f}%)")
print(f"  Test:       {len(X_test):,} ({len(X_test)/len(X)*100:.0f}%)")
print(f"\nTarget distribution (purchase_count):")
print(f"  Train mean: {y_train.mean():.3f}, Test mean: {y_test.mean():.3f}, Val mean: {y_val.mean():.3f}")

# COMPLETE FIX: Handle string AND boolean columns properly
cat_cols = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
num_cols = X_train.select_dtypes(include=['number', 'bool']).columns.tolist()

print(f"\n✓ Categorical columns: {len(cat_cols)}")
print(f"✓ Numerical columns (including {X_train.select_dtypes(include=['bool']).shape[1]} booleans): {len(num_cols)}")
print(f"✓ Total: {len(cat_cols) + len(num_cols)}/{X_train.shape[1]}")

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

# Define models to train
models = [
    ("Linear Regression", LinearRegression(), "linear_regression.pkl"),
    ("Ridge Regression", Ridge(alpha=1.0), "ridge_regression.pkl"),
    ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), "random_forest.pkl"),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42), "gradient_boosting.pkl"),
]

os.makedirs("data/hnm/models", exist_ok=True)
results = []

# Train each model
for model_name, model, pkl_name in models:
    print("\n" + "="*60)
    print(f"Training {model_name}...")
    print("="*60)

    # Create pipeline
    pipe = Pipeline([("prep", preprocess), ("reg", model)])

    # Train
    pipe.fit(X_train, y_train)

    # Predict on all sets
    y_pred_train = pipe.predict(X_train)
    y_pred_val = pipe.predict(X_val)
    y_pred_test = pipe.predict(X_test)

    # Calculate metrics for each set
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\n{model_name} Results:")
    print(f"  {'Set':<12} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print(f"  {'-'*42}")
    print(f"  {'Train':<12} {train_rmse:<10.4f} {train_mae:<10.4f} {train_r2:<10.4f}")
    print(f"  {'Validation':<12} {val_rmse:<10.4f} {val_mae:<10.4f} {val_r2:<10.4f}")
    print(f"  {'Test':<12} {test_rmse:<10.4f} {test_mae:<10.4f} {test_r2:<10.4f}")

    overfit_gap = train_rmse - val_rmse
    if abs(overfit_gap) > 0.1:
        print(f"  ** Overfitting detected: train-val RMSE gap = {overfit_gap:.4f}")

    # Save model
    model_path = f"data/hnm/models/{pkl_name}"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)

    print(f"✓ Model saved → {model_path}")

    # Store results for comparison
    results.append({
        'model': model_name,
        'train_rmse': train_rmse, 'val_rmse': val_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'val_mae': val_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'val_r2': val_r2, 'test_r2': test_r2,
    })

# Print comparison table
print("\n" + "="*60)
print("MODEL COMPARISON (Train / Validation / Test)")
print("="*60)
print(f"\n{'Model':<25} {'Train RMSE':<12} {'Val RMSE':<12} {'Test RMSE':<12} {'Val R²':<10}")
print("-" * 75)
for r in results:
    print(f"{r['model']:<25} {r['train_rmse']:<12.4f} {r['val_rmse']:<12.4f} {r['test_rmse']:<12.4f} {r['val_r2']:<10.4f}")

# Find best model by validation RMSE (not test!)
best_model = min(results, key=lambda x: x['val_rmse'])
print(f"\nBest model (lowest Validation RMSE): {best_model['model']}")
print(f"  Val RMSE:  {best_model['val_rmse']:.4f}")
print(f"  Test RMSE: {best_model['test_rmse']:.4f}")

print("\n✓ ALL MODELS TRAINED SUCCESSFULLY!")
