#!/usr/bin/env python3
"""Complete fixed training script"""
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

purchase_counts = tx.groupby(["customer_id", "article_id"]).size().reset_index(name="purchase_count")
data = purchase_counts.merge(items, on="article_id", how="left")
data = data.merge(users, on="customer_id", how="left")

id_cols = ["customer_id", "article_id", "product_code"]
cols_to_drop = ["purchase_count"] + [col for col in id_cols if col in data.columns]
X = data.drop(columns=cols_to_drop)
y = data["purchase_count"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Predict
    y_pred = pipe.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    # Save model
    model_path = f"data/hnm/models/{pkl_name}"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)

    print(f"✓ Model saved → {model_path}")

    # Store results for comparison
    results.append({
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    })

# Print comparison table
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(f"\n{'Model':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 60)
for r in results:
    print(f"{r['model']:<25} {r['rmse']:<10.4f} {r['mae']:<10.4f} {r['r2']:<10.4f}")

print("\n✓ ALL MODELS TRAINED SUCCESSFULLY!")
