#!/usr/bin/env python3
"""
Fixed training script - runs the model training with proper categorical detection
"""
import numpy as np
import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

# Load your processed features
print("Loading data...")
items = pd.read_csv("data/hnm/processed/articles_features.csv", dtype={"article_id": str})
users = pd.read_csv("data/hnm/processed/customers_features.csv", dtype={"customer_id": str})
tx = pd.read_csv("data/hnm/processed/transactions_sample.csv",
                 dtype={"customer_id": str, "article_id": str})

print("Items shape:", items.shape)
print("Users shape:", users.shape)
print("Transactions shape:", tx.shape)

# Create a regression target
print("\nCreating regression target...")
purchase_counts = tx.groupby(["customer_id", "article_id"]).size().reset_index(name="purchase_count")

# Merge with features
data = purchase_counts.merge(items, on="article_id", how="left")
data = data.merge(users, on="customer_id", how="left")

print("Merged data shape:", data.shape)

# Prepare train/test split
id_cols = ["customer_id", "article_id", "product_code"]
cols_to_drop = ["purchase_count"] + [col for col in id_cols if col in data.columns]

X = data.drop(columns=cols_to_drop)
y = data["purchase_count"]

print("Feature columns:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train set:", X_train.shape)
print("Test set:", X_test.shape)

# FIXED: Identify categorical and numerical columns properly
cat_cols = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
num_cols = X_train.select_dtypes(include=['number']).columns.tolist()

print(f"\n✓ Categorical columns: {len(cat_cols)}")
if cat_cols:
    print(f"  → {cat_cols}")
print(f"✓ Numerical columns: {len(num_cols)}")

# Create preprocessing pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

# Train model
print("\n" + "="*60)
print("Training Linear Regression...")
print("="*60)

pipe = Pipeline([
    ("prep", preprocess),
    ("reg", LinearRegression())
])

# Train
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nLinear Regression Results:")
print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"  MAE  (Mean Absolute Error):     {mae:.4f}")
print(f"  R²   (R-squared):               {r2:.4f}")

# Save model
os.makedirs("data/hnm/models", exist_ok=True)
model_path = "data/hnm/models/linear_regression.pkl"

with open(model_path, "wb") as f:
    pickle.dump(pipe, f)

print(f"\n✓ Model saved → {model_path}")
print("\n✓ SUCCESS! Training completed without errors.")
