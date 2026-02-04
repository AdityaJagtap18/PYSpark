#!/usr/bin/env python3
"""
PySpark Model Training Pipeline
Trains regression models using Spark MLlib
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import os

# Initialize Spark
print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("HnM_Model_Training") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

FEATURES_PATH = "data/hnm/pyspark_features"
MODELS_PATH = "data/hnm/pyspark_models"
os.makedirs(MODELS_PATH, exist_ok=True)

print("\n" + "="*60)
print("LOADING FEATURE DATA")
print("="*60)

# Load features
articles = spark.read.parquet(f"{FEATURES_PATH}/articles_features")
customers = spark.read.parquet(f"{FEATURES_PATH}/customers_features")
transactions = spark.read.parquet(f"{FEATURES_PATH}/transactions_sample")

print(f"\n✓ Articles:      {articles.count():,} rows")
print(f"✓ Customers:     {customers.count():,} rows")
print(f"✓ Transactions:  {transactions.count():,} rows")

print("\n" + "="*60)
print("CREATING TRAINING DATASET")
print("="*60)

# Create purchase counts
print("\nAggregating purchase counts...")
purchase_counts = transactions.groupBy("customer_id", "article_id") \
    .agg(count("*").alias("purchase_count"))

print(f"Purchase count pairs: {purchase_counts.count():,}")

# Join with features
print("\nJoining with features...")
data = purchase_counts \
    .join(articles, on="article_id", how="left") \
    .join(customers, on="customer_id", how="left")

print(f"Merged dataset: {data.count():,} rows, {len(data.columns)} columns")

# Select numeric features only (drop string columns)
print("\nSelecting numeric features...")
feature_cols = [c for c in data.columns if c not in
                ["customer_id", "article_id", "purchase_count", "product_code",
                 "prod_name", "product_type_name", "product_group_name",
                 "graphical_appearance_name", "colour_group_name",
                 "perceived_colour_value_name", "perceived_colour_master_name",
                 "department_name", "index_code", "index_name", "index_group_name",
                 "section_name", "garment_group_name", "detail_desc",
                 "colour_group_name_norm", "FN", "Active", "club_member_status",
                 "fashion_news_frequency", "postal_code", "transaction_date",
                 "t_dat", "price", "sales_channel_id"]]

print(f"Feature columns: {len(feature_cols)}")

# Fill nulls before assembling (VectorAssembler can't handle nulls)
print("\nFilling null values...")
data_filled = data.na.fill(0.0)

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
data_assembled = assembler.transform(data_filled).select("features", "purchase_count")

# Rename target column
data_final = data_assembled.withColumnRenamed("purchase_count", "label")

# Split data
print("\nSplitting train/test (80/20)...")
train_data, test_data = data_final.randomSplit([0.8, 0.2], seed=42)

print(f"✓ Train set: {train_data.count():,} rows")
print(f"✓ Test set: {test_data.count():,} rows")

# Cache for performance
train_data.cache()
test_data.cache()

print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

# Define models
models = [
    ("Linear Regression", LinearRegression(maxIter=10, regParam=0.0)),
    ("Ridge Regression", LinearRegression(maxIter=10, regParam=1.0, elasticNetParam=0.0)),
    ("Decision Tree", DecisionTreeRegressor(maxDepth=10)),
    ("Random Forest", RandomForestRegressor(numTrees=50, maxDepth=10)),
    ("Gradient Boosting", GBTRegressor(maxIter=50, maxDepth=5)),
]

# Evaluators
rmse_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
mae_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
r2_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

results = []

for model_name, model in models:
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")

    # Create scaler + model pipeline
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

    # Update model to use scaled features
    if hasattr(model, 'setFeaturesCol'):
        model.setFeaturesCol("scaled_features")

    pipeline = Pipeline(stages=[scaler, model])

    # Train
    print("  Fitting model...")
    trained_model = pipeline.fit(train_data)

    # Predict on test
    print("  Making predictions...")
    predictions = trained_model.transform(test_data)

    # Calculate metrics
    rmse = rmse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)
    r2 = r2_evaluator.evaluate(predictions)

    print(f"\n{model_name} Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    # Save model
    model_path = f"{MODELS_PATH}/{model_name.lower().replace(' ', '_')}"
    trained_model.write().overwrite().save(model_path)
    print(f"\n✓ Model saved → {model_path}")

    results.append({
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    })

# Print comparison
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(f"\n{'Model':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 60)
for r in results:
    print(f"{r['model']:<25} {r['rmse']:<10.4f} {r['mae']:<10.4f} {r['r2']:<10.4f}")

# Find best model
best_model = min(results, key=lambda x: x['rmse'])
print(f"\n🏆 Best model (lowest RMSE): {best_model['model']}")

# Save results summary
from pyspark.sql import Row
results_df = spark.createDataFrame([Row(**r) for r in results])
results_df.write.mode("overwrite").parquet(f"{MODELS_PATH}/training_results")
print(f"\n✓ Results saved to {MODELS_PATH}/training_results/")

spark.stop()
print("\n✓ MODEL TRAINING COMPLETE!")
