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
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.sql.shuffle.partitions", "10") \
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

# Stratified 3-way split: train (60%) / validation (20%) / test (20%)
# Create strata bins based on purchase_count distribution
print("\nCreating stratified train/validation/test split (60/20/20)...")
data_final = data_final.withColumn(
    "strata",
    when(col("label") <= 1, lit(0))
    .when(col("label") <= 2, lit(1))
    .when(col("label") <= 5, lit(2))
    .otherwise(lit(3))
)

# Compute per-stratum fractions for sampleBy
# First split: 60% train
train_data = data_final.sampleBy("strata", fractions={0: 0.6, 1: 0.6, 2: 0.6, 3: 0.6}, seed=42)
remaining = data_final.subtract(train_data)

# Second split: 50% of remaining -> 20% val, 20% test
val_data = remaining.sampleBy("strata", fractions={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}, seed=42)
test_data = remaining.subtract(val_data)

# Drop the strata column
train_data = train_data.drop("strata")
val_data = val_data.drop("strata")
test_data = test_data.drop("strata")
data_final = data_final.drop("strata")

train_count = train_data.count()
val_count = val_data.count()
test_count = test_data.count()
total = train_count + val_count + test_count

print(f"✓ Train set:      {train_count:,} rows ({train_count/total*100:.0f}%)")
print(f"✓ Validation set: {val_count:,} rows ({val_count/total*100:.0f}%)")
print(f"✓ Test set:       {test_count:,} rows ({test_count/total*100:.0f}%)")

# Cache for performance
train_data.cache()
val_data.cache()
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

    # Predict on all sets
    print("  Making predictions...")
    train_preds = trained_model.transform(train_data)
    val_preds = trained_model.transform(val_data)
    test_preds = trained_model.transform(test_data)

    # Calculate metrics for each set
    train_rmse = rmse_evaluator.evaluate(train_preds)
    val_rmse = rmse_evaluator.evaluate(val_preds)
    test_rmse = rmse_evaluator.evaluate(test_preds)

    train_mae = mae_evaluator.evaluate(train_preds)
    val_mae = mae_evaluator.evaluate(val_preds)
    test_mae = mae_evaluator.evaluate(test_preds)

    train_r2 = r2_evaluator.evaluate(train_preds)
    val_r2 = r2_evaluator.evaluate(val_preds)
    test_r2 = r2_evaluator.evaluate(test_preds)

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
    model_path = f"{MODELS_PATH}/{model_name.lower().replace(' ', '_')}"
    trained_model.write().overwrite().save(model_path)
    print(f"\n✓ Model saved → {model_path}")

    results.append({
        'model': model_name,
        'train_rmse': train_rmse, 'val_rmse': val_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'val_mae': val_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'val_r2': val_r2, 'test_r2': test_r2,
    })

# Print comparison
print("\n" + "="*60)
print("MODEL COMPARISON (Train / Validation / Test)")
print("="*60)
print(f"\n{'Model':<25} {'Train RMSE':<12} {'Val RMSE':<12} {'Test RMSE':<12} {'Val R²':<10}")
print("-" * 75)
for r in results:
    print(f"{r['model']:<25} {r['train_rmse']:<12.4f} {r['val_rmse']:<12.4f} {r['test_rmse']:<12.4f} {r['val_r2']:<10.4f}")

# Find best model by validation RMSE
best_model = min(results, key=lambda x: x['val_rmse'])
print(f"\nBest model (lowest Validation RMSE): {best_model['model']}")
print(f"  Val RMSE:  {best_model['val_rmse']:.4f}")
print(f"  Test RMSE: {best_model['test_rmse']:.4f}")

# Save results summary
from pyspark.sql import Row
results_df = spark.createDataFrame([Row(**r) for r in results])
results_df.write.mode("overwrite").parquet(f"{MODELS_PATH}/training_results")
print(f"\n✓ Results saved to {MODELS_PATH}/training_results/")

spark.stop()
print("\n✓ MODEL TRAINING COMPLETE!")
