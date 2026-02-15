#!/usr/bin/env python3
"""
PySpark Model Evaluation Pipeline
Evaluates trained Spark ML models and generates CSV outputs
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
import os
from pathlib import Path

# Initialize Spark
print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("HnM_Model_Evaluation") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

FEATURES_PATH = "data/hnm/pyspark_features"
MODELS_PATH = "data/hnm/pyspark_models"
EVAL_PATH = "data/hnm/pyspark_evaluation"

os.makedirs(EVAL_PATH, exist_ok=True)
os.makedirs(f"{EVAL_PATH}/predictions", exist_ok=True)

print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

# Load and prepare data (same as training)
articles = spark.read.parquet(f"{FEATURES_PATH}/articles_features")
customers = spark.read.parquet(f"{FEATURES_PATH}/customers_features")
transactions = spark.read.parquet(f"{FEATURES_PATH}/transactions_sample")

purchase_counts = transactions.groupBy("customer_id", "article_id") \
    .agg(count("*").alias("purchase_count"))

data = purchase_counts \
    .join(articles, on="article_id", how="left") \
    .join(customers, on="customer_id", how="left")

# Select numeric features
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

# Fill nulls before assembling
data_filled = data.na.fill(0.0)

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
data_assembled = assembler.transform(data_filled).select("customer_id", "article_id", "features", "purchase_count")
data_final = data_assembled.withColumnRenamed("purchase_count", "label")

# Stratified 3-way split matching training script
data_final = data_final.withColumn(
    "strata",
    when(col("label") <= 1, lit(0))
    .when(col("label") <= 2, lit(1))
    .when(col("label") <= 5, lit(2))
    .otherwise(lit(3))
)

train_data = data_final.sampleBy("strata", fractions={0: 0.6, 1: 0.6, 2: 0.6, 3: 0.6}, seed=42)
remaining = data_final.subtract(train_data)
val_data = remaining.sampleBy("strata", fractions={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}, seed=42)
test_data = remaining.subtract(val_data)

train_data = train_data.drop("strata")
val_data = val_data.drop("strata")
test_data = test_data.drop("strata")
data_final = data_final.drop("strata")

print(f"✓ Train set:      {train_data.count():,} rows")
print(f"✓ Validation set: {val_data.count():,} rows")
print(f"✓ Test set:       {test_data.count():,} rows")

# Find trained models
model_dirs = [d for d in Path(MODELS_PATH).iterdir()
              if d.is_dir() and not d.name.startswith('.') and d.name != "training_results"]

print(f"\n✓ Found {len(model_dirs)} trained models")

# Evaluators
rmse_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
mae_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
r2_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

print("\n" + "="*60)
print("EVALUATING MODELS")
print("="*60)

all_metrics = []

for model_dir in sorted(model_dirs):
    model_name = model_dir.name.replace('_', ' ').title()

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    try:
        # Load model
        model = PipelineModel.load(str(model_dir))

        # Predictions on all sets
        print("  Making predictions...")
        train_preds = model.transform(train_data)
        val_preds = model.transform(val_data)
        test_preds = model.transform(test_data)

        # Calculate metrics for all sets
        train_rmse = rmse_evaluator.evaluate(train_preds)
        train_mae = mae_evaluator.evaluate(train_preds)
        train_r2 = r2_evaluator.evaluate(train_preds)

        val_rmse = rmse_evaluator.evaluate(val_preds)
        val_mae = mae_evaluator.evaluate(val_preds)
        val_r2 = r2_evaluator.evaluate(val_preds)

        test_rmse = rmse_evaluator.evaluate(test_preds)
        test_mae = mae_evaluator.evaluate(test_preds)
        test_r2 = r2_evaluator.evaluate(test_preds)

        print(f"\n  {'Set':<12} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        print(f"  {'-'*42}")
        print(f"  {'Train':<12} {train_rmse:<10.4f} {train_mae:<10.4f} {train_r2:<10.4f}")
        print(f"  {'Validation':<12} {val_rmse:<10.4f} {val_mae:<10.4f} {val_r2:<10.4f}")
        print(f"  {'Test':<12} {test_rmse:<10.4f} {test_mae:<10.4f} {test_r2:<10.4f}")

        overfit_gap = train_rmse - val_rmse
        if abs(overfit_gap) > 0.1:
            print(f"  ** Overfitting detected: train-val RMSE gap = {overfit_gap:.4f}")

        # Save predictions
        pred_output = test_preds.select(
            "customer_id", "article_id",
            col("label").alias("actual"),
            col("prediction").alias("predicted")
        ).withColumn("error", col("actual") - col("predicted")) \
         .withColumn("abs_error", abs(col("error")))

        pred_file = f"{EVAL_PATH}/predictions/{model_dir.name}_predictions"
        pred_output.write.mode("overwrite").parquet(pred_file)

        # Also save as CSV
        pred_output.coalesce(1).write.mode("overwrite") \
            .option("header", True).csv(f"{pred_file}_csv")

        print(f"\n  ✓ Predictions saved → {pred_file}")

        # Store metrics
        all_metrics.append({
            "model": model_name,
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
            "train_r2": float(train_r2),
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),
            "val_r2": float(val_r2),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2),
            "overfitting_gap": float(train_rmse - val_rmse)
        })

    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        continue

print("\n" + "="*60)
print("SAVING METRICS SUMMARY")
print("="*60)

# Save metrics to parquet and CSV
from pyspark.sql import Row
metrics_df = spark.createDataFrame([Row(**m) for m in all_metrics])

metrics_df.write.mode("overwrite").parquet(f"{EVAL_PATH}/model_metrics_summary")
metrics_df.coalesce(1).write.mode("overwrite") \
    .option("header", True).csv(f"{EVAL_PATH}/model_metrics_summary_csv")

print(f"\n✓ Metrics saved → {EVAL_PATH}/model_metrics_summary")

# Print comparison
metrics_sorted = sorted(all_metrics, key=lambda x: x['val_rmse'])

print("\n" + "="*60)
print("MODEL COMPARISON (sorted by Validation RMSE)")
print("="*60)
print(f"\n{'Model':<30} {'Train RMSE':<12} {'Val RMSE':<12} {'Test RMSE':<12} {'Val R²':<10}")
print("-" * 80)
for m in metrics_sorted:
    print(f"{m['model']:<30} {m['train_rmse']:<12.4f} {m['val_rmse']:<12.4f} {m['test_rmse']:<12.4f} {m['val_r2']:<10.4f}")

if metrics_sorted:
    best = metrics_sorted[0]
    print(f"\nBest Model (by Validation RMSE): {best['model']}")
    print(f"   Val RMSE:  {best['val_rmse']:.4f}")
    print(f"   Test RMSE: {best['test_rmse']:.4f}")
    print(f"   Val R²:    {best['val_r2']:.4f}")

spark.stop()
print("\n✓ EVALUATION COMPLETE!")
