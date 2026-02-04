#!/usr/bin/env python3
"""
PySpark Tableau Data Preparation
Prepares evaluation data for Tableau visualization
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import os

# Initialize Spark
print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("HnM_Tableau_Prep") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

EVAL_PATH = "data/hnm/pyspark_evaluation"
TABLEAU_PATH = "data/hnm/pyspark_tableau"

os.makedirs(TABLEAU_PATH, exist_ok=True)

print("\n" + "="*60)
print("LOADING EVALUATION DATA")
print("="*60)

# Load metrics summary
metrics_df = spark.read.parquet(f"{EVAL_PATH}/model_metrics_summary")
print(f"✓ Loaded metrics for {metrics_df.count()} models")

print("\n" + "="*60)
print("CREATING TABLEAU-FRIENDLY DATASETS")
print("="*60)

# 1. Long format for metrics
print("\n1. Creating long-format metrics...")
metrics_long = metrics_df.select(
    "model",
    struct(
        lit("Training").alias("dataset"),
        lit("RMSE").alias("metric_name"),
        col("train_rmse").alias("metric_value")
    ).alias("row1"),
    struct(
        lit("Training").alias("dataset"),
        lit("MAE").alias("metric_name"),
        col("train_mae").alias("metric_value")
    ).alias("row2"),
    struct(
        lit("Training").alias("dataset"),
        lit("R²").alias("metric_name"),
        col("train_r2").alias("metric_value")
    ).alias("row3"),
    struct(
        lit("Test").alias("dataset"),
        lit("RMSE").alias("metric_name"),
        col("test_rmse").alias("metric_value")
    ).alias("row4"),
    struct(
        lit("Test").alias("dataset"),
        lit("MAE").alias("metric_name"),
        col("test_mae").alias("metric_value")
    ).alias("row5"),
    struct(
        lit("Test").alias("dataset"),
        lit("R²").alias("metric_name"),
        col("test_r2").alias("metric_value")
    ).alias("row6")
)

# Unpivot
from functools import reduce
long_dfs = []
for i in range(1, 7):
    long_dfs.append(
        metrics_long.select("model", f"row{i}.*")
    )

metrics_long_final = reduce(lambda a, b: a.union(b), long_dfs)

metrics_long_final.write.mode("overwrite").parquet(f"{TABLEAU_PATH}/metrics_long_format")
metrics_long_final.coalesce(1).write.mode("overwrite") \
    .option("header", True).csv(f"{TABLEAU_PATH}/metrics_long_format_csv")

print(f"  ✓ Saved: {TABLEAU_PATH}/metrics_long_format")

# 2. Model comparison with categories
print("\n2. Creating model comparison...")
comparison_df = metrics_df \
    .withColumn("model_type",
                when(col("model").contains("Fast"), "Fast (Sample)")
                .otherwise("Full")) \
    .withColumn("model_family",
                when(col("model").contains("Linear"), "Linear Regression")
                .when(col("model").contains("Ridge"), "Ridge Regression")
                .when(col("model").contains("Random"), "Random Forest")
                .when(col("model").contains("Gradient"), "Gradient Boosting")
                .when(col("model").contains("Decision"), "Decision Tree")
                .otherwise("Other")) \
    .withColumn("test_vs_train_rmse_diff", col("test_rmse") - col("train_rmse")) \
    .withColumn("is_overfitting", col("test_vs_train_rmse_diff") > 0.01)

comparison_df.write.mode("overwrite").parquet(f"{TABLEAU_PATH}/model_comparison")
comparison_df.coalesce(1).write.mode("overwrite") \
    .option("header", True).csv(f"{TABLEAU_PATH}/model_comparison_csv")

print(f"  ✓ Saved: {TABLEAU_PATH}/model_comparison")

# 3. Model family summary
print("\n3. Creating model family summary...")
family_summary = comparison_df.groupBy("model_family").agg(
    count("*").alias("num_models"),
    avg("test_rmse").alias("avg_test_rmse"),
    min("test_rmse").alias("min_test_rmse"),
    max("test_rmse").alias("max_test_rmse"),
    avg("test_r2").alias("avg_test_r2")
)

# Add best model per family
window_spec = Window.partitionBy("model_family").orderBy("test_rmse")
best_per_family = comparison_df \
    .withColumn("rank", row_number().over(window_spec)) \
    .filter(col("rank") == 1) \
    .select(col("model_family"), col("model").alias("best_model"))

family_summary = family_summary.join(best_per_family, on="model_family", how="left")

family_summary.write.mode("overwrite").parquet(f"{TABLEAU_PATH}/model_family_summary")
family_summary.coalesce(1).write.mode("overwrite") \
    .option("header", True).csv(f"{TABLEAU_PATH}/model_family_summary_csv")

print(f"  ✓ Saved: {TABLEAU_PATH}/model_family_summary")

# 4. Aggregate predictions sample
print("\n4. Aggregating prediction samples...")
from pathlib import Path

pred_dir = Path(f"{EVAL_PATH}/predictions")
pred_files = [f for f in pred_dir.glob("*_predictions") if f.is_dir()]

all_preds = []
for pred_file in pred_files:
    model_name = pred_file.name.replace('_predictions', '').replace('_', ' ').title()
    try:
        preds = spark.read.parquet(str(pred_file)) \
            .withColumn("model", lit(model_name)) \
            .limit(10000)  # Sample 10k per model
        all_preds.append(preds)
    except:
        continue

if all_preds:
    from functools import reduce
    combined_preds = reduce(lambda a, b: a.union(b), all_preds)

    combined_preds.write.mode("overwrite").parquet(f"{TABLEAU_PATH}/predictions_sample")
    combined_preds.coalesce(1).write.mode("overwrite") \
        .option("header", True).csv(f"{TABLEAU_PATH}/predictions_sample_csv")

    print(f"  ✓ Saved: {TABLEAU_PATH}/predictions_sample ({combined_preds.count():,} rows)")

# 5. Error distribution
print("\n5. Creating error distribution...")
if all_preds:
    error_dist = []

    for pred_file in pred_files:
        model_name = pred_file.name.replace('_predictions', '').replace('_', ' ').title()
        try:
            preds = spark.read.parquet(str(pred_file))
            total = preds.count()

            # Calculate error buckets
            buckets = [
                ("Perfect (error = 0)", preds.filter(col("abs_error") == 0).count()),
                ("Small (0 < error ≤ 0.5)", preds.filter((col("abs_error") > 0) & (col("abs_error") <= 0.5)).count()),
                ("Medium (0.5 < error ≤ 1)", preds.filter((col("abs_error") > 0.5) & (col("abs_error") <= 1)).count()),
                ("Large (error > 1)", preds.filter(col("abs_error") > 1).count()),
            ]

            for bucket_name, cnt in buckets:
                error_dist.append({
                    "model": model_name,
                    "error_bucket": bucket_name,
                    "count": int(cnt),
                    "percentage": float(cnt / total * 100) if total > 0 else 0.0
                })
        except:
            continue

    from pyspark.sql import Row
    error_df = spark.createDataFrame([Row(**e) for e in error_dist])

    error_df.write.mode("overwrite").parquet(f"{TABLEAU_PATH}/error_distribution")
    error_df.coalesce(1).write.mode("overwrite") \
        .option("header", True).csv(f"{TABLEAU_PATH}/error_distribution_csv")

    print(f"  ✓ Saved: {TABLEAU_PATH}/error_distribution")

print("\n" + "="*60)
print("TABLEAU DATA FILES CREATED")
print("="*60)
print(f"\nAll files saved in: {TABLEAU_PATH}/")
print("\nFiles created:")
print("  1. metrics_long_format - For metric comparisons")
print("  2. model_comparison - Enhanced metrics with categories")
print("  3. model_family_summary - Aggregated by model type")
print("  4. predictions_sample - Sample predictions")
print("  5. error_distribution - Error bucket analysis")

print("\n" + "="*60)
print("SUGGESTED TABLEAU VISUALIZATIONS")
print("="*60)
print("""
1. Model Performance Dashboard:
   - Bar chart: Test RMSE by model
   - Side-by-side: Train vs Test metrics
   - Scatter: RMSE vs R²

2. Overfitting Analysis:
   - Bullet chart: Train vs Test RMSE difference
   - Highlight overfitting models

3. Error Distribution:
   - Stacked bars: Error buckets by model
   - Pie charts for best model

4. Predictions:
   - Scatter: Actual vs Predicted
   - Histogram: Error distribution
   - Box plot: Errors by model

5. Model Family Comparison:
   - Bar chart: Average metrics by family
   - Heat map: Metric comparison matrix
""")

spark.stop()
print("\n✓ TABLEAU DATA PREPARATION COMPLETE!")
