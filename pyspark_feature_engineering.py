#!/usr/bin/env python3
"""
PySpark Feature Engineering Pipeline
Reads ingested data and creates features for ML models
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder
import os

# Initialize Spark Session
print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("HnM_Feature_Engineering") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Define paths
STAGING_PATH = "data/hnm/staging"
FEATURES_PATH = "data/hnm/pyspark_features"

# Create features directory
os.makedirs(FEATURES_PATH, exist_ok=True)

print("\n" + "="*60)
print("LOADING INGESTED DATA")
print("="*60)

# Load from staging
articles = spark.read.parquet(f"{STAGING_PATH}/articles")
customers = spark.read.parquet(f"{STAGING_PATH}/customers")
transactions = spark.read.parquet(f"{STAGING_PATH}/transactions")

print(f"\n✓ Articles:      {articles.count():,} rows")
print(f"✓ Customers:     {customers.count():,} rows")
print(f"✓ Transactions:  {transactions.count():,} rows")

print("\n" + "="*60)
print("FEATURE ENGINEERING: ARTICLES")
print("="*60)

# Create frequency encodings for categorical features
print("\n1. Creating frequency encodings...")

categorical_cols = [
    "product_type_name", "product_group_name", "graphical_appearance_name",
    "colour_group_name", "perceived_colour_value_name", "department_name",
    "index_name", "index_group_name", "section_name", "garment_group_name"
]

articles_features = articles

for col_name in categorical_cols:
    freq_col = f"{col_name}_freq"
    # Calculate frequency
    freq_df = articles.groupBy(col_name).agg(
        count("*").alias(freq_col)
    )
    articles_features = articles_features.join(freq_df, on=col_name, how="left")
    print(f"  ✓ {freq_col}")

# Create text features from product name and description
print("\n2. Creating text features...")
articles_features = articles_features \
    .withColumn("text_len", length(col("detail_desc"))) \
    .withColumn("text_word_count",
                size(split(col("detail_desc"), "\\s+"))) \
    .withColumn("has_description",
                when(length(col("detail_desc")) > 0, 1).otherwise(0))

# Normalize color name
print("\n3. Normalizing color names...")
articles_features = articles_features \
    .withColumn("colour_group_name_norm",
                lower(trim(col("colour_group_name"))))

# One-hot encode categorical features
print("\n4. Creating one-hot encoded features...")

onehot_cols = [
    "product_group_name", "graphical_appearance_name",
    "perceived_colour_value_name", "index_group_name",
    "garment_group_name"
]

for col_name in onehot_cols:
    print(f"  Processing {col_name}...")
    # Get unique values
    unique_vals = articles.select(col_name).distinct().rdd.flatMap(lambda x: x).collect()
    unique_vals = [v for v in unique_vals if v is not None]

    # Create boolean columns
    for val in unique_vals:
        safe_val = val.replace(" ", "_").replace("/", "_").replace(",", "").replace("&", "and")
        new_col = f"{col_name}_{safe_val}"
        articles_features = articles_features.withColumn(
            new_col,
            when(col(col_name) == val, True).otherwise(False)
        )
    print(f"  ✓ Created {len(unique_vals)} features for {col_name}")

print(f"\n✓ Total article features: {len(articles_features.columns)}")

# Save articles features
print("\nSaving articles features...")
articles_features.write.mode("overwrite").parquet(f"{FEATURES_PATH}/articles_features")
print(f"✓ Saved to {FEATURES_PATH}/articles_features/")

print("\n" + "="*60)
print("FEATURE ENGINEERING: CUSTOMERS")
print("="*60)

# Create age bins
print("\n1. Creating age bins...")
customers_features = customers \
    .withColumn("age_bin",
                when(col("age") < 20, 0)
                .when(col("age") < 30, 1)
                .when(col("age") < 40, 2)
                .when(col("age") < 50, 3)
                .when(col("age") < 60, 4)
                .otherwise(5))

# Create frequency encodings for customer features
print("\n2. Creating frequency encodings...")

customer_cat_cols = ["fashion_news_frequency", "club_member_status", "postal_code"]

for col_name in customer_cat_cols:
    freq_col = f"{col_name}_freq"
    freq_df = customers.groupBy(col_name).agg(
        count("*").alias(freq_col)
    )
    customers_features = customers_features.join(freq_df, on=col_name, how="left")
    print(f"  ✓ {freq_col}")

print(f"\n✓ Total customer features: {len(customers_features.columns)}")

# Save customers features
print("\nSaving customers features...")
customers_features.write.mode("overwrite").parquet(f"{FEATURES_PATH}/customers_features")
print(f"✓ Saved to {FEATURES_PATH}/customers_features/")

print("\n" + "="*60)
print("FEATURE ENGINEERING: TRANSACTIONS")
print("="*60)

# Sample transactions (10% for faster processing)
print("\n1. Sampling transactions (10%)...")
transactions_sample = transactions.sample(fraction=0.1, seed=42)
print(f"  Sampled: {transactions_sample.count():,} rows")

# Convert date to timestamp
print("\n2. Processing dates...")
transactions_sample = transactions_sample \
    .withColumn("transaction_date", to_date(col("t_dat")))

# Save transactions sample
print("\nSaving transactions sample...")
transactions_sample.write.mode("overwrite").parquet(f"{FEATURES_PATH}/transactions_sample")
print(f"✓ Saved to {FEATURES_PATH}/transactions_sample/")

# Also save as CSV for compatibility with existing pipeline
print("\nExporting to CSV for compatibility...")
transactions_sample.select("customer_id", "article_id", "transaction_date", "price", "sales_channel_id") \
    .coalesce(1) \
    .write.mode("overwrite") \
    .option("header", True) \
    .csv(f"{FEATURES_PATH}/transactions_sample_csv")
print(f"✓ Saved CSV version")

print("\n" + "="*60)
print("FEATURE ENGINEERING SUMMARY")
print("="*60)
print(f"\nFeatures saved to: {FEATURES_PATH}/")
print(f"\nArticles Features:")
print(f"  - Rows: {articles_features.count():,}")
print(f"  - Columns: {len(articles_features.columns)}")
print(f"\nCustomers Features:")
print(f"  - Rows: {customers_features.count():,}")
print(f"  - Columns: {len(customers_features.columns)}")
print(f"\nTransactions Sample:")
print(f"  - Rows: {transactions_sample.count():,}")
print(f"  - Columns: {len(transactions_sample.columns)}")

print("\n" + "="*60)
print("SAMPLE FEATURES")
print("="*60)
print("\nArticles Features (first 5):")
articles_features.select(
    "article_id", "product_code", "colour_group_name_norm",
    "text_len", "has_description"
).show(5, truncate=False)

print("\nCustomers Features (first 5):")
customers_features.select(
    "customer_id", "age", "age_bin", "age_missing"
).show(5, truncate=False)

# Stop Spark
spark.stop()
print("\n✓ FEATURE ENGINEERING COMPLETE!")
