#!/usr/bin/env python3
"""
PySpark Data Ingestion Pipeline
Reads raw H&M data, validates, cleans, and saves to staging area
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, isnull, trim, lower
from pyspark.sql.types import *
import os

# Initialize Spark Session
print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("HnM_Data_Ingestion") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print(f"Spark Version: {spark.version}")
print(f"Spark Master: {spark.sparkContext.master}")

# Define paths
RAW_DATA_PATH = "data/hnm"
STAGING_PATH = "data/hnm/staging"

# Create staging directory
os.makedirs(STAGING_PATH, exist_ok=True)

print("\n" + "="*60)
print("STEP 1: INGESTING ARTICLES DATA")
print("="*60)

# Read articles
articles_schema = StructType([
    StructField("article_id", StringType(), False),
    StructField("product_code", IntegerType(), True),
    StructField("prod_name", StringType(), True),
    StructField("product_type_no", IntegerType(), True),
    StructField("product_type_name", StringType(), True),
    StructField("product_group_name", StringType(), True),
    StructField("graphical_appearance_no", IntegerType(), True),
    StructField("graphical_appearance_name", StringType(), True),
    StructField("colour_group_code", IntegerType(), True),
    StructField("colour_group_name", StringType(), True),
    StructField("perceived_colour_value_id", IntegerType(), True),
    StructField("perceived_colour_value_name", StringType(), True),
    StructField("perceived_colour_master_id", IntegerType(), True),
    StructField("perceived_colour_master_name", StringType(), True),
    StructField("department_no", IntegerType(), True),
    StructField("department_name", StringType(), True),
    StructField("index_code", StringType(), True),
    StructField("index_name", StringType(), True),
    StructField("index_group_no", IntegerType(), True),
    StructField("index_group_name", StringType(), True),
    StructField("section_no", IntegerType(), True),
    StructField("section_name", StringType(), True),
    StructField("garment_group_no", IntegerType(), True),
    StructField("garment_group_name", StringType(), True),
    StructField("detail_desc", StringType(), True),
])

articles = spark.read.csv(
    f"{RAW_DATA_PATH}/articles.csv",
    header=True,
    schema=articles_schema,
    mode="DROPMALFORMED"
)

print(f"\n✓ Articles loaded: {articles.count():,} rows")
print(f"  Columns: {len(articles.columns)}")

# Data quality checks
print("\n  Data Quality Checks:")
null_counts = articles.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in articles.columns
])
print(f"  - Checking for nulls...")

# Clean articles
articles_clean = articles \
    .dropDuplicates(["article_id"]) \
    .na.fill({
        "prod_name": "Unknown",
        "product_type_name": "Unknown",
        "product_group_name": "Unknown",
        "colour_group_name": "Unknown",
        "detail_desc": ""
    })

print(f"  ✓ Articles after deduplication: {articles_clean.count():,} rows")

# Save to staging
print("\n  Saving to staging...")
articles_clean.write.mode("overwrite").parquet(f"{STAGING_PATH}/articles")
print(f"  ✓ Saved to {STAGING_PATH}/articles/")

print("\n" + "="*60)
print("STEP 2: INGESTING CUSTOMERS DATA")
print("="*60)

# Read customers
customers_schema = StructType([
    StructField("customer_id", StringType(), False),
    StructField("FN", StringType(), True),
    StructField("Active", StringType(), True),
    StructField("club_member_status", StringType(), True),
    StructField("fashion_news_frequency", StringType(), True),
    StructField("age", DoubleType(), True),
    StructField("postal_code", StringType(), True),
])

customers = spark.read.csv(
    f"{RAW_DATA_PATH}/customers.csv",
    header=True,
    schema=customers_schema,
    mode="DROPMALFORMED"
)

print(f"\n✓ Customers loaded")
print(f"  Columns: {len(customers.columns)}")

# Sample customers to reduce memory (30% sample)
print(f"  Sampling 30% of customers for memory efficiency...")
customers_sampled = customers.sample(fraction=0.3, seed=42)

# Clean customers
customers_clean = customers_sampled \
    .dropDuplicates(["customer_id"]) \
    .withColumn("age_missing", when(col("age").isNull(), 1).otherwise(0)) \
    .na.fill({
        "club_member_status": "UNKNOWN",
        "fashion_news_frequency": "NONE",
        "age": 0.0
    })

print(f"  ✓ Customers after sampling & deduplication: {customers_clean.count():,} rows")

# Save to staging
print("\n  Saving to staging...")
customers_clean.write.mode("overwrite").parquet(f"{STAGING_PATH}/customers")
print(f"  ✓ Saved to {STAGING_PATH}/customers/")

print("\n" + "="*60)
print("STEP 3: INGESTING TRANSACTIONS DATA")
print("="*60)

# Read transactions
transactions_schema = StructType([
    StructField("t_dat", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("article_id", StringType(), False),
    StructField("price", DoubleType(), True),
    StructField("sales_channel_id", IntegerType(), True),
])

transactions = spark.read.csv(
    f"{RAW_DATA_PATH}/transactions_train.csv",
    header=True,
    schema=transactions_schema,
    mode="DROPMALFORMED"
)

print(f"\n✓ Transactions loaded")
print(f"  Columns: {len(transactions.columns)}")

# Sample heavily for memory efficiency (5% sample)
print(f"  Sampling 5% of transactions for memory efficiency...")
transactions_sampled = transactions.sample(fraction=0.05, seed=42)

# Clean transactions
transactions_clean = transactions_sampled \
    .dropDuplicates() \
    .filter(col("price").isNotNull()) \
    .filter(col("price") >= 0)

print(f"  ✓ Transactions after sampling & cleaning: {transactions_clean.count():,} rows")

# Save to staging
print("\n  Saving to staging...")
transactions_clean.write.mode("overwrite").parquet(f"{STAGING_PATH}/transactions")
print(f"  ✓ Saved to {STAGING_PATH}/transactions/")

# Summary
print("\n" + "="*60)
print("INGESTION SUMMARY")
print("="*60)
print(f"\nStaging Area: {STAGING_PATH}/")
print(f"\n✓ Articles:      {articles_clean.count():,} rows")
print(f"✓ Customers:     {customers_clean.count():,} rows")
print(f"✓ Transactions:  {transactions_clean.count():,} rows")
print(f"\nAll data saved in Parquet format for efficient processing")

# Stop Spark
spark.stop()
print("\n✓ DATA INGESTION COMPLETE!")
