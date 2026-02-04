#!/usr/bin/env python3
"""
H&M End-to-End Big Data + ML pipeline (PySpark)

Bronze: HF cached CSV -> Parquet
Silver: cleaned/validated
Gold: features + label (repeat purchase within 7 days)
MLlib: LogisticRegression (CV), DecisionTree, LinearSVC
Sklearn baseline: sample single-node LR
Exports: Tableau CSVs

Run:
  source venv/bin/activate
  pip install -U pyspark huggingface_hub pandas scikit-learn numpy

  export SPARK_DRIVER_MEMORY=12g
  export SPARK_EXECUTOR_MEMORY=12g
  python3 hnm_bigdata_full_pipeline.py
"""

import os
import time
import json
import pickle
import numpy as np
from pathlib import Path

from huggingface_hub import hf_hub_download

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)
from pyspark.storagelevel import StorageLevel

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.storagelevel import StorageLevel



# ----------------------------
# CONFIG
# ----------------------------
HF_REPO_ID = "einrafh/hnm-fashion-recommendations-data"
HF_TX = "data/raw/transactions_train.csv"
HF_ART = "data/raw/articles.csv"
HF_CUST = "data/raw/customers.csv"

BASE_DIR = Path(__file__).resolve().parent
LAKE = BASE_DIR / "lake"
BRONZE = LAKE / "bronze"
SILVER = LAKE / "silver"
GOLD = LAKE / "gold"
MODELS = LAKE / "models"
REPORTS = LAKE / "reports"
TABLEAU = LAKE / "tableau"

for p in [BRONZE, SILVER, GOLD, MODELS, REPORTS, TABLEAU]:
    p.mkdir(parents=True, exist_ok=True)

BRONZE_TX = BRONZE / "tx"
BRONZE_ART = BRONZE / "articles"
BRONZE_CUST = BRONZE / "customers"

SILVER_TX = SILVER / "tx"
SILVER_ART = SILVER / "articles"
SILVER_CUST = SILVER / "customers"

GOLD_FEATURES = GOLD / "features"

# Tuning knobs
SHUFFLE_PARTITIONS = int(os.getenv("SHUFFLE_PARTITIONS", "128"))
TX_WRITE_PARTITIONS = int(os.getenv("TX_WRITE_PARTITIONS", "64"))
MAX_RECORDS_PER_FILE = int(os.getenv("MAX_RECORDS_PER_FILE", "750000"))

LABEL_WINDOW_DAYS = int(os.getenv("LABEL_WINDOW_DAYS", "7"))  # label = repeat purchase within N days

# ----------------------------
# SCHEMAS (avoid inferSchema)
# ----------------------------
TX_SCHEMA = StructType([
    StructField("t_dat", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("article_id", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("sales_channel_id", IntegerType(), True),
])

ART_SCHEMA = StructType([
    StructField("article_id", StringType(), True),
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

CUST_SCHEMA = StructType([
    StructField("customer_id", StringType(), True),
    StructField("FN", IntegerType(), True),
    StructField("Active", IntegerType(), True),
    StructField("club_member_status", StringType(), True),
    StructField("fashion_news_frequency", StringType(), True),
    StructField("age", DoubleType(), True),
    StructField("postal_code", StringType(), True),
])


# ----------------------------
# Spark session
# ----------------------------
def create_spark():
    return (
        SparkSession.builder
        .appName("HNM-BigData")
        .config("spark.sql.shuffle.partitions", "96")
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "12g"))
        .config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY", "12g"))
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.3")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )



def hf_cached_path(filename: str) -> str:
    return hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=filename)


# ----------------------------
# BRONZE: HF CSV -> Parquet
# ----------------------------
def ingest_bronze(spark):
    tx_csv = hf_cached_path(HF_TX)
    art_csv = hf_cached_path(HF_ART)
    cust_csv = hf_cached_path(HF_CUST)

    print("[HF] Cached paths:")
    print(" tx:", tx_csv)
    print("art:", art_csv)
    print("cust:", cust_csv)

    tx = (
        spark.read.option("header", True).schema(TX_SCHEMA).csv(tx_csv)
        .withColumn("t_dat", F.to_date("t_dat"))
        .filter(F.col("customer_id").isNotNull() & F.col("article_id").isNotNull() & F.col("t_dat").isNotNull())
        .withColumn("year_month", F.date_format("t_dat", "yyyy-MM"))
        .repartition(TX_WRITE_PARTITIONS, "year_month")
    )

    art = (
        spark.read.option("header", True).schema(ART_SCHEMA).csv(art_csv)
        .filter(F.col("article_id").isNotNull())
        .fillna({"detail_desc": ""})
    )

    cust = (
        spark.read.option("header", True).schema(CUST_SCHEMA).csv(cust_csv)
        .filter(F.col("customer_id").isNotNull())
    )

    print(f"[VALIDATION] tx={tx.count():,} | art={art.count():,} | cust={cust.count():,}")

    tx.write.mode("overwrite").partitionBy("year_month").parquet(str(BRONZE_TX))
    art.write.mode("overwrite").parquet(str(BRONZE_ART))
    cust.write.mode("overwrite").parquet(str(BRONZE_CUST))

    print("[BRONZE] Written.")


# ----------------------------
# SILVER: Clean + DQ metrics
# ----------------------------
def build_silver(spark):
    tx = spark.read.parquet(str(BRONZE_TX))
    art = spark.read.parquet(str(BRONZE_ART))
    cust = spark.read.parquet(str(BRONZE_CUST))

    dq = {}

    dq["tx_null_customer"] = tx.filter(F.col("customer_id").isNull()).count()
    dq["tx_null_article"] = tx.filter(F.col("article_id").isNull()).count()
    dq["tx_dup_triplet"] = tx.groupBy("customer_id", "article_id", "t_dat").count().filter(F.col("count") > 1).count()

    tx_s = tx.dropDuplicates(["customer_id", "article_id", "t_dat"])
    art_s = art.dropDuplicates(["article_id"])
    cust_s = cust.dropDuplicates(["customer_id"])

    tx_s.write.mode("overwrite").partitionBy("year_month").parquet(str(SILVER_TX))
    art_s.write.mode("overwrite").parquet(str(SILVER_ART))
    cust_s.write.mode("overwrite").parquet(str(SILVER_CUST))

    (REPORTS / "dq_metrics.json").write_text(json.dumps(dq, indent=2))
    print("[SILVER] Written + DQ metrics saved.")


# ----------------------------
# GOLD: Feature Engineering + Label
# Label: repeat purchase within LABEL_WINDOW_DAYS for same customer
# ----------------------------
def build_gold(spark):
    tx = spark.read.parquet(str(SILVER_TX)).select("customer_id", "article_id", "t_dat", "price", "sales_channel_id")
    art = spark.read.parquet(str(SILVER_ART)).select(
        "article_id", "product_type_name", "garment_group_name", "index_name", "section_name"
    )
    cust = spark.read.parquet(str(SILVER_CUST)).select(
        "customer_id", "age", "club_member_status", "fashion_news_frequency"
    )

    # Broadcast joins (art + cust are much smaller than tx)
    txe = (
        tx.join(F.broadcast(art), "article_id", "left")
          .join(F.broadcast(cust), "customer_id", "left")
    ).persist(StorageLevel.MEMORY_AND_DISK)

    # Popularity
    pop = txe.groupBy("article_id").agg(F.count("*").alias("article_popularity"))
    txe = txe.join(F.broadcast(pop), "article_id", "left")

    # Customer frequency (last 30 days): approximate using window on last_purchase_date
    last_date = txe.groupBy("customer_id").agg(F.max("t_dat").alias("cust_last_date"))
    txe = txe.join(F.broadcast(last_date), "customer_id", "left") \
             .withColumn("days_since_last_purchase", F.datediff(F.col("cust_last_date"), F.col("t_dat")))

    # Label computation (customer-day)
    cust_day = txe.select("customer_id", "t_dat").dropDuplicates()

    a = cust_day.alias("a")
    b = cust_day.alias("b")

    label_df = (
        a.join(
            b,
            (F.col("a.customer_id") == F.col("b.customer_id")) &
            (F.col("b.t_dat") > F.col("a.t_dat")) &
            (F.datediff(F.col("b.t_dat"), F.col("a.t_dat")) <= F.lit(LABEL_WINDOW_DAYS)),
            "left"
        )
        .select(
            F.col("a.customer_id").alias("customer_id"),
            F.col("a.t_dat").alias("t_dat"),
            F.col("b.t_dat").alias("future_date")
        )
    )

    label = (
        label_df.groupBy("customer_id", "t_dat")
        .agg((F.count("future_date") > 0).cast("int").alias("label"))
    )

    gold = (
        txe.select(
            "customer_id", "t_dat", "article_id",
            "price", "sales_channel_id",
            "product_type_name", "garment_group_name", "index_name", "section_name",
            "age", "club_member_status", "fashion_news_frequency",
            "article_popularity", "days_since_last_purchase"
        )
        .dropDuplicates(["customer_id", "t_dat", "article_id"])
        .join(label, ["customer_id", "t_dat"], "left")
        .fillna({"label": 0})
        .fillna({
            "product_type_name": "UNKNOWN",
            "garment_group_name": "UNKNOWN",
            "index_name": "UNKNOWN",
            "section_name": "UNKNOWN",
            "club_member_status": "UNKNOWN",
            "fashion_news_frequency": "UNKNOWN",
            "age": 0.0,
            "article_popularity": 0,
            "days_since_last_purchase": 0
        })
        .withColumn("year_month", F.date_format("t_dat", "yyyy-MM"))
    )

    # write
    gold.write.mode("overwrite").partitionBy("year_month").parquet(str(GOLD_FEATURES))
    txe.unpersist()
    print("[GOLD] Features saved.")


def temporal_split(df):
    # Use last month as test, previous half-month as val (simple temporal split)
    max_date = df.select(F.max("t_dat").alias("m")).collect()[0]["m"]
    # keep it simple: hard cutoffs known for this dataset (2020 range)
    train = df.filter(F.col("t_dat") <= F.lit("2020-08-31"))
    val = df.filter((F.col("t_dat") > F.lit("2020-08-31")) & (F.col("t_dat") <= F.lit("2020-09-15")))
    test = df.filter(F.col("t_dat") > F.lit("2020-09-15"))
    return train, val, test, max_date


# ----------------------------
# MLlib Training (3 algorithms + CV)
# ----------------------------
def train_mllib(spark):
    df = spark.read.parquet(str(GOLD_FEATURES))

    train_df, val_df, test_df, max_date = temporal_split(df)
    print(f"[SPLIT] max t_dat = {max_date}")

    # Persist for reuse
    train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)

    cat_cols = [
        "product_type_name", "garment_group_name", "index_name", "section_name",
        "club_member_status", "fashion_news_frequency"
    ]
    num_cols = ["price", "sales_channel_id", "age", "article_popularity", "days_since_last_purchase"]

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cat_cols],
        outputCols=[f"{c}_ohe" for c in cat_cols],
        handleInvalid="keep"
    )

    assembler = VectorAssembler(
        inputCols=[f"{c}_ohe" for c in cat_cols] + num_cols,
        outputCol="features"
    )

    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

    # 1) Logistic Regression + CrossValidator
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

    lr_pipe = Pipeline(stages=indexers + [encoder, assembler, lr])
    lr_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .build()
    )
    lr_cv = CrossValidator(
        estimator=lr_pipe,
        estimatorParamMaps=lr_grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=4
    )

    t0 = time.time()
    lr_cv_model = lr_cv.fit(train_df)
    lr_best = lr_cv_model.bestModel
    print(f"[LR] CV train seconds: {time.time() - t0:.1f}")
    lr_best.write().overwrite().save(str(MODELS / "lr_best"))

    lr_pred = lr_best.transform(test_df)
    lr_auc = evaluator.evaluate(lr_pred)

    # 2) Decision Tree
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=6)
    dt_pipe = Pipeline(stages=indexers + [encoder, assembler, dt])
    t0 = time.time()
    dt_model = dt_pipe.fit(train_df)
    print(f"[DT] train seconds: {time.time() - t0:.1f}")
    dt_model.write().overwrite().save(str(MODELS / "dt"))
    dt_auc = evaluator.evaluate(dt_model.transform(test_df))

    # 3) Linear SVC
    svm = LinearSVC(featuresCol="features", labelCol="label", maxIter=20, regParam=0.1)
    svm_pipe = Pipeline(stages=indexers + [encoder, assembler, svm])
    t0 = time.time()
    svm_model = svm_pipe.fit(train_df)
    print(f"[SVM] train seconds: {time.time() - t0:.1f}")
    svm_model.write().overwrite().save(str(MODELS / "svm"))

    # NOTE: evaluator uses rawPrediction; SVM has rawPrediction -> OK
    svm_auc = evaluator.evaluate(svm_model.transform(test_df))

    metrics = {
        "LR_AUC": float(lr_auc),
        "DT_AUC": float(dt_auc),
        "SVM_AUC": float(svm_auc),
        "label_window_days": LABEL_WINDOW_DAYS,
        "shuffle_partitions": SHUFFLE_PARTITIONS,
        "tx_write_partitions": TX_WRITE_PARTITIONS
    }
    (REPORTS / "mllib_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[MLLIB] Metrics saved:", metrics)

    train_df.unpersist()
    test_df.unpersist()
    return metrics


# ----------------------------
# sklearn baseline (sample)
# ----------------------------
def sklearn_baseline(spark):
    df = spark.read.parquet(str(GOLD_FEATURES))

    sample_pdf = (
        df.select(
            "product_type_name", "garment_group_name", "index_name", "section_name",
            "club_member_status", "fashion_news_frequency",
            "price", "sales_channel_id", "age", "article_popularity", "days_since_last_purchase",
            "label"
        )
        .sample(False, 0.005, seed=42)  # 0.5% sample
        .toPandas()
    )

    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder as SkOHE
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.linear_model import LogisticRegression as SkLR
    from sklearn.metrics import roc_auc_score

    y = sample_pdf["label"].astype(int).values
    X = sample_pdf.drop(columns=["label"])

    cat_cols = [
        "product_type_name", "garment_group_name", "index_name", "section_name",
        "club_member_status", "fashion_news_frequency"
    ]
    num_cols = ["price", "sales_channel_id", "age", "article_popularity", "days_since_last_purchase"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer([
        ("cat", SkOHE(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    clf = SkPipeline([
        ("pre", pre),
        ("lr", SkLR(max_iter=200))
    ])

    t0 = time.time()
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    with open(MODELS / "sklearn_lr.pkl", "wb") as f:
        pickle.dump(clf, f)

    out = {"sklearn_auc": float(auc), "train_seconds": float(time.time() - t0)}
    (REPORTS / "sklearn_metrics.json").write_text(json.dumps(out, indent=2))
    print("[SKLEARN] Baseline:", out)
    return out


# ----------------------------
# Tableau exports
# ----------------------------
def export_tableau(spark):
    df = spark.read.parquet(str(GOLD_FEATURES))

    # Dashboard 1: pipeline monitoring (rows by day)
    dq = df.groupBy("t_dat").agg(F.count("*").alias("rows"), F.sum("label").alias("label_ones"))
    dq.orderBy("t_dat").toPandas().to_csv(TABLEAU / "dq_rows_by_day.csv", index=False)

    # Dashboard 2: feature importance proxy (top product types where label=1)
    top = (
        df.filter(F.col("label") == 1)
          .groupBy("product_type_name")
          .agg(F.count("*").alias("repeat_within_window"))
          .orderBy(F.col("repeat_within_window").desc())
          .limit(100)
    )
    top.toPandas().to_csv(TABLEAU / "top_product_types_repeat.csv", index=False)

    # Dashboard 3: business insights by channel
    by_ch = (
        df.groupBy("sales_channel_id")
          .agg(F.count("*").alias("tx"), F.avg("price").alias("avg_price"), F.avg("label").alias("repeat_rate"))
          .orderBy("sales_channel_id")
    )
    by_ch.toPandas().to_csv(TABLEAU / "channel_insights.csv", index=False)

    print("[TABLEAU] Exports saved in:", TABLEAU)


def main():
    spark = create_spark()

    try:
        ingest_bronze(spark)
        build_silver(spark)
        build_gold(spark)

        mllib_metrics = train_mllib(spark)
        sklearn_metrics = sklearn_baseline(spark)
        export_tableau(spark)

        print("\n✅ FULL PIPELINE COMPLETE.")
        print("MLlib metrics:", mllib_metrics)
        print("Sklearn metrics:", sklearn_metrics)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
