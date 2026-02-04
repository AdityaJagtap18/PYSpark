#!/bin/bash
# PySpark Pipeline Runner
# Runs data ingestion and feature engineering in sequence

echo "=========================================="
echo "PySpark Data Pipeline"
echo "=========================================="
echo ""

# Check if PySpark is installed
if ! command -v spark-submit &> /dev/null; then
    echo "❌ spark-submit not found!"
    echo "Running with python3 instead (local mode)..."
    RUNNER="python3"
else
    echo "✓ Using spark-submit"
    RUNNER="spark-submit"
fi

echo ""
echo "=========================================="
echo "STEP 1: Data Ingestion"
echo "=========================================="
$RUNNER pyspark_data_ingestion.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Data ingestion failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "STEP 2: Feature Engineering"
echo "=========================================="
$RUNNER pyspark_feature_engineering.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Feature engineering failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Output locations:"
echo "  - Staging data:  data/hnm/staging/"
echo "  - Features:      data/hnm/pyspark_features/"
echo ""
