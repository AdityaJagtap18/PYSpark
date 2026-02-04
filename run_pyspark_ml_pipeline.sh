#!/bin/bash
# PySpark ML Pipeline Runner
# Trains models, evaluates them, and prepares Tableau data

echo "=========================================="
echo "PySpark ML Pipeline"
echo "=========================================="
echo ""

# Check if PySpark is available
if ! command -v spark-submit &> /dev/null; then
    echo "Running with python3 (local mode)..."
    RUNNER="python3"
else
    echo "✓ Using spark-submit"
    RUNNER="spark-submit"
fi

echo ""
echo "=========================================="
echo "STEP 1: Model Training"
echo "=========================================="
$RUNNER pyspark_model_training.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Model training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "STEP 2: Model Evaluation"
echo "=========================================="
$RUNNER pyspark_model_evaluation.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Model evaluation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "STEP 3: Tableau Data Preparation"
echo "=========================================="
$RUNNER pyspark_tableau_prep.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Tableau prep failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ ML PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Output locations:"
echo "  - Models:        data/hnm/pyspark_models/"
echo "  - Evaluation:    data/hnm/pyspark_evaluation/"
echo "  - Tableau Data:  data/hnm/pyspark_tableau/"
echo ""
