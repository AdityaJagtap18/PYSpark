# H&M Fashion Recommendation

A machine learning pipeline for predicting customer purchase behavior using H&M transaction data.

## Features

This project includes **two ML pipelines**:

1. **PySpark** - Recommended for large datasets
2. **Scikit-learn** - Faster setup for smaller datasets

Both pipelines train models and create Tableau-ready visualizations.

## Prerequisites

### Required Software

1. **Python 3.8+** - [Download](https://www.python.org/downloads/)
2. **Java 8 or 11** (PySpark only) - [Download](https://adoptium.net/)

### Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** On Linux/Mac, use `python3` instead of `python`

## Data Setup

Place your CSV files in `data/hnm/raw/`:

```
data/hnm/raw/
├── articles.csv
├── customers.csv
└── transactions_train.csv
```

## Quick Start

### Option A: PySpark Pipeline (Recommended)

```bash
# Full pipeline
python pyspark_data_ingestion.py && \
python pyspark_feature_engineering.py && \
python pyspark_model_training.py && \
python pyspark_model_evaluation.py && \
python pyspark_tableau_prep.py
```

### Option B: Scikit-learn Pipeline

```bash
# Full pipeline
python run_training.py && \
python evaluate_models.py && \
python prepare_tableau_data.py
```

## Output Locations

| Pipeline | Models | Evaluation | Tableau Data |
|----------|--------|------------|--------------|
| PySpark | `data/hnm/pyspark_models/` | `data/hnm/pyspark_evaluation/` | `data/hnm/pyspark_tableau/` |
| Sklearn | `data/hnm/models/` | `data/hnm/evaluation/` | `data/hnm/tableau/` |

## Models Trained

- **PySpark**: 5 models (Linear Regression, Decision Tree, Random Forest, GBT, Generalized Linear)
- **Sklearn**: 4 models (Linear, Ridge, Random Forest, Gradient Boosting)

## Jupyter Notebooks

For interactive exploration:

```bash
jupyter notebook model_training.ipynb
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python not found` | Reinstall Python with "Add to PATH" checked |
| `JAVA_HOME not set` | Set JAVA_HOME environment variable |
| Out of memory | Reduce sample size in `pyspark_data_ingestion.py` |
| Module not found | Run `pip install -r requirements.txt` |

## Project Structure

```
PYSpark/
├── data/hnm/raw/              # Input CSV files
├── pyspark_*.py               # PySpark pipeline scripts
├── run_training.py            # Sklearn training
├── evaluate_models.py         # Sklearn evaluation
├── prepare_tableau_data.py    # Sklearn Tableau prep
├── *.ipynb                    # Jupyter notebooks
├── requirements.txt           # Python dependencies
└── *.bat / *.sh               # Windows/Linux runners
```

## License

This project is for educational purposes.
