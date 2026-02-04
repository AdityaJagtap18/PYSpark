#!/usr/bin/env python3
"""
Prepare data in Tableau-friendly format for visualization
"""
import pandas as pd
from pathlib import Path

print("Reading model metrics...")
metrics_df = pd.read_csv("data/hnm/evaluation/model_metrics_summary.csv")

# Create output directory
output_dir = Path("data/hnm/tableau")
output_dir.mkdir(exist_ok=True)

# 1. Long format for metrics comparison
print("\nCreating long-format metrics file...")
metrics_long = []

for _, row in metrics_df.iterrows():
    model = row['model']

    # Training metrics
    metrics_long.append({
        'model': model,
        'dataset': 'Training',
        'metric_name': 'RMSE',
        'metric_value': row['train_rmse']
    })
    metrics_long.append({
        'model': model,
        'dataset': 'Training',
        'metric_name': 'MAE',
        'metric_value': row['train_mae']
    })
    metrics_long.append({
        'model': model,
        'dataset': 'Training',
        'metric_name': 'R²',
        'metric_value': row['train_r2']
    })

    # Test metrics
    metrics_long.append({
        'model': model,
        'dataset': 'Test',
        'metric_name': 'RMSE',
        'metric_value': row['test_rmse']
    })
    metrics_long.append({
        'model': model,
        'dataset': 'Test',
        'metric_name': 'MAE',
        'metric_value': row['test_mae']
    })
    metrics_long.append({
        'model': model,
        'dataset': 'Test',
        'metric_name': 'R²',
        'metric_value': row['test_r2']
    })

metrics_long_df = pd.DataFrame(metrics_long)
metrics_long_df.to_csv(output_dir / "metrics_long_format.csv", index=False)
print(f"✓ Saved: {output_dir / 'metrics_long_format.csv'}")

# 2. Model comparison with categories
print("\nCreating model comparison file...")
comparison_df = metrics_df.copy()

# Add model type (fast vs full)
comparison_df['model_type'] = comparison_df['model'].apply(
    lambda x: 'Fast (10% data)' if 'fast' in x else 'Full (100% data)'
)

# Add model family
def get_model_family(name):
    if 'linear' in name.lower():
        return 'Linear Regression'
    elif 'ridge' in name.lower():
        return 'Ridge Regression'
    elif 'random_forest' in name.lower():
        return 'Random Forest'
    elif 'gradient' in name.lower():
        return 'Gradient Boosting'
    return 'Other'

comparison_df['model_family'] = comparison_df['model'].apply(get_model_family)

# Calculate performance metrics
comparison_df['test_vs_train_rmse_diff'] = comparison_df['test_rmse'] - comparison_df['train_rmse']
comparison_df['is_overfitting'] = comparison_df['test_vs_train_rmse_diff'] > 0.01

comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
print(f"✓ Saved: {output_dir / 'model_comparison.csv'}")

# 3. Create a summary stats file
print("\nCreating summary statistics...")
summary_stats = []

for model_family in comparison_df['model_family'].unique():
    family_data = comparison_df[comparison_df['model_family'] == model_family]

    summary_stats.append({
        'model_family': model_family,
        'num_models': len(family_data),
        'avg_test_rmse': family_data['test_rmse'].mean(),
        'min_test_rmse': family_data['test_rmse'].min(),
        'max_test_rmse': family_data['test_rmse'].max(),
        'avg_test_r2': family_data['test_r2'].mean(),
        'best_model': family_data.loc[family_data['test_rmse'].idxmin(), 'model']
    })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(output_dir / "model_family_summary.csv", index=False)
print(f"✓ Saved: {output_dir / 'model_family_summary.csv'}")

# 4. Aggregate predictions for sample analysis
print("\nAggregating prediction samples...")
pred_dir = Path("data/hnm/evaluation/predictions")
all_predictions = []

for pred_file in pred_dir.glob("*_predictions.csv"):
    model_name = pred_file.stem.replace('_predictions', '')

    # Read and sample predictions (first 10000 rows for Tableau performance)
    pred_df = pd.read_csv(pred_file, nrows=10000)
    pred_df['model'] = model_name
    all_predictions.append(pred_df)

if all_predictions:
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    combined_predictions.to_csv(output_dir / "predictions_sample.csv", index=False)
    print(f"✓ Saved: {output_dir / 'predictions_sample.csv'} ({len(combined_predictions):,} rows)")

# 5. Error distribution summary
print("\nCreating error distribution summary...")
error_summary = []

for pred_file in pred_dir.glob("*_predictions.csv"):
    model_name = pred_file.stem.replace('_predictions', '')
    pred_df = pd.read_csv(pred_file)

    # Calculate error buckets
    error_summary.append({
        'model': model_name,
        'error_bucket': 'Perfect (error = 0)',
        'count': len(pred_df[pred_df['abs_error'] == 0]),
        'percentage': len(pred_df[pred_df['abs_error'] == 0]) / len(pred_df) * 100
    })
    error_summary.append({
        'model': model_name,
        'error_bucket': 'Small (0 < error ≤ 0.5)',
        'count': len(pred_df[(pred_df['abs_error'] > 0) & (pred_df['abs_error'] <= 0.5)]),
        'percentage': len(pred_df[(pred_df['abs_error'] > 0) & (pred_df['abs_error'] <= 0.5)]) / len(pred_df) * 100
    })
    error_summary.append({
        'model': model_name,
        'error_bucket': 'Medium (0.5 < error ≤ 1)',
        'count': len(pred_df[(pred_df['abs_error'] > 0.5) & (pred_df['abs_error'] <= 1)]),
        'percentage': len(pred_df[(pred_df['abs_error'] > 0.5) & (pred_df['abs_error'] <= 1)]) / len(pred_df) * 100
    })
    error_summary.append({
        'model': model_name,
        'error_bucket': 'Large (error > 1)',
        'count': len(pred_df[pred_df['abs_error'] > 1]),
        'percentage': len(pred_df[pred_df['abs_error'] > 1]) / len(pred_df) * 100
    })

error_summary_df = pd.DataFrame(error_summary)
error_summary_df.to_csv(output_dir / "error_distribution.csv", index=False)
print(f"✓ Saved: {output_dir / 'error_distribution.csv'}")

# Print summary
print("\n" + "="*60)
print("TABLEAU DATA FILES CREATED")
print("="*60)
print(f"\nAll files saved in: {output_dir}/")
print("\nFiles created:")
print("  1. metrics_long_format.csv - For metric comparisons across train/test")
print("  2. model_comparison.csv - Enhanced metrics with categories")
print("  3. model_family_summary.csv - Aggregated statistics by model type")
print("  4. predictions_sample.csv - Sample predictions (10K rows per model)")
print("  5. error_distribution.csv - Error bucket analysis")

print("\n" + "="*60)
print("SUGGESTED TABLEAU VISUALIZATIONS")
print("="*60)
print("""
1. Model Performance Dashboard:
   - Bar chart: Test RMSE by model (sorted)
   - Side-by-side bars: Train vs Test RMSE (color by dataset)
   - Scatter plot: Test RMSE vs Test R²

2. Overfitting Analysis:
   - Bullet chart: Train RMSE vs Test RMSE difference
   - Highlight table: Show overfitting models in red

3. Error Distribution:
   - Stacked bar chart: Error buckets by model
   - Pie charts: Error distribution for best model

4. Predictions Analysis:
   - Scatter plot: Actual vs Predicted
   - Histogram: Error distribution
   - Box plot: Absolute error by model

5. Model Family Comparison:
   - Bar chart: Average test RMSE by model family
   - Heat map: Metric comparison matrix
""")

print("\n✓ Data ready for Tableau!")
