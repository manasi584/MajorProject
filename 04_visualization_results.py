# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy>=2.0.2",
#     "matplotlib>=3.8.0",
#     "seaborn>=0.13.0",
#     "scikit-learn>=1.6.1",
#     "pandas>=2.3.3",
# ]
# ///

# ==========================================
# STEP 4 - VISUALIZATION & RESULTS ANALYSIS
# ==========================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import re

# ==========================================
# CONFIGURATION
# ==========================================

RESULTS_DIR = "./results"
FIGURES_DIR = "./results/figures"
TABLES_DIR = "./results/tables"

# Create output directories
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Class labels
CLASS_NAMES = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]

# ==========================================
# UTILITIES
# ==========================================

def parse_log_file(filepath):
    """Extract metrics from log file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Extract K-FOLD CV Accuracy
        cv_match = re.search(r'K-FOLD CV Accuracy:\s+(\d+\.\d+)', content)
        cv_acc = float(cv_match.group(1)) if cv_match else None

        # Extract TEST ACCURACY
        test_match = re.search(r'TEST ACCURACY:\s+(\d+\.\d+)', content)
        test_acc = float(test_match.group(1)) if test_match else None

        return cv_acc, test_acc
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None, None

def extract_config(filename):
    """Extract feature and selection method from filename"""
    # Format: features_xxx_method.txt
    parts = filename.replace('.txt', '').split('_')
    feature = '_'.join(parts[:-1])  # e.g., features_mobilenet
    method = parts[-1]  # e.g., none, quantum_puma
    return feature, method

# ==========================================
# RESULTS SUMMARY TABLE
# ==========================================

def create_results_summary():
    """Create summary table of all runs"""
    print("\n" + "="*60)
    print("GENERATING RESULTS SUMMARY TABLE")
    print("="*60)

    log_files = glob.glob(os.path.join(RESULTS_DIR, "*.txt"))

    results = []
    for log_file in sorted(log_files):
        filename = os.path.basename(log_file)
        feature, method = extract_config(filename)
        cv_acc, test_acc = parse_log_file(log_file)

        if cv_acc is not None and test_acc is not None:
            results.append({
                'Feature Extractor': feature.replace('features_', ''),
                'Feature Selection': method,
                'CV Accuracy (%)': round(cv_acc, 2),
                'Test Accuracy (%)': round(test_acc, 2),
                'Improvement': round(test_acc - cv_acc, 2)
            })

    df = pd.DataFrame(results)

    # Save as CSV
    csv_path = os.path.join(TABLES_DIR, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Summary table saved: {csv_path}\n")
    print(df.to_string(index=False))

    return df

# ==========================================
# COMPARISON PLOTS
# ==========================================

def plot_accuracy_comparison(df):
    """Plot accuracy comparison across features and methods"""
    print("\nGenerating accuracy comparison plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By Feature Extractor
    feature_acc = df.groupby('Feature Extractor')[['CV Accuracy (%)', 'Test Accuracy (%)']].mean()
    feature_acc.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
    axes[0].set_title('Accuracy by Feature Extractor', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_xlabel('Feature Extractor')
    axes[0].set_ylim([90, 100])
    axes[0].legend(['CV Accuracy', 'Test Accuracy'])
    axes[0].grid(axis='y', alpha=0.3)

    # By Feature Selection Method
    method_acc = df.groupby('Feature Selection')[['CV Accuracy (%)', 'Test Accuracy (%)']].mean()
    method_acc.plot(kind='bar', ax=axes[1], color=['#3498db', '#e74c3c'])
    axes[1].set_title('Accuracy by Feature Selection Method', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_xlabel('Feature Selection')
    axes[1].set_ylim([90, 100])
    axes[1].legend(['CV Accuracy', 'Test Accuracy'])
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "accuracy_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()

def plot_heatmap_accuracy(df):
    """Create heatmap of accuracies"""
    print("Generating accuracy heatmap...")

    # Pivot for heatmap
    pivot = df.pivot_table(
        values='Test Accuracy (%)',
        index='Feature Extractor',
        columns='Feature Selection'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=90, vmax=100,
                cbar_kws={'label': 'Test Accuracy (%)'}, ax=ax, linewidths=0.5)
    ax.set_title('Test Accuracy Heatmap\n(Feature Extractor vs Selection Method)',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "accuracy_heatmap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()

# ==========================================
# PERFORMANCE RANKING
# ==========================================

def create_performance_ranking(df):
    """Create ranking table of best configurations"""
    print("Generating performance ranking...")

    ranked = df.sort_values('Test Accuracy (%)', ascending=False).reset_index(drop=True)
    ranked['Rank'] = range(1, len(ranked) + 1)

    # Reorder columns
    ranked = ranked[['Rank', 'Feature Extractor', 'Feature Selection',
                     'CV Accuracy (%)', 'Test Accuracy (%)', 'Improvement']]

    # Save as CSV
    csv_path = os.path.join(TABLES_DIR, "performance_ranking.csv")
    ranked.to_csv(csv_path, index=False)
    print(f"✓ Ranking saved: {csv_path}\n")
    print(ranked.to_string(index=False))

    return ranked

# ==========================================
# DETAILED STATISTICS
# ==========================================

def create_statistics_summary(df):
    """Create statistical summary"""
    print("\nGenerating statistics summary...")

    stats = {
        'Metric': [
            'Best Test Accuracy',
            'Worst Test Accuracy',
            'Mean Test Accuracy',
            'Std Dev Test Accuracy',
            'Best CV Accuracy',
            'Worst CV Accuracy',
            'Mean Improvement (Test - CV)'
        ],
        'Value': [
            f"{df['Test Accuracy (%)'].max():.2f}%",
            f"{df['Test Accuracy (%)'].min():.2f}%",
            f"{df['Test Accuracy (%)'].mean():.2f}%",
            f"{df['Test Accuracy (%)'].std():.2f}%",
            f"{df['CV Accuracy (%)'].max():.2f}%",
            f"{df['CV Accuracy (%)'].min():.2f}%",
            f"{df['Improvement'].mean():.2f}%"
        ]
    }

    stats_df = pd.DataFrame(stats)

    # Save as CSV
    csv_path = os.path.join(TABLES_DIR, "statistics_summary.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"✓ Statistics saved: {csv_path}\n")
    print(stats_df.to_string(index=False))

    return stats_df

# ==========================================
# SUMMARY REPORT
# ==========================================

def create_summary_report(df, ranked, stats):
    """Create text report"""
    print("Generating summary report...")

    best_config = ranked.iloc[0]
    worst_config = ranked.iloc[-1]

    report = f"""
{'='*70}
EXPERIMENT RESULTS SUMMARY REPORT
{'='*70}

BEST CONFIGURATION:
  Feature Extractor: {best_config['Feature Extractor']}
  Feature Selection: {best_config['Feature Selection']}
  CV Accuracy:      {best_config['CV Accuracy (%)']}%
  Test Accuracy:    {best_config['Test Accuracy (%)']}%
  Improvement:      {best_config['Improvement']}%

WORST CONFIGURATION:
  Feature Extractor: {worst_config['Feature Extractor']}
  Feature Selection: {worst_config['Feature Selection']}
  CV Accuracy:      {worst_config['CV Accuracy (%)']}%
  Test Accuracy:    {worst_config['Test Accuracy (%)']}%

OVERALL STATISTICS:
  Number of Configurations: {len(df)}
  Best Test Accuracy:       {df['Test Accuracy (%)'].max():.2f}%
  Worst Test Accuracy:      {df['Test Accuracy (%)'].min():.2f}%
  Mean Test Accuracy:       {df['Test Accuracy (%)'].mean():.2f}%
  Std Dev:                  {df['Test Accuracy (%)'].std():.2f}%

FEATURE EXTRACTORS TESTED:
{df['Feature Extractor'].unique().tolist()}

FEATURE SELECTION METHODS TESTED:
{df['Feature Selection'].unique().tolist()}

{'='*70}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""

    # Save report
    report_path = os.path.join(TABLES_DIR, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"✓ Report saved: {report_path}")
    print(report)

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STEP 4 - VISUALIZATION & RESULTS ANALYSIS")
    print("="*60)

    # Check if results exist
    log_files = glob.glob(os.path.join(RESULTS_DIR, "*.txt"))
    if not log_files:
        print("\n⚠ No result files found in ./results/")
        print("Run step 03 (03_kfold_evaluation.py) first to generate results.")
        exit(1)

    # Create results summary
    df = create_results_summary()

    if len(df) == 0:
        print("⚠ No valid results found.")
        exit(1)

    # Generate plots
    plot_accuracy_comparison(df)
    plot_heatmap_accuracy(df)

    # Create ranking
    ranked = create_performance_ranking(df)

    # Create statistics
    stats = create_statistics_summary(df)

    # Create summary report
    create_summary_report(df, ranked, stats)

    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS GENERATED")
    print("="*60)
    print(f"\nFigures saved to:  {FIGURES_DIR}/")
    print(f"Tables saved to:   {TABLES_DIR}/")
    print("\n")
