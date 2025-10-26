#!/usr/bin/env python
"""
Analyze results from sweep stored in CSV files.

Usage:
    python analyze_sweep_results.py
    python analyze_sweep_results.py sweep_results.csv
    python analyze_sweep_results.py temp/sweep_results/
"""

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(csv_path="temp/sweep_results"):
    """Load sweep results from CSV file(s) or directory."""
    path = Path(csv_path)

    # If it's a directory, load all CSV files in it
    if path.is_dir():
        csv_files = list(path.glob("*.csv"))
        if not csv_files:
            print(f"Error: No CSV files found in directory: {csv_path}")
            print("Make sure to run the sweep first:")
            print("  python train.py +sweeps=dataset_size_sweep --multirun data.num_train_samples=10000,20000,30000,40000,50000")
            return None

        print(f"Found {len(csv_files)} CSV file(s) in {csv_path}")
        dfs = []
        for csv_file in sorted(csv_files):
            print(f"  Loading: {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                print(f"  Warning: Could not load {csv_file.name}: {e}")

        if not dfs:
            print("Error: No CSV files could be loaded")
            return None

        # Combine all dataframes
        df = pd.concat(dfs, ignore_index=True)
        print(f"\nCombined {len(dfs)} file(s) into {len(df)} total runs")
        return df

    # Otherwise treat as single CSV file
    elif path.is_file():
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"Error loading results: {e}")
            return None

    # Path doesn't exist
    else:
        print(f"Error: Path not found: {csv_path}")
        print("Make sure to run the sweep first:")
        print("  python train.py +sweeps=dataset_size_sweep --multirun data.num_train_samples=10000,20000,30000,40000,50000")
        return None

def print_summary(df):
    """Print summary of results."""
    print("=" * 80)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTotal runs: {len(df)}")
    print(f"\nColumns: {', '.join(df.columns.tolist())}")

    print("\n" + "=" * 80)
    print("ALL RESULTS")
    print("=" * 80)

    # Select key columns to display
    display_cols = ['num_train_samples', 'val_loss', 'val_exact_match', 'actual_epochs', 'learning_rate', 'batch_size']
    display_cols = [col for col in display_cols if col in df.columns]

    print(df[display_cols].to_string(index=False))

    # Find best result
    if 'val_loss' in df.columns:
        best_idx = df['val_loss'].idxmin()
        worst_idx = df['val_loss'].idxmax()

        print("\n" + "=" * 80)
        print("BEST RESULT")
        print("=" * 80)
        print(df.loc[best_idx][display_cols].to_string())

        print("\n" + "=" * 80)
        print("WORST RESULT")
        print("=" * 80)
        print(df.loc[worst_idx][display_cols].to_string())

        # Calculate improvement
        improvement = ((df.loc[worst_idx, 'val_loss'] - df.loc[best_idx, 'val_loss']) /
                      df.loc[worst_idx, 'val_loss']) * 100
        print(f"\nImprovement: {improvement:.2f}%")

def plot_results(df, save_path="sweep_analysis.png"):
    """Create visualizations."""
    if 'num_train_samples' not in df.columns or 'val_loss' not in df.columns:
        print("Cannot create plots: missing required columns")
        return

    # Sort by num_train_samples for proper plotting
    df_sorted = df.sort_values('num_train_samples')

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Val Loss vs Training Samples
    ax = axes[0, 0]
    ax.plot(df_sorted['num_train_samples'], df_sorted['val_loss'], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss vs Dataset Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add value labels
    for _, row in df_sorted.iterrows():
        if pd.notna(row['val_loss']):
            ax.annotate(f'{row["val_loss"]:.3f}',
                       (row['num_train_samples'], row['val_loss']),
                       textcoords="offset points", xytext=(0,10),
                       ha='center', fontsize=9)

    # Plot 2: Val Exact Match vs Training Samples (if available)
    ax = axes[0, 1]
    if 'val_exact_match' in df.columns and df['val_exact_match'].notna().any():
        ax.plot(df_sorted['num_train_samples'], df_sorted['val_exact_match'], 'o-',
               color='green', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Training Samples', fontsize=12)
        ax.set_ylabel('Validation Exact Match', fontsize=12)
        ax.set_title('Exact Match Accuracy vs Dataset Size', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No exact match data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Exact Match (N/A)', fontsize=14)

    # Plot 3: Training time vs Dataset Size (if available)
    ax = axes[1, 0]
    if 'actual_epochs' in df.columns:
        ax.bar(df_sorted['num_train_samples'], df_sorted['actual_epochs'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Number of Training Samples', fontsize=12)
        ax.set_ylabel('Epochs Until Convergence', fontsize=12)
        ax.set_title('Training Duration vs Dataset Size', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No epoch data', ha='center', va='center', transform=ax.transAxes)

    # Plot 4: Improvement from baseline
    ax = axes[1, 1]
    baseline_loss = df_sorted.iloc[0]['val_loss']
    improvements = ((baseline_loss - df_sorted['val_loss']) / baseline_loss) * 100

    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax.bar(df_sorted['num_train_samples'], improvements, color=colors, alpha=0.7)
    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('Improvement from Baseline (%)', fontsize=12)
    ax.set_title('Relative Improvement vs Dataset Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Add value labels
    for i, (samples, improvement) in enumerate(zip(df_sorted['num_train_samples'], improvements)):
        ax.text(samples, improvement, f'{improvement:.1f}%',
               ha='center', va='bottom' if improvement > 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plots saved to: {save_path}")
    plt.show()

def analyze_efficiency(df):
    """Analyze data efficiency."""
    if 'num_train_samples' not in df.columns or 'val_loss' not in df.columns:
        return

    df_sorted = df.sort_values('num_train_samples')

    print("\n" + "=" * 80)
    print("DATA EFFICIENCY ANALYSIS")
    print("=" * 80)

    for i in range(1, len(df_sorted)):
        prev = df_sorted.iloc[i-1]
        curr = df_sorted.iloc[i]

        samples_increase = curr['num_train_samples'] - prev['num_train_samples']
        loss_decrease = prev['val_loss'] - curr['val_loss']

        if samples_increase > 0:
            efficiency = (loss_decrease / samples_increase) * 10000
            print(f"\n{int(prev['num_train_samples']):,} → {int(curr['num_train_samples']):,} samples:")
            print(f"  Loss change: {loss_decrease:+.4f}")
            print(f"  Efficiency: {efficiency:.6f} loss decrease per 10k samples")

            if loss_decrease < 0:
                print(f"  ⚠️  Warning: Loss increased!")

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "temp/sweep_results/data_sizes.csv"

    # Load results
    df = load_results(csv_path)
    if df is None:
        return

    # Print summary
    print_summary(df)

    # Analyze efficiency
    analyze_efficiency(df)

    # Create plots
    plot_results(df)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
