#!/usr/bin/env python3
"""
Scoring Function Impact Analysis (Presence vs Absence)

Generates horizontal bar charts showing the median performance difference
when a scoring function is present vs absent in workflows.

Performance difference is calculated as: median(with SF) - median(without SF)

Analyzes workflows at different percentile cutoffs:
- All Workflows
- Top 1%
- Top 0.1%

Scoring functions are color-coded by category:
- ML (blue): RTMScore, NNScore, CNN-Affinity, RFScoreVS, GenScore variants, CNN-Score
- Empirical (green): AD4, LinF9, CHEMPLP, Vinardo, GNINA-Affinity
- Knowledge (red): KORP-PL, ConvexPLR
- Other (gray): Uncategorized functions

Usage:
    python sf_impact.py --base-path PATH --output-dir PATH
                        --metrics ref,ef --thresholds 0p1,1
"""

import itertools
import traceback
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    DATASETS, DATASET_DIRS, METRICS, THRESHOLDS, THRESH_MAP,
    get_output_dir, get_aggregated_dir, get_sf_color, parse_list_arg, setup_argument_parser,
    load_metric_data, calculate_overall_performance, get_top_workflows_pd
)
from .plot_helpers import setup_plot_style, handle_empty_subplot


PERCENTILES = [100.0, 1.0, 0.1]
PERCENTILE_LABELS = ["All Workflows", "Top 1%", "Top 0.1%"]


def generate_sf_impact_plots(
    aggregated_dir: Path,
    output_dir: Path,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None
) -> int:
    """
    Generate scoring function impact (presence vs absence) plots.

    Args:
        aggregated_dir: Path to aggregated parquet directory
        output_dir: Output directory for plots
        metrics: List of metrics to process
        thresholds: List of thresholds to process
        datasets: List of datasets to process

    Returns:
        Number of plots generated
    """
    if metrics is None:
        metrics = METRICS
    if thresholds is None:
        thresholds = THRESHOLDS
    if datasets is None:
        datasets = DATASETS

    setup_plot_style()
    print("Generating Scoring Function Impact (Presence vs. Absence) Plots across Percentiles...")
    analysis_type_folder = "scoring_function_impact_pvsa_percentiles"
    plots_generated = 0

    for metric, threshold in itertools.product(metrics, thresholds):
        print(f"\n--- Metric: {metric.upper()}, Threshold: {THRESH_MAP.get(threshold, threshold)} ---")

        loaded_data = {}
        overall_sfs_found = set()
        data_found = False
        print("Loading data...")

        for ds_name in datasets:
            if ds_name not in DATASET_DIRS:
                print(f"   Warning: Unknown dataset {ds_name}. Skipping.")
                continue

            df_raw = load_metric_data(aggregated_dir, ds_name, metric, threshold)

            if df_raw is None:
                print(f"   Warning: File not found for {ds_name}. Skipping.")
                loaded_data[ds_name] = None
                continue

            try:
                df_median = calculate_overall_performance(df_raw.copy(), use_median=True)

                if 'scoring' not in df_median.columns:
                    raise ValueError("'scoring' column not found in DataFrame.")
                if 'overall_performance' not in df_median.columns:
                    raise ValueError("'overall_performance' column not found after calculation.")

                df_median = df_median.rename(columns={'overall_performance': 'overall_performance_median'})
                loaded_data[ds_name] = df_median

                if not df_median.empty and 'scoring' in df_median.columns:
                    sf_series = df_median['scoring'].dropna().astype(str)
                    for sf_combined in sf_series:
                        if sf_combined:
                            individual_sfs = [s.strip() for s in sf_combined.split('+')]
                            overall_sfs_found.update(individual_sfs)

                print(f"   Loaded {ds_name} ({len(df_raw)} rows)")
                data_found = True
            except Exception as e:
                print(f"   Error loading or processing file for {ds_name}: {e}")
                traceback.print_exc()
                loaded_data[ds_name] = None

        if not data_found:
            print("   No data found for any dataset. Skipping plots for this metric/threshold.")
            continue

        if not overall_sfs_found:
            print("   No scoring functions found across loaded datasets. Skipping plots.")
            continue
        print(f"   Found {len(overall_sfs_found)} unique scoring functions overall for this metric/threshold.")

        ranking_col = 'overall_performance_median'

        output_subdir = output_dir / analysis_type_folder / "median_ranking_performance_difference_percentiles" / metric
        output_subdir.mkdir(parents=True, exist_ok=True)

        num_datasets = len(datasets)
        num_percentiles = len(PERCENTILES)
        fig, axes = plt.subplots(num_percentiles, num_datasets,
                                 figsize=(5 * num_datasets, 4 * num_percentiles),
                                 sharex=False, sharey=False)

        fig.suptitle(f'Scoring Function Impact (Median Performance Difference: Present - Absent)\n({metric.upper()} @ {THRESH_MAP.get(threshold, threshold)})',
                     fontsize=14, weight='bold')

        if num_percentiles == 1 and num_datasets == 1:
            axes = np.array([[axes]])
        elif num_percentiles == 1:
            axes = axes.reshape(1, num_datasets)
        elif num_datasets == 1:
            axes = axes.reshape(num_percentiles, 1)

        for j, ds_name in enumerate(datasets):
            df_full = loaded_data.get(ds_name)

            if df_full is None or df_full.empty:
                print(f"   No data for {ds_name}, skipping column.")
                for i in range(num_percentiles):
                    ax = axes[i, j]
                    handle_empty_subplot(ax, f"{ds_name}\n({PERCENTILE_LABELS[i]})")
                continue

            for i, percentile in enumerate(PERCENTILES):
                ax = axes[i, j]
                percentile_label = PERCENTILE_LABELS[i]
                plot_title = f"{ds_name}\n({percentile_label})"

                df_subset = get_top_workflows_pd(df_full, percentile, ranking_col=ranking_col)

                if df_subset.empty:
                    print(f"     No data for {ds_name} at {percentile_label}, skipping subplot.")
                    handle_empty_subplot(ax, plot_title, message=f'No data\n({percentile_label})')
                    continue

                subset_sfs = set()
                sf_series_subset = df_subset['scoring'].dropna().astype(str)
                for sf_combined in sf_series_subset:
                    if sf_combined:
                        individual_sfs = sf_combined.split('+')
                        subset_sfs.update([s.strip() for s in individual_sfs])

                if not subset_sfs:
                    print(f"     No scoring functions identified in the subset for {ds_name} at {percentile_label}. Skipping subplot.")
                    handle_empty_subplot(ax, plot_title, message='No SFs in subset')
                    continue

                performance_diffs = {}
                for sf in sorted(list(subset_sfs)):
                    mask_with_sf = df_subset['scoring'].astype(str).str.contains(sf, regex=False)
                    df_with_sf_subset = df_subset[mask_with_sf]
                    df_without_sf_subset = df_subset[~mask_with_sf]

                    median_perf_with = df_with_sf_subset[ranking_col].median()
                    median_perf_without = df_without_sf_subset[ranking_col].median()

                    if df_with_sf_subset.empty or pd.isna(median_perf_with):
                        diff = 0
                    elif df_without_sf_subset.empty or pd.isna(median_perf_without):
                        diff = 0
                    else:
                        diff = median_perf_with - median_perf_without

                    performance_diffs[sf] = diff

                sf_diff_df = pd.DataFrame(list(performance_diffs.items()), columns=['scoring_function', 'performance_difference'])
                if sf_diff_df.empty:
                    print(f"     No performance differences calculated for {ds_name} at {percentile_label}, skipping subplot.")
                    handle_empty_subplot(ax, plot_title, message='Calc Error')
                    continue

                sf_diff_df = sf_diff_df.sort_values('performance_difference', ascending=True)

                colors = [get_sf_color(sf) for sf in sf_diff_df['scoring_function']]

                ax.barh(sf_diff_df['scoring_function'], sf_diff_df['performance_difference'],
                        color=colors, edgecolor='black', linewidth=0.5)

                ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7, zorder=5)

                ax.set_title(plot_title, fontsize=10, weight='bold')

                if i == num_percentiles - 1:
                    ax.set_xlabel("Median Perf. Diff.\n(Present - Absent)", fontsize=9)
                else:
                    ax.tick_params(axis='x', labelbottom=False)

                if j == 0:
                    ax.set_ylabel("Scoring Function", fontsize=9)

                ax.tick_params(axis='y', labelsize=7)
                ax.tick_params(axis='x', labelsize=7)

                ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_subdir / f"sf_impact_pvsa_percentiles_thresh{threshold}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        plots_generated += 1
        print(f"   Plot saved for threshold {THRESH_MAP.get(threshold, threshold)}")

    print("\nScoring Function Impact (Presence vs. Absence) across Percentiles plots generated.")
    return plots_generated


def main():
    parser = setup_argument_parser(description=__doc__)
    args = parser.parse_args()

    output_dir = get_output_dir(args.output_dir)
    aggregated_dir = get_aggregated_dir(output_dir)

    metrics = parse_list_arg(args.metrics, METRICS)
    thresholds = parse_list_arg(args.thresholds, THRESHOLDS)
    datasets = parse_list_arg(args.datasets, DATASETS)

    print(f"Aggregated Dir: {aggregated_dir.resolve()}")
    print(f"Output Directory: {output_dir.resolve()}")
    print(f"Metrics: {metrics}")
    print(f"Thresholds: {thresholds}")
    print(f"Datasets: {datasets}")

    output_dir.mkdir(parents=True, exist_ok=True)

    generate_sf_impact_plots(
        aggregated_dir=aggregated_dir,
        output_dir=output_dir,
        metrics=metrics,
        thresholds=thresholds,
        datasets=datasets
    )


if __name__ == "__main__":
    main()
