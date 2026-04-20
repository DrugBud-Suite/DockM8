#!/usr/bin/env python3
"""
Scoring Function Frequency Analysis

Generates horizontal bar charts showing how frequently each scoring function
appears in workflows at different percentile cutoffs (Top 1%, Top 0.1%, Top 0.01%).

Frequency is calculated as: (subset count / full count) × 100%

Scoring functions are color-coded by category:
- ML (blue): RTMScore, NNScore, CNN-Affinity, RFScoreVS, GenScore variants, CNN-Score
- Empirical (green): AD4, LinF9, CHEMPLP, Vinardo, GNINA-Affinity
- Knowledge (red): KORP-PL, ConvexPLR
- Other (gray): Uncategorized functions

Usage:
    python sf_frequency.py --base-path PATH --output-dir PATH
                           --metrics ref,ef --thresholds 0p1,1
"""

import itertools
import traceback
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .config import (
    DATASETS, DATASET_DIRS, METRICS, THRESHOLDS, THRESH_MAP,
    get_output_dir, get_aggregated_dir, get_sf_color, parse_list_arg, setup_argument_parser,
    load_metric_data, calculate_overall_performance, get_top_workflows_pd
)
from .plot_helpers import setup_plot_style, reshape_axes, handle_empty_subplot


PERCENTILES = [0.1, 0.01]
PLOT_LABELS = ["Top 0.1%", "Top 0.01%"]


def generate_sf_frequency_plots(
    aggregated_dir: Path,
    output_dir: Path,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None
) -> int:
    """
    Generate scoring function frequency plots.

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
    print("Generating Scoring Function Frequency Plots...")
    analysis_type_folder = "scoring_function_frequency"
    plots_generated = 0

    for metric, threshold in itertools.product(metrics, thresholds):
        print(f"\n--- Metric: {metric.upper()}, Threshold: {THRESH_MAP.get(threshold, threshold)} ---")

        loaded_data = {}
        all_scoring_functions = set()
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

                loaded_data[ds_name] = df_median.rename(columns={'overall_performance': 'overall_performance_median'})

                if not df_median.empty and 'scoring' in df_median.columns:
                    sf_series = df_median['scoring'].dropna()
                    for sf_combined in sf_series:
                        if pd.notna(sf_combined) and sf_combined:
                            individual_sfs = sf_combined.split('+')
                            all_scoring_functions.update(individual_sfs)

                print(f"   Loaded {ds_name} ({len(df_raw)} rows)")
                data_found = True
            except Exception as e:
                print(f"   Error loading or processing file for {ds_name}: {e}")
                traceback.print_exc()
                loaded_data[ds_name] = None

        if not data_found:
            print("   No data found for any dataset. Skipping plots for this metric/threshold.")
            continue

        all_scoring_functions = sorted(list(all_scoring_functions))
        print(f"   Found {len(all_scoring_functions)} unique scoring functions")

        ranking_col = 'overall_performance_median'
        print(f"   Generating Plot: Rank=median, Content=frequency")

        output_subdir = output_dir / analysis_type_folder / "median_ranking_frequency" / metric
        output_subdir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, len(datasets), figsize=(4.5 * len(datasets), 8),
                                 sharex=False, sharey=False)
        fig.suptitle(f'Scoring Function Frequency ({metric.upper()} @ {THRESH_MAP.get(threshold, threshold)})',
                     fontsize=14, weight='bold')
        axes = reshape_axes(axes, 2, len(datasets))

        for j, ds_name in enumerate(datasets):
            df_full = loaded_data.get(ds_name)

            if df_full is None or df_full.empty:
                print(f"   No data for {ds_name}, skipping.")
                for i in range(2):
                    handle_empty_subplot(axes[i, j], f"{ds_name} ({PLOT_LABELS[i]})")
                continue

            full_sf_counts = {}
            existing_sfs = set()
            for sf in all_scoring_functions:
                exists = df_full['scoring'].astype(str).str.contains(sf, regex=False).any()
                if exists:
                    existing_sfs.add(sf)
                    mask = df_full['scoring'].astype(str).str.contains(sf, regex=False)
                    full_sf_counts[sf] = mask.sum()
                else:
                    full_sf_counts[sf] = 0

            for i, percentile in enumerate(PERCENTILES):
                ax = axes[i, j]
                plot_title = f"{ds_name} ({PLOT_LABELS[i]})"

                df_subset = get_top_workflows_pd(df_full, percentile, ranking_col=ranking_col)

                if df_subset.empty:
                    handle_empty_subplot(ax, plot_title, message=f'No data\n({PLOT_LABELS[i]})')
                    continue

                subset_sf_counts = {}
                for sf in existing_sfs:
                    mask = df_subset['scoring'].astype(str).str.contains(sf, regex=False)
                    subset_sf_counts[sf] = mask.sum()

                sf_frequency = {}
                baseline_frequency = percentile
                print(f"      Baseline frequency for {PLOT_LABELS[i]}: {baseline_frequency}")

                for sf in subset_sf_counts.keys():
                    if full_sf_counts.get(sf, 0) > 0:
                        sf_frequency[sf] = (subset_sf_counts[sf] / full_sf_counts[sf]) * 100
                    else:
                        sf_frequency[sf] = 0

                sf_freq_df = pd.DataFrame(list(sf_frequency.items()), columns=['scoring_function', 'frequency'])
                sf_freq_df = sf_freq_df.sort_values('frequency', ascending=False)

                colors = [get_sf_color(sf) for sf in sf_freq_df['scoring_function']]

                ax.barh(sf_freq_df['scoring_function'], sf_freq_df['frequency'],
                        color=colors, edgecolor='black', linewidth=0.5)

                ax.axvline(x=baseline_frequency, color='black', linestyle='--', linewidth=2, alpha=0.8, zorder=5)

                ax.set_title(plot_title, fontsize=12, weight='bold')
                ax.set_xlabel("Frequency (%)" if i == 1 else "", fontsize=11, fontweight='bold')
                ax.set_ylabel("Scoring Function" if j == 0 else "", fontsize=11, fontweight='bold')

                max_freq = sf_freq_df['frequency'].max() if not sf_freq_df.empty else 0
                ax.set_xlim(0, min(100, max_freq * 1.2 if max_freq > 0 else 10))

                ax.tick_params(axis='y', labelsize=9)
                ax.tick_params(axis='x', labelsize=9)

        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(output_subdir / f"sf_frequency_all_datasets_thresh{threshold}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        plots_generated += 1

    print("\nScoring Function Frequency plots generated.")
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

    generate_sf_frequency_plots(
        aggregated_dir=aggregated_dir,
        output_dir=output_dir,
        metrics=metrics,
        thresholds=thresholds,
        datasets=datasets
    )


if __name__ == "__main__":
    main()
