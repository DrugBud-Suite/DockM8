#!/usr/bin/env python3
"""
Docking vs Selection Method Analysis

Generates performance heatmaps and occurrence count heatmaps for docking method
vs selection method combinations. Compares mean-based and median-based ranking
for selecting top workflows.

This script produces two types of figures:
- Figure A: All Workflows vs Top 1%
- Figure B: Top 1% vs Top 0.1%

For each ranking method (mean/median) and content type (performance/count).

Usage:
    python docking_selection.py --base-path PATH --output-dir PATH
                                --metrics ref,ef --thresholds 0p1,1
"""

import itertools
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import (
    DATASETS, DATASET_DIRS, METRICS, THRESHOLDS, THRESH_MAP,
    get_output_dir, get_aggregated_dir, parse_list_arg, setup_argument_parser,
    load_metric_data, calculate_overall_performance, get_top_workflows_pd
)
from .plot_helpers import setup_plot_style, add_dash_annotations, calculate_vmin_vmax


def _generate_single_heatmap(
    ax,
    df_full: pd.DataFrame,
    percentile: Optional[float],
    ranking_col: str,
    agg_func: str,
    selection_methods: List[str],
    docking_methods: List[str],
    plot_title: str,
    show_xlabel: bool
) -> None:
    """Generate a single heatmap subplot."""
    if df_full.empty or not selection_methods or not docking_methods:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=9)
        ax.set_title(plot_title, fontsize=10)
        return

    df_subset = get_top_workflows_pd(df_full, percentile, ranking_col=ranking_col) if percentile is not None else df_full

    if df_subset.empty:
        label = f'No data\n(Top {percentile}%)' if percentile else 'No data'
        ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=9)
        ax.set_title(plot_title, fontsize=10)
        return

    if agg_func == 'median':
        heatmap_data = df_subset.groupby(['docking', 'selection_method'])[ranking_col].median().reset_index()
        value_col = ranking_col
    else:
        heatmap_data = df_subset.groupby(['docking', 'selection_method']).size().reset_index(name='count')
        value_col = 'count'

    try:
        heatmap_pivot = heatmap_data.pivot(index='selection_method', columns='docking', values=value_col)
        heatmap_pivot = heatmap_pivot.reindex(index=selection_methods, columns=docking_methods)
        if agg_func == 'size':
            heatmap_pivot.fillna(0, inplace=True)
    except Exception as e:
        print(f"  Error pivoting data: {e}")
        ax.text(0.5, 0.5, 'Pivot Error', ha='center', va='center', fontsize=9, color='red')
        ax.set_title(plot_title, fontsize=10)
        return

    vmin, vmax = calculate_vmin_vmax(heatmap_pivot.values) if agg_func == 'median' else (None, None)
    fmt = ".2f" if agg_func == 'median' else ".0f"

    sns.heatmap(heatmap_pivot, annot=True, fmt=fmt, cmap="Blues", linewidths=.5,
                cbar=False, ax=ax, annot_kws={"size": 8}, vmin=vmin, vmax=vmax)
    if agg_func == 'median':
        add_dash_annotations(ax, heatmap_pivot)

    ax.set_title(plot_title, fontsize=10, weight='bold')
    ax.set_xlabel("Docking Method" if show_xlabel else "", fontsize=9, fontweight='bold')
    ax.set_ylabel("Selection Method", fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


def generate_docking_selection_plots(
    aggregated_dir: Path,
    output_dir: Path,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None
) -> int:
    """
    Generate all docking vs selection method plots.

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
    print("Generating Docking/Selection Plots...")
    analysis_type_folder = "docking_selection"
    plots_generated = 0

    ranking_methods = {'mean': False, 'median': True}
    plot_contents = {'performance': 'median', 'count': 'size'}

    percentiles_a = [None, 1.0]
    percentiles_b = [1.0, 0.1]
    plot_labels_a = ["All Workflows", "Top 1%"]
    plot_labels_b = ["Top 1%", "Top 0.1%"]

    for metric, threshold in itertools.product(metrics, thresholds):
        print(f"\n--- Metric: {metric.upper()}, Threshold: {THRESH_MAP.get(threshold, threshold)} ---")

        loaded_data = {}
        dataset_axes = {}
        data_found = False
        print("Loading data...")

        for ds_name in datasets:
            if ds_name not in DATASET_DIRS:
                print(f"  Warning: Unknown dataset {ds_name}. Skipping.")
                continue

            df_raw = load_metric_data(aggregated_dir, ds_name, metric, threshold)

            if df_raw is not None:
                df_mean = calculate_overall_performance(df_raw.copy(), use_median=False)
                df_median = calculate_overall_performance(df_raw.copy(), use_median=True)
                loaded_data[ds_name] = {
                    'mean': df_mean.rename(columns={'overall_performance': 'overall_performance_mean'}),
                    'median': df_median.rename(columns={'overall_performance': 'overall_performance_median'})
                }
                ds_selection = sorted(list(loaded_data[ds_name]['mean']['selection_method'].dropna().unique()))
                ds_docking = sorted(list(loaded_data[ds_name]['mean']['docking'].dropna().unique()))
                dataset_axes[ds_name] = {'selection': ds_selection, 'docking': ds_docking}
                print(f"  Loaded {ds_name} ({len(df_raw)} rows). Found {len(ds_docking)} docking, {len(ds_selection)} selection methods.")
                data_found = True
            else:
                print(f"  Warning: File not found for {ds_name}. Skipping.")
                loaded_data[ds_name] = None
                dataset_axes[ds_name] = {'selection': [], 'docking': []}

        if not data_found:
            print("  No data found for any dataset. Skipping plots for this metric/threshold.")
            continue

        for rank_name, use_median_rank in ranking_methods.items():
            ranking_col = f'overall_performance_{rank_name}'

            for content_name, agg_func in plot_contents.items():
                print(f"  Generating Plots: Rank={rank_name}, Content={content_name} ({agg_func})")

                output_subdir = output_dir / analysis_type_folder / f"{rank_name}_ranking_{content_name}" / metric
                output_subdir.mkdir(parents=True, exist_ok=True)

                fig_a, axes_a = plt.subplots(2, len(datasets), figsize=(4 * len(datasets), 9), sharex=False, sharey=False)
                fig_a.suptitle(f'Docking vs Selection ({content_name.capitalize()}) ({metric.upper()} @ {THRESH_MAP.get(threshold, threshold)}) - Rank: {rank_name.capitalize()} - All vs Top 1%', fontsize=12, weight='bold')
                if len(datasets) == 1:
                    axes_a = axes_a.reshape(2, 1)
                elif axes_a.ndim == 1:
                    axes_a = axes_a.reshape(2, -1)

                fig_b, axes_b = plt.subplots(2, len(datasets), figsize=(4 * len(datasets), 9), sharex=False, sharey=False)
                fig_b.suptitle(f'Docking vs Selection ({content_name.capitalize()}) ({metric.upper()} @ {THRESH_MAP.get(threshold, threshold)}) - Rank: {rank_name.capitalize()} - Top 1% vs Top 0.1%', fontsize=12, weight='bold')
                if len(datasets) == 1:
                    axes_b = axes_b.reshape(2, 1)
                elif axes_b.ndim == 1:
                    axes_b = axes_b.reshape(2, -1)

                for j, ds_name in enumerate(datasets):
                    plot_title_a0 = f"{ds_name} ({plot_labels_a[0]})"
                    plot_title_a1 = f"{ds_name} ({plot_labels_a[1]})"
                    plot_title_b0 = f"{ds_name} ({plot_labels_b[0]})"
                    plot_title_b1 = f"{ds_name} ({plot_labels_b[1]})"

                    df_dict = loaded_data.get(ds_name)
                    ds_selection_methods = dataset_axes.get(ds_name, {}).get('selection', [])
                    ds_docking_methods = dataset_axes.get(ds_name, {}).get('docking', [])

                    if df_dict is None:
                        for i in range(2):
                            axes_a[i, j].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=9)
                            axes_a[i, j].set_title(plot_title_a0 if i == 0 else plot_title_a1, fontsize=10)
                            axes_b[i, j].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=9)
                            axes_b[i, j].set_title(plot_title_b0 if i == 0 else plot_title_b1, fontsize=10)
                        continue

                    df_full = df_dict[rank_name]

                    for i, percentile in enumerate(percentiles_a):
                        ax = axes_a[i, j]
                        plot_title = plot_title_a0 if i == 0 else plot_title_a1
                        _generate_single_heatmap(
                            ax, df_full, percentile, ranking_col, agg_func,
                            ds_selection_methods, ds_docking_methods, plot_title, i == 1
                        )

                    for i, percentile in enumerate(percentiles_b):
                        ax = axes_b[i, j]
                        plot_title = plot_title_b0 if i == 0 else plot_title_b1
                        _generate_single_heatmap(
                            ax, df_full, percentile, ranking_col, agg_func,
                            ds_selection_methods, ds_docking_methods, plot_title, i == 1
                        )

                fig_a.tight_layout(rect=[0, 0.03, 1, 0.97])
                fig_b.tight_layout(rect=[0, 0.03, 1, 0.97])
                fig_a.savefig(output_subdir / f"1A_dock_select_thresh{threshold}.png", dpi=300, bbox_inches='tight')
                fig_b.savefig(output_subdir / f"1B_dock_select_top_thresh{threshold}.png", dpi=300, bbox_inches='tight')
                plt.close(fig_a)
                plt.close(fig_b)
                plots_generated += 2

    print("\nDocking/Selection plots generated.")
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

    generate_docking_selection_plots(
        aggregated_dir=aggregated_dir,
        output_dir=output_dir,
        metrics=metrics,
        thresholds=thresholds,
        datasets=datasets
    )


if __name__ == "__main__":
    main()
