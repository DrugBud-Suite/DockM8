#!/usr/bin/env python3
"""
Interaction Plots: Number of Scoring Functions vs Consensus Method

Generates success rate heatmaps showing how combinations of scoring function
counts and consensus methods perform at different workflow percentile cutoffs.

Success rate is calculated as: (count in top percentile / count in full data) × 100

The plots show Top 1%, Top 0.1%, and Top 0.01% workflows using median-based ranking.

Usage:
    python interaction_plots.py --base-path PATH --output-dir PATH
                                --metrics ref,ef --thresholds 0p1,1
"""

import itertools
import traceback
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    DATASETS, DATASET_DIRS, METRICS, THRESHOLDS, THRESH_MAP,
    SINGLE_SF_PLACEHOLDER,
    get_output_dir, get_aggregated_dir, parse_list_arg, setup_argument_parser,
    load_metric_data, calculate_overall_performance, get_top_workflows_pd
)
from .plot_helpers import setup_plot_style, reshape_axes, handle_empty_subplot


PERCENTILES = [0.1, 0.01]
PLOT_LABELS = ["Top 0.1%", "Top 0.01%"]


def generate_interaction_plots(
    aggregated_dir: Path,
    output_dir: Path,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None
) -> int:
    """
    Generate interaction plots (NumSFs vs Consensus - Success Rate).

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
    print("Generating Interaction Plots (Num SFs vs Consensus - Success Rate)...")
    analysis_type_folder = "interaction_sfs_consensus"
    plots_generated = 0

    for metric, threshold in itertools.product(metrics, thresholds):
        print(f"\n--- Metric: {metric.upper()}, Threshold: {THRESH_MAP.get(threshold, threshold)} ---")

        loaded_data = {}
        all_num_sfs = set()
        all_consensus_methods = set()
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

                # Vectorized equivalent of the per-row split/len apply over 12.28M rows:
                # SFs in a '+'-joined scoring string = count('+') + 1, with the empty
                # string mapped to 0 (matching `len(x.split('+')) if x else 0`; after
                # .astype(str), pd.notna is always True so only '' maps to 0).
                scoring_str = df_median['scoring'].astype(str)
                num_sfs = (scoring_str.str.count(r'\+') + 1).where(scoring_str != '', 0)
                df_median['num_sfs'] = num_sfs.astype(np.int16)

                if 'consensus_method' not in df_median.columns:
                    raise ValueError("'consensus_method' column not found in DataFrame.")

                df_median['consensus_method'] = df_median['consensus_method'].astype(str).replace('nan', np.nan).fillna(SINGLE_SF_PLACEHOLDER)

                loaded_data[ds_name] = df_median.rename(columns={'overall_performance': 'overall_performance_median'})
                all_num_sfs.update(loaded_data[ds_name]['num_sfs'].unique())
                all_consensus_methods.update(loaded_data[ds_name]['consensus_method'].unique())
                print(f"   Loaded {ds_name} ({len(df_raw)} rows)")
                data_found = True
            except Exception as e:
                print(f"   Error loading or processing file for {ds_name}: {e}")
                traceback.print_exc()
                loaded_data[ds_name] = None

        if not data_found:
            print("   No data found for any dataset. Skipping plots for this metric/threshold.")
            continue

        all_num_sfs = sorted([s for s in all_num_sfs if s > 0])
        sf_placeholder = [SINGLE_SF_PLACEHOLDER] if SINGLE_SF_PLACEHOLDER in all_consensus_methods else []
        other_methods = sorted([m for m in all_consensus_methods if m != SINGLE_SF_PLACEHOLDER and m != 'None'])
        all_consensus_methods = sf_placeholder + other_methods

        ranking_col = 'overall_performance_median'
        print(f"   Generating Plot: Rank=median, Content=success_rate")

        output_subdir = output_dir / analysis_type_folder / "median_ranking_success_rate" / metric
        output_subdir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, len(datasets), figsize=(4.5 * len(datasets), 8), sharex=False, sharey=False)
        fig.suptitle(f'Normalized Counts ({metric.upper()} @ {THRESH_MAP.get(threshold, threshold)})', fontsize=14, weight='bold')
        axes = reshape_axes(axes, 2, len(datasets))

        for j, ds_name in enumerate(datasets):
            df_full = loaded_data.get(ds_name)

            if df_full is None or df_full.empty:
                for i in range(2):
                    handle_empty_subplot(axes[i, j], f"{ds_name} ({PLOT_LABELS[i]})")
                continue

            if not {'num_sfs', 'consensus_method'}.issubset(df_full.columns):
                print(f"   Error: Missing 'num_sfs' or 'consensus_method' for {ds_name}. Skipping.")
                for i in range(2):
                    handle_empty_subplot(axes[i, j], f"{ds_name} ({PLOT_LABELS[i]})", message='Data Error')
                continue

            counts_full_by_n_c = df_full.groupby(['num_sfs', 'consensus_method']).size()
            first_row_label = all_num_sfs[0] if all_num_sfs else None
            first_col_label = all_consensus_methods[0] if all_consensus_methods else None

            for i, percentile in enumerate(PERCENTILES):
                ax = axes[i, j]
                plot_title = f"{ds_name} ({PLOT_LABELS[i]})"

                df_subset = get_top_workflows_pd(df_full, percentile, ranking_col=ranking_col)

                if df_subset.empty:
                    ax.text(0.5, 0.5, f'No data\n({PLOT_LABELS[i]})', ha='center', va='center', fontsize=10)
                    ax.set_title(plot_title, fontsize=12)
                    ax.set_xticks(np.arange(len(all_consensus_methods)) + 0.5)
                    ax.set_xticklabels(all_consensus_methods, fontsize=10, rotation=45, ha="right", rotation_mode="anchor")
                    ax.set_yticks(np.arange(len(all_num_sfs)) + 0.5)
                    ax.set_yticklabels(all_num_sfs, rotation=0, fontsize=10)
                    continue

                counts_subset_by_n_c = df_subset.groupby(['num_sfs', 'consensus_method']).size()

                success_rate_data = {}
                for num_sf in all_num_sfs:
                    success_rate_data[num_sf] = {}
                    for consensus in all_consensus_methods:
                        C_s_nc = counts_subset_by_n_c.get((num_sf, consensus), 0)
                        C_f_nc = counts_full_by_n_c.get((num_sf, consensus), 0)

                        if C_f_nc > 0:
                            success_rate = (C_s_nc / C_f_nc) * 100
                        else:
                            success_rate = 0.0 if C_s_nc == 0 else np.nan

                        success_rate_data[num_sf][consensus] = success_rate

                success_rate_pivot = pd.DataFrame.from_dict(success_rate_data, orient='index')
                success_rate_pivot = success_rate_pivot.reindex(index=all_num_sfs, columns=all_consensus_methods)

                annot_df = success_rate_pivot.copy().astype(object)

                if first_row_label is not None and first_col_label is not None:
                    for r_idx, row_label in enumerate(success_rate_pivot.index):
                        for c_idx, col_label in enumerate(success_rate_pivot.columns):
                            val = success_rate_pivot.iloc[r_idx, c_idx]
                            is_first_row = (row_label == first_row_label)
                            is_first_col = (col_label == first_col_label)

                            if is_first_row and is_first_col:
                                annot_df.iloc[r_idx, c_idx] = '-' if pd.isna(val) else f"{val:.2f}"
                            elif is_first_row or is_first_col:
                                annot_df.iloc[r_idx, c_idx] = '/'
                            else:
                                if pd.isna(val) or val == 0:
                                    annot_df.iloc[r_idx, c_idx] = '-'
                                else:
                                    annot_df.iloc[r_idx, c_idx] = f"{val:.2f}"
                else:
                    for r_idx in range(len(success_rate_pivot.index)):
                        for c_idx in range(len(success_rate_pivot.columns)):
                            val = success_rate_pivot.iloc[r_idx, c_idx]
                            annot_df.iloc[r_idx, c_idx] = '-' if pd.isna(val) else f"{val:.2f}"

                sns.heatmap(success_rate_pivot, annot=annot_df, fmt="", cmap="Blues",
                            linewidths=.5, cbar=False, ax=ax, annot_kws={"size": 10})

                ax.set_title(plot_title, fontsize=12, weight='bold')
                ax.set_xlabel("Consensus Method" if i == 1 else "", fontsize=11, fontweight='bold')
                ax.set_ylabel("Number of SFs" if j == 0 else "", fontsize=11, fontweight='bold')
                ax.set_xticks(np.arange(len(all_consensus_methods)) + 0.5)
                ax.set_xticklabels(all_consensus_methods, fontsize=10, rotation=45, ha="right", rotation_mode="anchor")
                ax.set_yticks(np.arange(len(all_num_sfs)) + 0.5)
                ax.set_yticklabels(all_num_sfs, rotation=0, fontsize=10)

        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(output_subdir / f"4C_interaction_success_rate_thresh{threshold}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        plots_generated += 1

    print("\nInteraction Success Rate plots generated.")
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

    generate_interaction_plots(
        aggregated_dir=aggregated_dir,
        output_dir=output_dir,
        metrics=metrics,
        thresholds=thresholds,
        datasets=datasets
    )


if __name__ == "__main__":
    main()
