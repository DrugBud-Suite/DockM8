"""
Generate internal boxplot comparisons for DockM8 scoring functions vs consensus methods.

This script produces one plot per dataset per metric, comparing single scoring functions
to DockM8 consensus methods (DockM8-best, DockM8-median).

Usage:
    python internal_boxplots.py [--output-dir PATH] [--datasets DEKOIS,DUD-E,Lit-PCBA]
                                [--metrics NEF_calc,REF,PM,AUC_ROC,BEDROC]
"""

import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import (
    get_output_dir, get_dockm8_results_dir, get_sf_type_and_color,
    SF_CATEGORIES, NUMERIC_COLUMNS, parse_list_arg
)
from .utils import sanitize_filename, clean_column_names
from .plot_helpers import setup_plot_style, save_figure, sort_models_by_median, style_boxplot_labels


# --- Configuration ---
DEFAULT_METRICS = ['NEF_calc', 'REF', 'PM', 'AUC_ROC', 'BEDROC']
THRESHOLD_INDEPENDENT_METRICS = {'AUC_ROC', 'BEDROC'}

DATASET_FILENAMES = {
    'DEKOIS': 'DEKOIS_dockm8_internal.csv',
    'DUD-E': 'DUD-E_dockm8_internal.csv',
    'Lit-PCBA': 'lit-pcba_dockm8_internal.csv'
}


def get_dataset_file_path(dockm8_results_dir: Path, dataset_name: str) -> Path:
    """
    Get the file path for a dataset's internal data.

    Args:
        dockm8_results_dir: Path to dockm8_results directory
        dataset_name: Name of the dataset

    Returns:
        Path to the CSV file
    """
    filename = DATASET_FILENAMES.get(dataset_name)
    if filename is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dockm8_results_dir / filename


def load_internal_dataset(
    dockm8_results_dir: Path,
    dataset_name: str,
    numeric_cols: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Load and clean internal dataset for a single dataset.

    Args:
        dockm8_results_dir: Path to dockm8_results directory
        dataset_name: Name of the dataset
        numeric_cols: List of columns to convert to numeric

    Returns:
        Cleaned DataFrame or None if loading fails
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLUMNS

    file_path = get_dataset_file_path(dockm8_results_dir, dataset_name)

    if not file_path.exists():
        print(f"Warning: File not found for {dataset_name}: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df = clean_column_names(df)
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"Successfully loaded {dataset_name} internal data with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading internal data for {dataset_name} from {file_path}: {e}")
        return None


def create_internal_boxplot(
    df: pd.DataFrame,
    dataset_name: str,
    metric: str,
    figsize: Tuple[float, float] = (10, 14),
    dpi: int = 300,
    selected_threshold: float = 1.0,
    save_path: Optional[str] = None,
    main_title_base: str = "Comparison between DockM8 single SFs and DockM8 consensus methods"
) -> bool:
    """
    Create a boxplot for a single dataset comparing SFs to consensus methods.

    Colors bars based on Scoring Function (SF) type defined in SF_CATEGORIES.
    'DockM8' models get bold labels. Displays 'EF' for 'NEF_calc' metric.
    Sets x-axis to log scale for 'Lit-PCBA' dataset.

    Args:
        df: DataFrame for the dataset
        dataset_name: Name of the dataset (DEKOIS, DUD-E, Lit-PCBA)
        metric: Metric to plot (e.g., 'NEF_calc', 'AUC_ROC')
        figsize: Size of the figure
        dpi: Resolution of the figure
        selected_threshold: Threshold (Percentile) to filter on
        save_path: Path to save the figure
        main_title_base: The base text for the main figure title

    Returns:
        True if plot was generated successfully, False otherwise
    """
    if df is None or df.empty:
        print(f"Warning: DataFrame for {dataset_name} is empty or None.")
        return False

    display_metric_name = 'EF' if metric == 'NEF_calc' else metric
    is_threshold_dependent = metric not in THRESHOLD_INDEPENDENT_METRICS

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if is_threshold_dependent:
        full_main_title = f'{main_title_base} ({display_metric_name} @ 1%)\n{dataset_name} Dataset'
    else:
        full_main_title = f'{main_title_base} ({display_metric_name})\n{dataset_name} Dataset'
    fig.suptitle(full_main_title, fontsize=16, fontweight='bold', y=0.98)

    if 'Model' not in df.columns:
        print(f"Warning: 'Model' column not found in {dataset_name} dataset.")
        ax.text(0.5, 0.5, "'Model' column missing", ha='center', va='center', fontsize=12)
        plt.close(fig)
        return False

    plot_data = df.copy()
    plot_data['Model'] = plot_data['Model'].str.strip()

    if metric not in plot_data.columns:
        print(f"Warning: Metric '{metric}' not found in {dataset_name} dataset.")
        ax.text(0.5, 0.5, f"Metric '{metric}' not available", ha='center', va='center', fontsize=12)
        plt.close(fig)
        return False

    if 'Percentile' in plot_data.columns:
        plot_data = plot_data[plot_data['Percentile'] == selected_threshold]
        if plot_data.empty:
            print(f"Warning: No data matches threshold {selected_threshold} for {dataset_name}.")
            plt.close(fig)
            return False

    if dataset_name == 'Lit-PCBA' and metric in plot_data.columns:
        original_count = len(plot_data)
        plot_data = plot_data[plot_data[metric] > 0]
        removed_count = original_count - len(plot_data)
        if removed_count > 0:
            print(f"Info: Removed {removed_count} non-positive '{metric}' values for '{dataset_name}' before applying log scale.")

    plot_data = plot_data.dropna(subset=[metric])

    if plot_data.empty:
        print(f"Warning: No valid data for metric '{metric}' in {dataset_name} after cleaning.")
        plt.close(fig)
        return False

    sorted_models = sort_models_by_median(plot_data, 'Model', metric)
    if not sorted_models:
        print(f"Warning: No models found after sorting for '{metric}' in {dataset_name}.")
        plt.close(fig)
        return False

    medianprops = {'linewidth': 2.5, 'color': 'black'}
    boxprops = {'linewidth': 1.5}
    flierprops = dict(marker='.', markerfacecolor='grey', markersize=4,
                      linestyle='none', markeredgecolor='grey')

    try:
        sns.boxplot(
            data=plot_data,
            x=metric,
            y='Model',
            order=sorted_models,
            orient='h',
            ax=ax,
            width=0.6,
            showfliers=True,
            flierprops=flierprops,
            medianprops=medianprops,
            boxprops=boxprops
        )

        if len(ax.patches) > 0:
            model_to_sf_color = {
                model: get_sf_type_and_color(model, SF_CATEGORIES)[1]
                for model in sorted_models
            }

            for i, patch in enumerate(ax.patches):
                if i < len(sorted_models):
                    model_name = sorted_models[i]
                    sf_color = model_to_sf_color.get(model_name, SF_CATEGORIES['other']['color'])

                    if hasattr(patch, 'set_facecolor'):
                        patch.set_facecolor(sf_color)
                        patch.set_alpha(0.75)
                        patch.set_edgecolor('black')

        style_boxplot_labels(ax, 'dockm8-', axis='y')

        if dataset_name == 'Lit-PCBA':
            try:
                ax.set_xscale('log')
            except ValueError as e:
                print(f"Warning: Could not set log scale for x-axis on {dataset_name}: {e}")

    except Exception as e:
        print(f"Error during seaborn plotting for {dataset_name}: {e}")
        plt.close(fig)
        return False

    ax.set_xlabel(f'{display_metric_name} Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('')
    ax.tick_params(axis='y', rotation=0, labelsize=10, which='major', pad=5)
    ax.margins(y=0.015)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    plt.tight_layout()

    if save_path:
        success = save_figure(fig, save_path, dpi=dpi)
        plt.close(fig)
        return success

    plt.close(fig)
    return True


def generate_internal_boxplots(
    dockm8_results_dir: Path,
    output_dir: Path,
    datasets: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    threshold: float = 1.0,
    dpi: int = 300
) -> int:
    """
    Generate all internal boxplots.

    Args:
        dockm8_results_dir: Path to dockm8_results directory
        output_dir: Output directory for plots
        datasets: List of datasets to process
        metrics: List of metrics to plot
        threshold: Percentile threshold to filter on
        dpi: DPI for saved figures

    Returns:
        Number of successfully generated plots
    """
    if datasets is None:
        datasets = list(DATASET_FILENAMES.keys())
    if metrics is None:
        metrics = DEFAULT_METRICS

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    print(f"\nGenerating internal boxplots for datasets: {datasets}")
    print(f"Metrics: {metrics}")
    print(f"Output directory: {output_dir}")

    success_count = 0
    total_count = 0

    for dataset_name in datasets:
        if dataset_name not in DATASET_FILENAMES:
            print(f"Warning: Unknown dataset '{dataset_name}'. Skipping.")
            continue

        df = load_internal_dataset(dockm8_results_dir, dataset_name, NUMERIC_COLUMNS)

        if df is None:
            continue

        for metric in metrics:
            total_count += 1
            print(f"\n--- Generating plot for {dataset_name} - {metric} ---")

            metric_for_filename = 'ef' if metric == 'NEF_calc' else metric.lower()
            safe_dataset = sanitize_filename(dataset_name.lower())
            filename = f"internal_{safe_dataset}_{metric_for_filename}_sf_colored.png"
            save_path = str(output_dir / filename)

            success = create_internal_boxplot(
                df=df,
                dataset_name=dataset_name,
                metric=metric,
                figsize=(10, 14),
                dpi=dpi,
                selected_threshold=threshold,
                save_path=save_path
            )

            if success:
                success_count += 1

    print(f"\n=== Summary ===")
    print(f"Successfully generated {success_count}/{total_count} plots")
    return success_count


def main():
    """Main entry point for generating internal boxplots."""
    parser = argparse.ArgumentParser(
        description='Generate boxplots comparing DockM8 SFs to consensus methods'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for plots (default: results/output/plots/internal)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default=None,
        help='Comma-separated datasets to plot (default: DEKOIS,DUD-E,Lit-PCBA)'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        default=None,
        help=f'Comma-separated metrics to plot (default: {",".join(DEFAULT_METRICS)})'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=1.0,
        help='Percentile threshold to filter on (default: 1.0)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figures (default: 300)'
    )

    args = parser.parse_args()

    dockm8_results_dir = get_dockm8_results_dir()
    output_dir = args.output_dir or (get_output_dir() / 'plots' / 'internal')

    datasets = parse_list_arg(args.datasets, list(DATASET_FILENAMES.keys()))
    metrics = parse_list_arg(args.metrics, DEFAULT_METRICS)

    success_count = generate_internal_boxplots(
        dockm8_results_dir=dockm8_results_dir,
        output_dir=output_dir,
        datasets=datasets,
        metrics=metrics,
        threshold=args.threshold,
        dpi=args.dpi
    )

    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
