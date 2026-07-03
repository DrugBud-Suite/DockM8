"""
Generate literature comparison boxplots for DockM8 vs literature models.

This script produces one plot per dataset per metric, comparing DockM8 models
with literature-reported scoring functions across DEKOIS, DUD-E, and Lit-PCBA datasets.

Usage:
    python literature_boxplots.py --lit-data-path PATH --dockm8-data-path PATH --output-dir PATH
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import get_base_path, get_output_dir, NUMERIC_COLUMNS, parse_list_arg
from .data_loader import load_all_literature_datasets
from .utils import sanitize_filename
from .plot_helpers import setup_plot_style, save_figure, sort_models_by_median, style_boxplot_labels


# --- Configuration ---
DATASET_FILES = {
    'DEKOIS': {
        'Literature': 'dekois_complete_metrics.csv',
        'DockM8': 'DEKOIS_dockm8_results.csv'
    },
    'DUD-E': {
        'Literature': 'dude_complete_metrics.csv',
        'DockM8': 'DUD-E_dockm8_results.csv'
    },
    'Lit-PCBA': {
        'Literature': 'litpcba_complete_metrics.csv',
        'DockM8': 'lit-pcba_dockm8_results.csv'
    }
}

EXCLUDE_MODELS = [
    "KarmaDock Aligned",
    "Karmadock Raw",
    "GlideSP",
    "RF-Scorev3(ODDT)/GlideSP",
    "RF-Scorev2(ODDT)/GlideSP",
    "RF-Scorev1(ODDT)/GlideSP",
    "RF-Scorev4/GlideSP",
    "NNscore(ODDT)/GlideSP"
]

DEFAULT_METRICS = ['NEF_calc', 'REF', 'PM', 'AUC_ROC', 'BEDROC']
LITPCBA_XLIM = {
    'NEF_calc': (0, 40),
    'REF': (0, 75),
    'PM': None,
    'AUC_ROC': None,
    'BEDROC': None
}

DOCKM8_HIGHLIGHT_COLOR = '#ff7700'
# Distinct colour for the DockM8 consensus recomputed WITHOUT RFScoreVS, so the
# with/without-RFScoreVS DockM8 bars are visually separable in the same plot.
DOCKM8_NORF_HIGHLIGHT_COLOR = '#9467bd'
# Single flat colour shared by all literature (non-DockM8) methods.
LITERATURE_HIGHLIGHT_COLOR = '#4C72B0'


def create_literature_boxplot(
    df: pd.DataFrame,
    dataset_name: str,
    metric: str,
    figsize: Tuple[float, float] = (10, 15),
    dpi: int = 300,
    use_thresholds: bool = True,
    selected_thresholds: Optional[List[float]] = None,
    dockm8_highlight_color: str = DOCKM8_HIGHLIGHT_COLOR,
    model_palette: str = 'Blues',
    match_dockm8_targets: bool = False,
    vertical_layout: bool = True,
    save_path: Optional[str] = None,
    exclude_models: Optional[List[str]] = None,
    show_title: bool = False,
    xlim: Optional[Tuple[float, float]] = None
) -> bool:
    """
    Create a single boxplot for one dataset comparing DockM8 to literature models.

    Args:
        df: DataFrame for the dataset
        dataset_name: Name of the dataset
        metric: Metric to plot
        figsize: Size of the figure
        dpi: Resolution of the figure
        use_thresholds: If True, expects 'Percentile' column and groups by it
        selected_thresholds: List of thresholds to display
        dockm8_highlight_color: Color used to highlight DockM8 models
        model_palette: Seaborn palette name for models
        match_dockm8_targets: If True, only include targets present in DockM8 data
        vertical_layout: If True, models on y-axis, metric on x-axis
        save_path: Path to save the figure
        exclude_models: List of model names to exclude
        show_title: If True, show the figure title
        xlim: Tuple (min, max) to set x-axis limits

    Returns:
        True if plot was generated successfully, False otherwise
    """
    if df is None or df.empty:
        print(f"Warning: DataFrame for {dataset_name} is empty or None.")
        return False

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    if show_title:
        title_suffix = " Across Selected Percentile Thresholds" if use_thresholds else ""
        fig.suptitle(
            f'{dataset_name} - Model {metric} Performance{title_suffix} (DockM8 Models Highlighted)',
            fontsize=18, y=1.0
        )

    dockm8_targets = set()
    if match_dockm8_targets:
        if 'Target' in df.columns and 'Model' in df.columns:
            dockm8_models_mask = df['Model'].str.contains('DockM8', case=False)
            if dockm8_models_mask.any():
                dockm8_targets = set(df.loc[dockm8_models_mask, 'Target'].unique())
                print(f"Found {len(dockm8_targets)} targets in DockM8 data for {dataset_name}")
            else:
                print(f"Warning: No DockM8 models found in {dataset_name} dataset")

    if 'Model' not in df.columns:
        print(f"Warning: 'Model' column not found in {dataset_name} dataset.")
        ax.text(0.5, 0.5, "'Model' column missing", ha='center', va='center', fontsize=14)
        plt.close(fig)
        return False

    plot_data = df.copy()
    plot_data['Model'] = plot_data['Model'].str.strip()

    if exclude_models is not None and len(exclude_models) > 0:
        original_count = len(plot_data)
        plot_data = plot_data[~plot_data['Model'].isin(exclude_models)]
        filtered_count = len(plot_data)
        if original_count > filtered_count:
            print(f"Excluded {original_count - filtered_count} rows with models from {dataset_name}")

        if plot_data.empty:
            print(f"Warning: No data left after excluding models for {dataset_name}.")
            plt.close(fig)
            return False

    target_filter_desc = "all targets"

    if 'Target' in plot_data.columns:
        all_targets = plot_data['Target'].unique()
        initial_target_count = len(all_targets)
        target_count = initial_target_count

        if match_dockm8_targets and dockm8_targets:
            original_count = len(plot_data)
            plot_data = plot_data[plot_data['Target'].isin(dockm8_targets)]
            filtered_count = len(plot_data)

            filtered_targets = plot_data['Target'].unique()
            target_count = len(filtered_targets)
            target_filter_desc = f"{target_count} DockM8 targets"

            print(f"Filtered {dataset_name} from {original_count} to {filtered_count} rows")
            print(f"Reduced from {initial_target_count} to {target_count} unique targets")

            if plot_data.empty:
                print(f"Warning: No data after filtering to DockM8 targets for {dataset_name}.")
                plt.close(fig)
                return False
        else:
            if match_dockm8_targets:
                print(f"Warning: Cannot filter by DockM8 targets for {dataset_name}")
            target_filter_desc = f"{target_count} targets"

    ax.set_title(f'{dataset_name} Dataset ({target_filter_desc})', pad=20, fontsize=16)

    if metric not in plot_data.columns:
        print(f"Warning: Metric '{metric}' not found in {dataset_name} dataset.")
        ax.text(0.5, 0.5, f"Metric '{metric}' not available", ha='center', va='center', fontsize=14)
        plt.close(fig)
        return False

    plot_data['is_dockm8'] = plot_data['Model'].str.contains('DockM8', case=False)
    plot_data = plot_data.dropna(subset=[metric])

    has_percentile = 'Percentile' in plot_data.columns
    use_thresholds_local = use_thresholds and has_percentile

    if use_thresholds and not has_percentile:
        print(f"Warning: 'Percentile' column not found in {dataset_name}. Plotting without threshold grouping.")

    if use_thresholds_local and selected_thresholds is not None:
        plot_data = plot_data[plot_data['Percentile'].isin(selected_thresholds)]
        if plot_data.empty:
            print(f"Warning: No data matches the selected thresholds for {dataset_name}.")
            plt.close(fig)
            return False

    if plot_data.empty:
        print(f"Warning: No valid data for metric '{metric}' in {dataset_name} after cleaning.")
        plt.close(fig)
        return False

    try:
        if use_thresholds_local and selected_thresholds and (1.0 in selected_thresholds) and (1.0 in plot_data['Percentile'].unique()):
            threshold1_data = plot_data[plot_data['Percentile'] == 1.0]
            sorted_models = sort_models_by_median(threshold1_data, 'Model', metric)
        else:
            sorted_models = sort_models_by_median(plot_data, 'Model', metric)

        if not sorted_models:
            print(f"Warning: No models found after sorting for '{metric}' in {dataset_name}.")
            plt.close(fig)
            return False
    except Exception as e:
        print(f"Error during model sorting for {dataset_name}: {e}")
        plt.close(fig)
        return False

    if vertical_layout:
        x_var = metric
        y_var = 'Model'
    else:
        x_var = 'Model'
        y_var = metric

    hue_var = None
    palette = None
    hue_order = None
    num_hues = 1
    boxplot_width = 0.6

    if use_thresholds_local:
        hue_var = 'Percentile'
        available_thresholds = sorted(plot_data['Percentile'].unique())
        hue_order = available_thresholds
        num_hues = len(available_thresholds)

        try:
            palette = sns.color_palette('viridis', num_hues)
        except Exception:
            palette = sns.color_palette(n_colors=num_hues)
        boxplot_width = 0.8

    medianprops = {'linewidth': 2.5, 'color': 'black'}
    boxprops = {'linewidth': 1.5}
    flierprops = dict(marker='.', markerfacecolor='grey', markersize=4,
                      linestyle='none', markeredgecolor='grey')

    try:
        order_param = {'order': sorted_models}
        orient_param = 'h' if vertical_layout else None

        sns.boxplot(
            data=plot_data,
            x=x_var,
            y=y_var,
            hue=hue_var,
            **order_param,
            orient=orient_param,
            hue_order=hue_order,
            palette=palette,
            ax=ax,
            width=boxplot_width,
            dodge=use_thresholds_local,
            showfliers=True,
            flierprops=flierprops,
            medianprops=medianprops,
            boxprops=boxprops
        )

        if len(ax.patches) > 0:
            boxes_by_model = {}
            num_thresholds = 1 if not use_thresholds_local else len(available_thresholds)

            if vertical_layout:
                for i, model in enumerate(sorted_models):
                    indices = []
                    for j in range(num_thresholds):
                        idx = i + j * len(sorted_models)
                        if idx < len(ax.patches):
                            indices.append(idx)
                    boxes_by_model[model] = [ax.patches[idx] for idx in indices if idx < len(ax.patches)]
            else:
                for i, model in enumerate(sorted_models):
                    start_idx = i * num_thresholds
                    end_idx = start_idx + num_thresholds
                    boxes_by_model[model] = ax.patches[start_idx:end_idx]

            if use_thresholds_local and num_thresholds == 1:
                for model, boxes in boxes_by_model.items():
                    for box in boxes:
                        if 'no RFScoreVS' in model:
                            box.set_facecolor(DOCKM8_NORF_HIGHLIGHT_COLOR)
                            box.set_alpha(0.7)
                            box.set_edgecolor('black')
                        elif '@DockM8' in model:
                            box.set_facecolor("gray")
                            box.set_alpha(0.7)
                            box.set_edgecolor('black')
                        elif 'DockM8' in model:
                            box.set_facecolor(dockm8_highlight_color)
                            box.set_alpha(0.7)
                            box.set_edgecolor('black')
                        else:
                            box.set_facecolor(LITERATURE_HIGHLIGHT_COLOR)
                            box.set_alpha(0.7)
                            box.set_edgecolor('black')
            else:
                for model, boxes in boxes_by_model.items():
                    if 'no RFScoreVS' in model:
                        for box in boxes:
                            box.set_facecolor(DOCKM8_NORF_HIGHLIGHT_COLOR)
                            box.set_alpha(0.7)
                    elif '@DockM8' in model:
                        for box in boxes:
                            box.set_facecolor("gray")
                            box.set_alpha(0.7)
                    elif 'DockM8' in model:
                        for box in boxes:
                            box.set_facecolor(dockm8_highlight_color)
                            box.set_alpha(0.7)

        style_boxplot_labels(ax, ['DockM8'], axis='y' if vertical_layout else 'x')

    except Exception as e:
        print(f"Error during seaborn plotting for {dataset_name}: {e}")
        plt.close(fig)
        return False

    if vertical_layout:
        ax.set_xlabel(f'{metric} Value', fontsize=14)
        ax.set_ylabel('', fontsize=14)
        ax.tick_params(axis='y', rotation=0, labelsize=12, which='major', pad=5)
        ax.tick_params(axis='x', labelsize=12)
        if xlim is not None:
            ax.set_xlim(xlim)
    else:
        ax.set_ylabel(f'{metric} Value', fontsize=14)
        ax.set_xlabel('', fontsize=14)
        ax.tick_params(axis='x', rotation=30, labelsize=12, which='major', pad=5)
        ax.tick_params(axis='y', labelsize=12)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    if vertical_layout:
        ax.margins(y=0.015)
    else:
        ax.margins(x=0.015)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    print(f"Plot generated for {dataset_name}")
    plt.tight_layout()

    if save_path:
        success = save_figure(fig, save_path, dpi=dpi)
        plt.close(fig)
        return success

    plt.close(fig)
    return True


def create_per_dataset_boxplots(
    datasets_dict: Dict[str, pd.DataFrame],
    metric: str,
    output_dir: Path,
    figsize: Tuple[float, float] = (10, 15),
    dpi: int = 300,
    use_thresholds: bool = True,
    selected_thresholds: Optional[List[float]] = None,
    dockm8_highlight_color: str = DOCKM8_HIGHLIGHT_COLOR,
    model_palette: str = 'Blues',
    match_dockm8_targets: bool = False,
    vertical_layout: bool = True,
    filename_suffix: str = '',
    exclude_models: Optional[List[str]] = None,
    show_title: bool = False,
    litpcba_xlim: Optional[Tuple[float, float]] = None
) -> int:
    """
    Create one plot per dataset.

    Args:
        datasets_dict: Dictionary where keys are dataset names and values are DataFrames
        metric: Metric to plot
        output_dir: Base output directory
        figsize: Size of the figure
        dpi: Resolution
        use_thresholds: Whether to use threshold filtering
        selected_thresholds: List of thresholds to filter on
        dockm8_highlight_color: Color for DockM8 models
        model_palette: Palette for model colors
        match_dockm8_targets: Whether to filter to DockM8 targets only
        vertical_layout: Whether to use horizontal boxplots
        filename_suffix: Suffix to add to filename
        exclude_models: List of models to exclude
        show_title: Whether to show plot title
        litpcba_xlim: X-axis limits for Lit-PCBA dataset

    Returns:
        Number of successfully generated plots
    """
    success_count = 0

    for dataset_name, df in datasets_dict.items():
        if df is None or df.empty:
            print(f"Skipping {dataset_name}: No data available")
            continue

        safe_dataset = sanitize_filename(dataset_name.replace('-', '_').replace(' ', '_'))
        filename = f"{safe_dataset}_{metric}_vertical_1{filename_suffix}.png"
        save_path = str(output_dir / filename)

        xlim = None
        if dataset_name == 'Lit-PCBA' and litpcba_xlim is not None:
            xlim = litpcba_xlim

        success = create_literature_boxplot(
            df=df,
            dataset_name=dataset_name,
            metric=metric,
            figsize=figsize,
            dpi=dpi,
            use_thresholds=use_thresholds,
            selected_thresholds=selected_thresholds,
            dockm8_highlight_color=dockm8_highlight_color,
            model_palette=model_palette,
            match_dockm8_targets=match_dockm8_targets,
            vertical_layout=vertical_layout,
            save_path=save_path,
            exclude_models=exclude_models,
            show_title=show_title,
            xlim=xlim
        )

        if success:
            success_count += 1

    return success_count


def generate_all_literature_plots(
    datasets_dict: Dict[str, pd.DataFrame],
    output_dir: Path,
    dpi: int = 300
) -> int:
    """
    Generate all plot variations as defined in the original notebook.

    Args:
        datasets_dict: Dictionary of dataset DataFrames
        output_dir: Output directory for plots
        dpi: Resolution for saved figures

    Returns:
        Total number of successfully generated plots
    """
    total_success = 0
    valid_datasets = {name: df for name, df in datasets_dict.items() if df is not None and not df.empty}

    if not valid_datasets:
        print("No valid datasets loaded to generate plots.")
        return 0

    print("\nGenerating per-dataset plots with DockM8 highlighting...")

    common_params = {
        'use_thresholds': True,
        'selected_thresholds': [1.0],
        'dockm8_highlight_color': DOCKM8_HIGHLIGHT_COLOR,
        'model_palette': 'RdYlBu',
        'figsize': (10, 15),
        'vertical_layout': True,
        'dpi': dpi
    }

    metric_configs = [
        ('NEF_calc', (0, 40), True),
        ('REF', (0, 75), True),
        ('PM', None, True),
        ('AUC_ROC', None, False),
        ('BEDROC', None, False),
    ]

    for metric, xlim, has_variants in metric_configs:
        print(f"\n=== {metric} metric ===")

        if has_variants:
            total_success += create_per_dataset_boxplots(
                valid_datasets, metric, output_dir,
                match_dockm8_targets=True,
                exclude_models=EXCLUDE_MODELS,
                filename_suffix='',
                litpcba_xlim=xlim,
                **common_params
            )
            total_success += create_per_dataset_boxplots(
                valid_datasets, metric, output_dir,
                match_dockm8_targets=True,
                exclude_models=None,
                filename_suffix='_allmodels',
                litpcba_xlim=xlim,
                **common_params
            )
            total_success += create_per_dataset_boxplots(
                valid_datasets, metric, output_dir,
                match_dockm8_targets=False,
                exclude_models=EXCLUDE_MODELS,
                filename_suffix='_alltargets',
                litpcba_xlim=xlim,
                **common_params
            )
            total_success += create_per_dataset_boxplots(
                valid_datasets, metric, output_dir,
                match_dockm8_targets=False,
                exclude_models=None,
                filename_suffix='_allmodels_alltargets',
                litpcba_xlim=xlim,
                **common_params
            )
        else:
            total_success += create_per_dataset_boxplots(
                valid_datasets, metric, output_dir,
                match_dockm8_targets=True,
                exclude_models=None,
                filename_suffix='',
                litpcba_xlim=xlim,
                **common_params
            )

    return total_success


def main():
    """Main entry point for generating literature comparison boxplots."""
    parser = argparse.ArgumentParser(
        description='Generate literature comparison boxplots for DockM8 vs literature models'
    )
    parser.add_argument(
        '--base-path',
        type=Path,
        default=None,
        help='Base path to results data'
    )
    parser.add_argument(
        '--lit-data-path',
        type=Path,
        default=None,
        help='Path to literature data directory'
    )
    parser.add_argument(
        '--dockm8-data-path',
        type=Path,
        default=None,
        help='Path to DockM8 data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for plots'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figures (default: 300)'
    )

    args = parser.parse_args()

    base_path = get_base_path(args.base_path)

    if args.lit_data_path:
        lit_data_path = args.lit_data_path
    else:
        lit_data_path = base_path / 'literature_results' / 'lit_calc_data_rounded'

    if args.dockm8_data_path:
        dockm8_data_path = args.dockm8_data_path
    else:
        dockm8_data_path = base_path / 'dockm8_extracted_data'

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = base_path / 'plots' / 'literature_per_dataset'

    dataset_info = {}
    for dataset_name, files in DATASET_FILES.items():
        dataset_info[dataset_name] = {
            'Literature': lit_data_path / files['Literature'],
            'DockM8': dockm8_data_path / files['DockM8']
        }

    setup_plot_style(font_scale=1.4)

    print(f"\nLoading data from:")
    print(f"  Literature: {lit_data_path}")
    print(f"  DockM8: {dockm8_data_path}")
    print(f"Output directory: {output_dir}")

    datasets = load_all_literature_datasets(dataset_info, NUMERIC_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_success = generate_all_literature_plots(datasets, output_dir, args.dpi)

    print(f"\n=== Summary ===")
    print(f"Successfully generated {total_success} plots")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
