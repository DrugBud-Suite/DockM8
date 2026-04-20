"""
Shared plotting utilities for DockM8 analysis scripts.

This module provides common plotting functions used across multiple visualization
modules to ensure consistent styling and reduce code duplication.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# COLOR CONSTANTS
# =============================================================================

ALL_WORKFLOWS_COLOR = '#7f7f7f'
TOP_WORKFLOWS_COLOR = '#ff7f0e'
DOCKM8_HIGHLIGHT_COLOR = '#ff7700'


# =============================================================================
# PLOT STYLE SETUP
# =============================================================================

def setup_plot_style():
    """
    Configure matplotlib and seaborn plot style for publication-quality figures.

    Sets consistent DPI, font sizes, and whitegrid theme across all plots.
    """
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 12


# =============================================================================
# FIGURE MANAGEMENT
# =============================================================================

def save_figure(
    fig: plt.Figure,
    output_path: Path,
    dpi: int = 300,
    bbox_inches: str = 'tight',
    close_after: bool = True
) -> bool:
    """
    Save matplotlib figure with consistent settings.

    Args:
        fig: Matplotlib figure to save
        output_path: Path to save the figure
        dpi: Resolution (default: 300)
        bbox_inches: Bounding box setting (default: 'tight')
        close_after: Close the figure after saving (default: True)

    Returns:
        True if save succeeded, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
        if close_after:
            plt.close(fig)
        return True
    except Exception as e:
        print(f"Error saving figure to {output_path}: {e}")
        if close_after:
            plt.close(fig)
        return False


def create_subplot_grid(
    n_rows: int,
    n_cols: int,
    figsize: Optional[Tuple[float, float]] = None,
    squeeze: bool = False,
    **kwargs
) -> Tuple[plt.Figure, Any]:
    """
    Create a subplot grid with sensible defaults.

    Args:
        n_rows: Number of rows
        n_cols: Number of columns
        figsize: Figure size (auto-calculated if None)
        squeeze: Whether to squeeze axes array
        **kwargs: Additional arguments passed to plt.subplots

    Returns:
        Tuple of (figure, axes)
    """
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=squeeze, **kwargs)
    return fig, axes


# =============================================================================
# AXES UTILITIES
# =============================================================================

def reshape_axes(axes: Any, n_rows: int, n_cols: int) -> np.ndarray:
    """
    Reshape axes to always be a 2D array for consistent indexing.

    Handles the case where plt.subplots returns different shapes
    depending on row/column counts.

    Args:
        axes: Axes from plt.subplots (can be single, 1D array, or 2D array)
        n_rows: Number of rows in subplot grid
        n_cols: Number of columns in subplot grid

    Returns:
        2D numpy array of axes objects
    """
    if n_rows == 1 and n_cols == 1:
        return np.array([[axes]])
    elif n_rows == 1:
        return np.array([axes])
    elif n_cols == 1:
        return np.array([[ax] for ax in axes])
    else:
        return axes


def handle_empty_subplot(
    ax: plt.Axes,
    message: str = "No data available",
    remove_ticks: bool = True
) -> None:
    """
    Handle empty subplots with a consistent message.

    Args:
        ax: Matplotlib axes object
        message: Message to display (default: "No data available")
        remove_ticks: Whether to remove tick marks (default: True)
    """
    ax.text(
        0.5, 0.5, message,
        ha='center', va='center',
        fontsize=10, color='gray',
        transform=ax.transAxes
    )
    if remove_ticks:
        ax.set_xticks([])
        ax.set_yticks([])


# =============================================================================
# PERCENTILE FORMATTING
# =============================================================================

def format_percentile_label(percentile: float) -> str:
    """
    Format percentile value for display labels.

    Args:
        percentile: Percentile value (e.g., 1.0, 0.1, 0.01)

    Returns:
        Formatted string (e.g., "Top 1%", "Top 0.1%", "Top 0.01%")
    """
    if percentile >= 1:
        return f"Top {percentile:.0f}%"
    elif percentile >= 0.1:
        return f"Top {percentile:.1f}%"
    else:
        return f"Top {percentile:.2f}%"


def format_percentile_for_filename(percentile: float) -> str:
    """
    Format percentile value for use in filenames.

    Args:
        percentile: Percentile value (e.g., 1.0, 0.1, 0.01)

    Returns:
        Filename-safe string (e.g., "top1pct", "top0_1pct", "top0_01pct")
    """
    if percentile >= 1:
        return f"top{percentile:.0f}pct"
    else:
        return f"top{percentile}pct".replace('.', '_')


# =============================================================================
# HEATMAP UTILITIES
# =============================================================================

def add_dash_annotations(ax: plt.Axes, pivot_data: pd.DataFrame) -> None:
    """
    Add '-' text annotation to heatmap cells with NaN values.

    Args:
        ax: Matplotlib axes object with heatmap
        pivot_data: Pivot table DataFrame used for the heatmap
    """
    if pivot_data is None:
        return

    for i, idx in enumerate(pivot_data.index):
        for j, col in enumerate(pivot_data.columns):
            if pd.isna(pivot_data.iloc[i, j]):
                ax.text(
                    j + 0.5, i + 0.5, "-",
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10,
                    color='grey'
                )


def calculate_vmin_vmax(
    data_array: np.ndarray,
    lower_percentile: float = 5,
    upper_percentile: float = 95
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate vmin and vmax for heatmap color scaling using percentiles.

    Args:
        data_array: Numpy array of values
        lower_percentile: Lower percentile for vmin (default: 5)
        upper_percentile: Upper percentile for vmax (default: 95)

    Returns:
        Tuple of (vmin, vmax) or (None, None) if calculation not possible
    """
    try:
        valid_data = data_array.flatten()
        valid_data = valid_data[~np.isnan(valid_data)]
        if valid_data.size < 2:
            return None, None

        vmin = np.nanpercentile(valid_data, lower_percentile)
        vmax = np.nanpercentile(valid_data, upper_percentile)

        if vmin == vmax:
            if valid_data.size > 0 and valid_data.min() != valid_data.max():
                vmin = np.nanmin(valid_data)
                vmax = np.nanmax(valid_data)
            else:
                return None, None

        return vmin, vmax
    except Exception:
        return None, None


# =============================================================================
# LEGEND UTILITIES
# =============================================================================

def create_sf_category_legend(
    ax: plt.Axes,
    categories: dict,
    loc: str = 'upper right',
    title: str = 'SF Category'
) -> None:
    """
    Create a legend showing scoring function category colors.

    Args:
        ax: Matplotlib axes object
        categories: Dictionary with category info (from config.SF_CATEGORIES)
        loc: Legend location (default: 'upper right')
        title: Legend title (default: 'SF Category')
    """
    from matplotlib.patches import Patch

    legend_elements = []
    category_labels = {
        'ml': 'ML-Based',
        'empirical': 'Empirical',
        'knowledge': 'Knowledge-Based',
        'consensus': 'DockM8 Consensus',
        'other': 'Other'
    }

    for cat_key, cat_info in categories.items():
        if cat_key != 'other' or cat_info['functions']:
            label = category_labels.get(cat_key, cat_key.title())
            legend_elements.append(
                Patch(facecolor=cat_info['color'], label=label)
            )

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc=loc,
            title=title,
            framealpha=0.9
        )


# =============================================================================
# BOXPLOT UTILITIES
# =============================================================================

def sort_models_by_median(
    df: pd.DataFrame,
    model_col: str,
    value_col: str,
    ascending: bool = False
) -> List[str]:
    """
    Sort model names by their median performance value.

    Args:
        df: DataFrame with model and value columns
        model_col: Column name for model identifiers
        value_col: Column name for performance values
        ascending: Sort order (default: False = descending, best first)

    Returns:
        List of model names sorted by median performance
    """
    medians = df.groupby(model_col)[value_col].median()
    return medians.sort_values(ascending=ascending).index.tolist()


def style_boxplot_labels(
    ax: plt.Axes,
    highlight_patterns: List[str],
    bold: bool = True,
    axis: str = 'y'
) -> None:
    """
    Style boxplot labels by making certain patterns bold.

    Args:
        ax: Matplotlib axes object
        highlight_patterns: List of patterns to match for highlighting
        bold: Whether to make matching labels bold (default: True)
        axis: Which axis labels to style ('x' or 'y', default: 'y')
    """
    if axis == 'y':
        labels = ax.get_yticklabels()
    else:
        labels = ax.get_xticklabels()

    for label in labels:
        text = label.get_text()
        for pattern in highlight_patterns:
            if pattern.lower() in text.lower():
                if bold:
                    label.set_fontweight('bold')
                break
