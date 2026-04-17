"""
Plotting functions for AVE (Analogue Versus Enrichment) analysis.

This module generates scatter plots comparing training vs validation performance
for workflows at various percentile levels.
"""

import math
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from .utils import sanitize_filename, format_metric_name, calculate_correlation_metrics
from .config import THRESHOLD_METRIC_BASES
from .plot_helpers import setup_plot_style, save_figure, ALL_WORKFLOWS_COLOR, TOP_WORKFLOWS_COLOR


def is_threshold_metric(metric: str) -> bool:
    """Check if a metric requires threshold filtering."""
    return metric.lower() in THRESHOLD_METRIC_BASES


def generate_overlay_plot(
    df_all: pd.DataFrame,
    df_top: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    metric_display_name: str,
    plot_metric_name: str,
    output_dir: Path,
    plot_threshold: Optional[float] = None,
    top_percentile: float = 10.0,
    ranking_label: str = "",
    skip_existing: bool = True
) -> Optional[List[Dict[str, Any]]]:
    """
    Generate overlay plot showing all workflows and top percentile in different colors.

    Args:
        df_all: Pivoted dataframe with all workflows
        df_top: Pivoted dataframe with top percentile workflows only
        x_col: Column name for x-axis values
        y_col: Column name for y-axis values
        x_label: Display label for x-axis
        y_label: Display label for y-axis
        metric_display_name: Formatted metric name for display
        plot_metric_name: Raw metric name for filename
        output_dir: Directory to save plots
        plot_threshold: Threshold value for filename (if threshold-dependent)
        top_percentile: Percentile value for labeling (e.g., 10.0 for top 10%)
        ranking_label: Label for ranking method (for filename)
        skip_existing: Skip generation if file exists

    Returns:
        List of per-target correlation metrics, or None if skipped/failed
    """
    threshold_str = (
        f"thresh{plot_threshold:.1f}".replace('.', '_')
        if plot_threshold is not None
        else ""
    )
    safe_metric = sanitize_filename(plot_metric_name)
    x_label_fn = sanitize_filename(x_label.lower())
    y_label_fn = sanitize_filename(y_label.lower().replace(' ', '_'))

    if top_percentile < 1:
        pct_str = f"top{top_percentile}pct".replace('.', '_')
    else:
        pct_str = f"top{top_percentile:.0f}pct"
    ranking_str = f"_{ranking_label}" if ranking_label else ""

    if threshold_str:
        base_filename = f"overlay_{x_label_fn}_vs_{y_label_fn}_{safe_metric}_{threshold_str}_{pct_str}{ranking_str}"
    else:
        base_filename = f"overlay_{x_label_fn}_vs_{y_label_fn}_{safe_metric}_{pct_str}{ranking_str}"

    plot_path = output_dir / f"{base_filename}.png"

    if skip_existing and plot_path.exists():
        print(f"    Skipping existing: {plot_path.name}")
        return None

    df_all_plot = df_all.dropna(subset=[x_col, y_col]).copy()
    df_top_plot = df_top.dropna(subset=[x_col, y_col]).copy()

    if df_all_plot.empty:
        print(f"    No data for {x_label} vs {y_label}. Skipping.")
        return None

    targets = sorted(df_all_plot['target'].unique())
    n_targets = len(targets)
    if n_targets == 0:
        return None

    ncols = min(n_targets, 4) if n_targets > 1 else 1
    nrows = math.ceil(n_targets / ncols)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(ncols * 4.2, nrows * 4.0),
        squeeze=False
    )
    axes_flat = axes.flatten()

    metrics_data = []
    if top_percentile < 1:
        top_pct_label = f"Top {top_percentile}%"
    else:
        top_pct_label = f"Top {top_percentile:.0f}%"

    for i, target in enumerate(targets):
        ax = axes_flat[i]

        df_target_all = df_all_plot[df_all_plot['target'] == target]
        df_target_top = df_top_plot[df_top_plot['target'] == target]

        x_all = df_target_all[x_col].values
        y_all = df_target_all[y_col].values

        all_vals = np.concatenate([x_all, y_all])
        all_vals = all_vals[~np.isnan(all_vals)]
        if len(all_vals) > 0:
            v_min, v_max = all_vals.min(), all_vals.max()
            data_range = v_max - v_min
            padding = data_range * 0.05 if data_range > 1e-6 else 0.05
            lims = [v_min - padding, v_max + padding]
        else:
            lims = [0, 1]
            data_range = 1

        rho_all, tau_all, rmse_all = calculate_correlation_metrics(x_all, y_all)

        if not df_target_top.empty:
            x_top = df_target_top[x_col].values
            y_top = df_target_top[y_col].values
            rho_top, tau_top, rmse_top = calculate_correlation_metrics(x_top, y_top)
        else:
            rho_top, tau_top, rmse_top = np.nan, np.nan, np.nan

        metrics_data.append({
            'target': target,
            'metric': plot_metric_name,
            'threshold': plot_threshold,
            'n_all_workflows': len(df_target_all),
            'n_top_workflows': len(df_target_top),
            'top_percentile': top_percentile,
            'rho_all': rho_all,
            'tau_all': tau_all,
            'rmse_all': rmse_all,
            'rho_top': rho_top,
            'tau_top': tau_top,
            'rmse_top': rmse_top
        })

        sns.scatterplot(
            data=df_target_all, x=x_col, y=y_col,
            alpha=0.7, s=20, ax=ax,
            color=ALL_WORKFLOWS_COLOR, edgecolor='none',
            label='All Workflows' if i == 0 else "", marker='o', zorder=2
        )

        if not df_target_top.empty:
            sns.scatterplot(
                data=df_target_top, x=x_col, y=y_col,
                alpha=0.9, s=35, ax=ax,
                color=TOP_WORKFLOWS_COLOR, edgecolor='black', linewidth=0.3,
                label=top_pct_label if i == 0 else "", marker='o', zorder=3
            )

        ax.plot(lims, lims, color='dimgray', linestyle='--', alpha=0.7, linewidth=1, zorder=1)

        rho_all_txt = f"{rho_all:.2f}" if not np.isnan(rho_all) else "-"
        tau_all_txt = f"{tau_all:.2f}" if not np.isnan(tau_all) else "-"
        rmse_all_txt = f"{rmse_all:.2f}" if not np.isnan(rmse_all) else "-"
        rho_top_txt = f"{rho_top:.2f}" if not np.isnan(rho_top) else "-"
        tau_top_txt = f"{tau_top:.2f}" if not np.isnan(tau_top) else "-"
        rmse_top_txt = f"{rmse_top:.2f}" if not np.isnan(rmse_top) else "-"

        ax.set_title(
            f"{target.upper()}\n"
            f"All: ρ={rho_all_txt}, τ={tau_all_txt}, RMSE={rmse_all_txt}\n"
            f"Top: ρ={rho_top_txt}, τ={tau_top_txt}, RMSE={rmse_top_txt}",
            fontsize=7, pad=5, fontweight='bold'
        )

        ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='lightgrey')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        if i % ncols == 0:
            ax.set_ylabel(f'{metric_display_name} ({y_label})', fontsize=9, fontweight='bold')
        else:
            ax.set_ylabel('')

        if i // ncols == nrows - 1 or i >= n_targets - ncols:
            ax.set_xlabel(f'{metric_display_name} ({x_label})', fontsize=9, fontweight='bold')
        else:
            ax.set_xlabel('')

        ax.tick_params(axis='both', which='major', labelsize=8)

        if data_range < 1.5 and data_range > 1e-6:
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        elif data_range > 50:
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

    for j in range(n_targets, len(axes_flat)):
        axes_flat[j].set_visible(False)

    ranking_info = f" ({top_pct_label} ranked by {ranking_label})" if ranking_label else f" ({top_pct_label})"
    fig_title = f'{x_label} vs. {y_label} ({metric_display_name}){ranking_info}'
    fig.suptitle(fig_title, y=1.0, fontsize=plt.rcParams['figure.titlesize'], fontweight='bold')

    handles, labels = axes_flat[0].get_legend_handles_labels() if n_targets > 0 else ([], [])
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.02), fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    else:
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {plot_path.name}")
    except Exception as e:
        print(f"    Error saving {plot_path.name}: {e}")

    plt.close(fig)

    return metrics_data


def generate_all_plots(
    df: pd.DataFrame,
    metrics: List[str],
    thresholds: List[float],
    top_workflows: Dict[str, Set[str]],
    ranking_label: str,
    output_dir: Path,
    skip_existing: bool = True
) -> List[Dict[str, Any]]:
    """
    Generate overlay plots for the given configuration.

    Args:
        df: Performance dataframe
        metrics: List of metrics to plot
        thresholds: List of thresholds for threshold-dependent metrics
        top_workflows: Dict mapping percentile name to set of workflow IDs
        ranking_label: Label for the ranking method (for filename)
        output_dir: Output directory
        skip_existing: Skip existing files

    Returns:
        List of all collected metrics
    """
    all_metrics = []

    for current_metric in metrics:
        is_thresh = is_threshold_metric(current_metric)
        print(f"\n== Processing metric: {current_metric} ==")

        for threshold in thresholds:
            if not is_thresh and threshold != thresholds[0]:
                continue

            threshold_label = f" @ {threshold}%" if is_thresh else ""
            print(f"  -- Threshold{threshold_label} --")

            if is_thresh:
                if 'threshold' not in df.columns:
                    print(f"    Warning: 'threshold' column missing")
                    continue
                df_thresh = df.copy()
                df_thresh['threshold'] = pd.to_numeric(df_thresh['threshold'], errors='coerce')
                df_thresh = df_thresh[df_thresh['threshold'] == threshold]
            else:
                df_thresh = df.copy()
                df_thresh = df_thresh.drop_duplicates(subset=['target', 'set', 'workflow_id'])

            if df_thresh.empty or current_metric not in df_thresh.columns:
                print(f"    No data for {current_metric}")
                continue

            try:
                df_pivot = df_thresh.pivot_table(
                    index=['target', 'workflow_id'],
                    columns='set',
                    values=current_metric
                ).reset_index()
            except Exception as e:
                print(f"    Error pivoting: {e}")
                continue

            if df_pivot.empty:
                continue

            if 'training' not in df_pivot.columns or 'validation' not in df_pivot.columns:
                print(f"    Missing training or validation columns")
                continue

            metric_display = format_metric_name(current_metric, threshold if is_thresh else None)
            threshold_val = threshold if is_thresh else None

            for pct_name, wf_ids in top_workflows.items():
                if not wf_ids:
                    continue

                try:
                    pct_value = float(pct_name.replace('top_', '').replace('pct', '').replace('_', '.'))
                except ValueError:
                    pct_value = 10.0

                df_top = df_pivot[df_pivot['workflow_id'].isin(wf_ids)]

                result = generate_overlay_plot(
                    df_all=df_pivot,
                    df_top=df_top,
                    x_col='training',
                    y_col='validation',
                    x_label='Training',
                    y_label='Validation',
                    metric_display_name=metric_display,
                    plot_metric_name=current_metric,
                    output_dir=output_dir,
                    plot_threshold=threshold_val,
                    top_percentile=pct_value,
                    ranking_label=ranking_label,
                    skip_existing=skip_existing
                )

                if result:
                    for m in result:
                        m['percentile'] = pct_name
                        m['ranking'] = ranking_label
                    all_metrics.extend(result)

    return all_metrics
