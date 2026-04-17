#!/usr/bin/env python3
"""
AVE Analysis - Compare training/validation/full dataset performance.

This script analyzes AVE (Analogue Versus Enrichment) split results,
comparing workflow performance across training, validation, and full datasets.

Generates scatter plots comparing training vs validation performance at
various percentile levels.

Usage:
    python ave_analysis.py --split-dir PATH --full-dir PATH

Examples:
    # Basic usage
    python ave_analysis.py \\
        --split-dir /path/to/AVE_analysis \\
        --full-dir /path/to/lit-pcba

    # Custom targets and metrics
    python ave_analysis.py \\
        --split-dir /path/to/AVE_analysis \\
        --full-dir /path/to/lit-pcba \\
        --targets aldh1,fen1,gba \\
        --metrics auc_roc,bedroc,ref \\
        --thresholds 0.5,1.0,5.0

    # Custom ranking metrics
    python ave_analysis.py \\
        --split-dir /path/to/AVE_analysis \\
        --full-dir /path/to/lit-pcba \\
        --ranking-metrics "ref:1.0,bedroc"
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

from .data_loader import (
    validate_targets,
    discover_targets,
    load_all_data,
    validate_metrics,
)
from .workflow_ranking import identify_top_workflows
from .plotting import generate_all_plots
from .utils import parse_ranking_metrics, is_threshold_metric
from .plot_helpers import setup_plot_style


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AVE Analysis - Compare training/validation/full dataset performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--split-dir',
        type=Path,
        required=True,
        help='Path to AVE split analysis directory'
    )
    parser.add_argument(
        '--full-dir',
        type=Path,
        required=True,
        help='Path to full dataset results directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./plots/AVE_analysis'),
        help='Output directory for plots (default: ./plots/AVE_analysis)'
    )
    parser.add_argument(
        '--targets',
        type=str,
        default='auto',
        help='Comma-separated target names, or "auto" to discover (default: auto)'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        default='auc_roc,bedroc,ref',
        help='Comma-separated metrics to analyze (default: auc_roc,bedroc,ref)'
    )
    parser.add_argument(
        '--thresholds',
        type=str,
        default='1.0',
        help='Comma-separated thresholds for threshold-dependent metrics (default: 1.0)'
    )
    parser.add_argument(
        '--ranking-metrics',
        type=str,
        default='ref:1.0,bedroc',
        help='Metrics for ranking workflows. Format: metric:threshold,metric (default: ref:1.0,bedroc)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers for data loading (default: CPU count)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Regenerate plots even if they exist'
    )

    return parser.parse_args()


def run_ave_analysis(
    split_dir: Path,
    full_dir: Path,
    output_dir: Path,
    targets: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
    ranking_metrics: Optional[List[Tuple[str, Optional[float]]]] = None,
    workers: Optional[int] = None,
    skip_existing: bool = True
) -> int:
    """
    Run AVE analysis programmatically.

    Args:
        split_dir: Path to AVE split analysis directory
        full_dir: Path to full dataset results directory
        output_dir: Output directory for plots
        targets: List of target names (None for auto-discovery)
        metrics: List of metrics to analyze
        thresholds: List of thresholds for threshold-dependent metrics
        ranking_metrics: List of (metric, threshold) tuples for ranking
        workers: Number of parallel workers
        skip_existing: Skip existing plots

    Returns:
        Number of plots generated
    """
    if metrics is None:
        metrics = ['auc_roc', 'bedroc', 'ref']
    if thresholds is None:
        thresholds = [1.0]
    if ranking_metrics is None:
        ranking_metrics = [('ref', 1.0), ('bedroc', None)]

    print("=" * 60)
    print("AVE Analysis")
    print("=" * 60)

    if not split_dir.is_dir():
        print(f"Error: Split directory not found: {split_dir}")
        return 0
    if not full_dir.is_dir():
        print(f"Error: Full directory not found: {full_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    print("\n--- Target Discovery ---")
    if targets is None:
        targets = discover_targets(split_dir)
        print(f"Discovered {len(targets)} targets: {', '.join(targets)}")
    else:
        print(f"Using {len(targets)} specified targets")

    targets = validate_targets(targets, split_dir, full_dir)
    if not targets:
        print("Error: No valid targets found")
        return 0
    print(f"Validated {len(targets)} targets: {', '.join(targets)}")

    print(f"\nMetrics: {metrics}")
    print(f"Thresholds: {thresholds}")
    print(f"Ranking metrics: {ranking_metrics}")

    print("\n--- Loading Data ---")
    df, warnings = load_all_data(
        targets,
        split_dir,
        full_dir,
        max_workers=workers
    )

    if df.empty:
        print("Error: No data loaded")
        return 0

    print(f"\nLoaded {len(df)} records for {df['target'].nunique()} targets")
    print(f"Set distribution:\n{df['set'].value_counts().to_string()}")

    if warnings:
        print("\nWarnings during loading:")
        for target, warns in warnings.items():
            print(f"  {target}: {len(warns)} warning(s)")

    metrics = validate_metrics(df, metrics)
    if not metrics:
        print("Error: No valid metrics found")
        return 0
    print(f"\nValidated metrics: {metrics}")

    setup_plot_style()

    all_collected_metrics = []
    plots_generated = 0

    for rank_metric, rank_threshold in ranking_metrics:
        is_thresh = is_threshold_metric(rank_metric)
        rank_label = f"{rank_metric}" + (f"_{rank_threshold:.1f}".replace('.', '_') if is_thresh and rank_threshold else "")

        print(f"\n{'='*60}")
        print(f"Ranking by: {rank_metric}" + (f" @ {rank_threshold}%" if is_thresh and rank_threshold else ""))
        print("=" * 60)

        top_result = identify_top_workflows(
            df, rank_metric, rank_threshold,
            percentiles=(10.0, 1.0, 0.1)
        )

        scores = top_result['scores']
        if scores.empty:
            print(f"No scores calculated for {rank_metric}. Skipping.")
            continue

        print(f"Ranked {len(scores)} workflows")

        top_workflows = {
            'top_10pct': top_result.get('top_10_0', set()),
            'top_1pct': top_result.get('top_1_0', set()),
            'top_0_1pct': top_result.get('top_0_1', set()),
        }

        for name, wf_set in top_workflows.items():
            print(f"  {name}: {len(wf_set)} workflows")

        print("\n--- Generating Plots ---")
        collected = generate_all_plots(
            df=df,
            metrics=metrics,
            thresholds=thresholds,
            top_workflows=top_workflows,
            ranking_label=rank_label,
            output_dir=output_dir,
            skip_existing=skip_existing
        )
        all_collected_metrics.extend(collected)
        plots_generated += len(collected)

    if all_collected_metrics:
        summary_path = output_dir / "correlation_metrics_summary.csv"
        pd.DataFrame(all_collected_metrics).to_csv(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)

    return plots_generated


def main():
    """Main entry point for CLI."""
    args = parse_args()

    targets = None if args.targets.lower() == 'auto' else [t.strip() for t in args.targets.split(',')]
    metrics = [m.strip() for m in args.metrics.split(',')]
    thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    ranking_metrics = parse_ranking_metrics(args.ranking_metrics)
    skip_existing = not args.no_skip_existing

    result = run_ave_analysis(
        split_dir=args.split_dir,
        full_dir=args.full_dir,
        output_dir=args.output_dir,
        targets=targets,
        metrics=metrics,
        thresholds=thresholds,
        ranking_metrics=ranking_metrics,
        workers=args.workers,
        skip_existing=skip_existing
    )

    if result == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
