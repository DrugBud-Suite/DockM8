#!/usr/bin/env python3
"""
DockM8 Analysis Master Orchestrator

This script provides a unified command-line interface for generating all
DockM8 analysis plots. It supports individual plot types or running the
complete analysis pipeline.

Usage:
    python -m analysis.run_all --help
    python -m analysis.run_all <command> --help

    # Run individual analyses
    python -m analysis.run_all aggregate --base-path /path/to/zenodo/data
    python -m analysis.run_all extract
    python -m analysis.run_all boxplots --type internal
    python -m analysis.run_all boxplots --type literature
    python -m analysis.run_all heatmaps --type docking
    python -m analysis.run_all heatmaps --type interaction
    python -m analysis.run_all barplots --type frequency
    python -m analysis.run_all barplots --type impact
    python -m analysis.run_all ave --split-dir PATH --full-dir PATH

    # Run complete pipeline (aggregate → extract → all plots)
    python -m analysis.run_all all --base-path /path/to/zenodo/data
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import (
    get_base_path, get_output_dir, get_aggregated_dir,
    get_dockm8_results_dir, get_literature_dir,
    DATASETS, METRICS, THRESHOLDS,
    parse_list_arg
)


def cmd_aggregate(args):
    """Run data aggregation."""
    from .data_aggregation import aggregate_dataset_results

    base_path = get_base_path(args.base_path)
    output_dir = get_output_dir(args.output_dir)
    datasets = parse_list_arg(args.datasets, DATASETS)

    print("=" * 60)
    print("Data Aggregation")
    print("=" * 60)
    print(f"Base path:    {base_path}")
    print(f"Output:       {get_aggregated_dir(output_dir)}")
    print(f"Datasets:     {datasets}")

    for dataset in datasets:
        print(f"\n--- Aggregating {dataset} ---")
        aggregate_dataset_results(base_path, dataset)

    print("\nAggregation complete.")


def cmd_extract(args):
    """Extract consolidated multi-metric CSVs from aggregated pivot tables."""
    # Memory-lean, vectorized extractor (~13x faster than the original per-row loops,
    # peak RSS ~25 GB vs ~100 GB+swap). Output is parity-identical to data_extraction's
    # run_extraction and surfaces the correct GenScore variants (balanced = GT_ft_0.5)
    # + RTMScore for all datasets, from the shared CUSTOM_WORKFLOWS_RESULTS map.
    from .fast_dockm8_extract import run_extraction_fast

    output_dir = get_output_dir(args.output_dir)
    datasets = parse_list_arg(args.datasets, DATASETS)

    print("=" * 60)
    print("Data Extraction")
    print("=" * 60)
    print(f"Aggregated:   {get_aggregated_dir(output_dir)}")
    print(f"DockM8 data:  {get_dockm8_results_dir(output_dir)}")
    print(f"Datasets:     {datasets}")

    run_extraction_fast(
        aggregated_dir=get_aggregated_dir(output_dir),
        dockm8_results_dir=get_dockm8_results_dir(output_dir),
        datasets=datasets,
    )

    print("\nExtraction complete.")


def cmd_boxplots_internal(args):
    """Run internal boxplot generation."""
    from .internal_boxplots import generate_internal_boxplots

    output_dir = get_output_dir(args.output_dir)
    dockm8_results_dir = get_dockm8_results_dir(output_dir)
    datasets = parse_list_arg(args.datasets, ['DEKOIS', 'DUD-E', 'Lit-PCBA'])
    metrics = parse_list_arg(args.metrics, ['NEF_calc', 'REF', 'PM', 'AUC_ROC', 'BEDROC'])

    print("=" * 60)
    print("Internal Boxplots (DockM8 SFs vs Consensus)")
    print("=" * 60)

    generate_internal_boxplots(
        dockm8_results_dir=dockm8_results_dir,
        output_dir=output_dir / 'plots' / 'internal',
        datasets=datasets,
        metrics=metrics,
        threshold=args.threshold,
        dpi=args.dpi
    )


def cmd_boxplots_literature(args):
    """Run literature comparison boxplot generation."""
    from .literature_boxplots import generate_all_literature_plots
    from .data_loader import load_all_literature_datasets
    from .config import NUMERIC_COLUMNS
    from .plot_helpers import setup_plot_style

    output_dir = get_output_dir(args.output_dir)
    lit_data_path = args.lit_data_path or (get_literature_dir() / 'lit_calc_data_rounded')
    dockm8_data_path = args.dockm8_data_path or get_dockm8_results_dir(output_dir)

    from .literature_boxplots import DATASET_FILES

    dataset_info = {}
    for dataset_name, files in DATASET_FILES.items():
        dataset_info[dataset_name] = {
            'Literature': lit_data_path / files['Literature'],
            'DockM8': dockm8_data_path / files['DockM8']
        }

    print("=" * 60)
    print("Literature Comparison Boxplots")
    print("=" * 60)
    print(f"Literature data: {lit_data_path}")
    print(f"DockM8 data:     {dockm8_data_path}")
    print(f"Output:          {output_dir}")

    setup_plot_style(font_scale=1.4)
    datasets = load_all_literature_datasets(dataset_info, NUMERIC_COLUMNS)

    output_path = output_dir / 'plots' / 'literature_per_dataset'
    output_path.mkdir(parents=True, exist_ok=True)

    generate_all_literature_plots(datasets, output_path, args.dpi)


def cmd_heatmaps_interaction(args):
    """Run interaction heatmap generation."""
    from .interaction_plots import generate_interaction_plots

    output_dir = get_output_dir(args.output_dir)
    metrics = parse_list_arg(args.metrics, METRICS)
    thresholds = parse_list_arg(args.thresholds, THRESHOLDS)
    datasets = parse_list_arg(args.datasets, DATASETS)

    print("=" * 60)
    print("Interaction Heatmaps (NumSFs vs Consensus)")
    print("=" * 60)

    generate_interaction_plots(
        aggregated_dir=get_aggregated_dir(output_dir),
        output_dir=output_dir,
        metrics=metrics,
        thresholds=thresholds,
        datasets=datasets
    )


def cmd_heatmaps_docking(args):
    """Run docking selection heatmap generation."""
    from .docking_selection import generate_docking_selection_plots

    output_dir = get_output_dir(args.output_dir)
    metrics = parse_list_arg(args.metrics, METRICS)
    thresholds = parse_list_arg(args.thresholds, THRESHOLDS)
    datasets = parse_list_arg(args.datasets, DATASETS)

    print("=" * 60)
    print("Docking/Selection Heatmaps")
    print("=" * 60)

    generate_docking_selection_plots(
        aggregated_dir=get_aggregated_dir(output_dir),
        output_dir=output_dir,
        metrics=metrics,
        thresholds=thresholds,
        datasets=datasets
    )


def cmd_barplots_frequency(args):
    """Run SF frequency bar plot generation."""
    from .sf_frequency import generate_sf_frequency_plots

    output_dir = get_output_dir(args.output_dir)
    metrics = parse_list_arg(args.metrics, METRICS)
    thresholds = parse_list_arg(args.thresholds, THRESHOLDS)
    datasets = parse_list_arg(args.datasets, DATASETS)

    print("=" * 60)
    print("Scoring Function Frequency Bar Plots")
    print("=" * 60)

    generate_sf_frequency_plots(
        aggregated_dir=get_aggregated_dir(output_dir),
        output_dir=output_dir,
        metrics=metrics,
        thresholds=thresholds,
        datasets=datasets
    )


def cmd_barplots_impact(args):
    """Run SF impact bar plot generation."""
    from .sf_impact import generate_sf_impact_plots

    output_dir = get_output_dir(args.output_dir)
    metrics = parse_list_arg(args.metrics, METRICS)
    thresholds = parse_list_arg(args.thresholds, THRESHOLDS)
    datasets = parse_list_arg(args.datasets, DATASETS)

    print("=" * 60)
    print("Scoring Function Impact Bar Plots")
    print("=" * 60)

    generate_sf_impact_plots(
        aggregated_dir=get_aggregated_dir(output_dir),
        output_dir=output_dir,
        metrics=metrics,
        thresholds=thresholds,
        datasets=datasets
    )


def cmd_ave(args):
    """Run AVE analysis."""
    from .ave_analysis import run_ave_analysis
    from .utils import parse_ranking_metrics

    output_dir = get_output_dir(args.output_dir) / 'AVE_analysis'

    targets = None if args.targets.lower() == 'auto' else [t.strip() for t in args.targets.split(',')]
    metrics = [m.strip() for m in args.metrics.split(',')]
    thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    ranking_metrics = parse_ranking_metrics(args.ranking_metrics)

    run_ave_analysis(
        split_dir=args.split_dir,
        full_dir=args.full_dir,
        output_dir=output_dir,
        targets=targets,
        metrics=metrics,
        thresholds=thresholds,
        ranking_metrics=ranking_metrics,
        workers=args.workers,
        skip_existing=not args.no_skip_existing
    )


def cmd_all(args):
    """Run complete analysis pipeline."""
    print("=" * 60)
    print("Running Complete Analysis Pipeline")
    print("=" * 60)

    base_path = get_base_path(args.base_path)
    output_dir = get_output_dir(args.output_dir)

    print(f"\nBase path: {base_path}")
    print(f"Output directory: {output_dir}")
    print()

    class MockArgs:
        pass

    mock_args = MockArgs()
    mock_args.base_path = args.base_path
    mock_args.output_dir = args.output_dir
    mock_args.datasets = args.datasets
    mock_args.metrics = args.metrics
    mock_args.thresholds = args.thresholds
    mock_args.dpi = args.dpi
    mock_args.threshold = 1.0
    mock_args.lit_data_path = None
    mock_args.dockm8_data_path = None

    steps = [
        ("Data Aggregation", cmd_aggregate),
        ("Data Extraction", cmd_extract),
        ("Docking/Selection Heatmaps", cmd_heatmaps_docking),
        ("Internal Boxplots", cmd_boxplots_internal),
        ("Literature Boxplots", cmd_boxplots_literature),
        ("SF Frequency Plots", cmd_barplots_frequency),
        ("SF Impact Plots", cmd_barplots_impact),
        ("Interaction Heatmaps", cmd_heatmaps_interaction),
    ]

    # Aggregation and extraction produce the inputs every plot step reads; if either
    # fails, the plots would run on missing/stale data and emit silently-wrong figures.
    # Abort on those; a failed individual plot step is non-fatal (others still run).
    prerequisite_steps = {"Data Aggregation", "Data Extraction"}

    for step_name, step_func in steps:
        print(f"\n{'#'*60}")
        print(f"# {step_name}")
        print(f"{'#'*60}\n")
        try:
            step_func(mock_args)
        except Exception as e:
            if step_name in prerequisite_steps:
                print(f"ERROR: {step_name} failed — aborting pipeline (plots depend on its output): {e}")
                raise
            print(f"Warning: {step_name} failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print("Complete Pipeline Finished")
    print("=" * 60)


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument('--base-path', type=Path, default=None, help='Base path to results data')
    common_args.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    common_args.add_argument('--datasets', type=str, default=None, help='Comma-separated datasets')
    common_args.add_argument('--metrics', type=str, default=None, help='Comma-separated metrics')
    common_args.add_argument('--thresholds', type=str, default=None, help='Comma-separated thresholds')
    common_args.add_argument('--dpi', type=int, default=300, help='DPI for saved figures')

    agg_parser = subparsers.add_parser('aggregate', parents=[common_args], help='Aggregate data into parquet files')
    agg_parser.set_defaults(func=cmd_aggregate)

    ext_parser = subparsers.add_parser('extract', parents=[common_args], help='Extract consolidated CSVs from aggregated data')
    ext_parser.set_defaults(func=cmd_extract)

    boxplots_parser = subparsers.add_parser('boxplots', parents=[common_args], help='Generate boxplots')
    boxplots_parser.add_argument('--type', choices=['internal', 'literature'], required=True,
                                  help='Type of boxplot to generate')
    boxplots_parser.add_argument('--threshold', type=float, default=1.0, help='Percentile threshold (for internal)')
    boxplots_parser.add_argument('--lit-data-path', type=Path, default=None, help='Literature data path')
    boxplots_parser.add_argument('--dockm8-data-path', type=Path, default=None, help='DockM8 data path')

    def boxplots_dispatch(args):
        if args.type == 'internal':
            cmd_boxplots_internal(args)
        else:
            cmd_boxplots_literature(args)
    boxplots_parser.set_defaults(func=boxplots_dispatch)

    heatmaps_parser = subparsers.add_parser('heatmaps', parents=[common_args], help='Generate heatmaps')
    heatmaps_parser.add_argument('--type', choices=['interaction', 'docking'], required=True,
                                  help='Type of heatmap to generate')

    def heatmaps_dispatch(args):
        if args.type == 'interaction':
            cmd_heatmaps_interaction(args)
        else:
            cmd_heatmaps_docking(args)
    heatmaps_parser.set_defaults(func=heatmaps_dispatch)

    barplots_parser = subparsers.add_parser('barplots', parents=[common_args], help='Generate bar plots')
    barplots_parser.add_argument('--type', choices=['frequency', 'impact'], required=True,
                                  help='Type of bar plot to generate')

    def barplots_dispatch(args):
        if args.type == 'frequency':
            cmd_barplots_frequency(args)
        else:
            cmd_barplots_impact(args)
    barplots_parser.set_defaults(func=barplots_dispatch)

    ave_parser = subparsers.add_parser('ave', help='Run AVE analysis')
    ave_parser.add_argument('--split-dir', type=Path, required=True, help='AVE split analysis directory')
    ave_parser.add_argument('--full-dir', type=Path, required=True, help='Full dataset results directory')
    ave_parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    ave_parser.add_argument('--targets', type=str, default='auto', help='Target names (comma-sep or "auto")')
    ave_parser.add_argument('--metrics', type=str, default='auc_roc,bedroc,ref', help='Metrics to analyze')
    ave_parser.add_argument('--thresholds', type=str, default='1.0', help='Thresholds for threshold-dependent metrics')
    ave_parser.add_argument('--ranking-metrics', type=str, default='ref:1.0,bedroc', help='Ranking metrics')
    ave_parser.add_argument('--workers', type=int, default=None, help='Parallel workers')
    ave_parser.add_argument('--no-skip-existing', action='store_true', help='Regenerate existing plots')
    ave_parser.set_defaults(func=cmd_ave)

    all_parser = subparsers.add_parser('all', parents=[common_args], help='Run complete pipeline')
    all_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed error traces')
    all_parser.set_defaults(func=cmd_all)

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
