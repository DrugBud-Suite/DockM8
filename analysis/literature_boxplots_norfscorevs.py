#!/usr/bin/env python3
"""Literature-comparison boxplots using the no-RFScoreVS DockM8 variant.

The standard literature boxplot path (run_all boxplots --type literature) only loads
`{DATASET}_dockm8_results.csv` (DockM8-best/median computed WITH RFScoreVS). The extraction
step can also emit `{DATASET}_dockm8_results_noRFScoreVS.csv` (DockM8-best/median recomputed
with every RFScoreVS workflow excluded), but nothing plots it. This runner wires that file
into the existing literature-boxplot generator and writes to a separate
`plots/literature_per_dataset_noRFScoreVS/` subdir, so it never collides with the standard plots.

Works for any dataset (DEKOIS, DUD-E, Lit-PCBA). If the noRFScoreVS CSV is missing and
`--aggregated-dir` is given, it is generated from the aggregated parquet pivots (memory-lean,
RFScoreVS excluded) before plotting.

Reads:  <noRFScoreVS results CSV>
        analysis/data/literature/lit_calc_data_rounded/<dataset>_complete_metrics.csv
Writes: <out>/<DATASET>_*_vertical_1*.png

Examples:
    # DUD-E (back-compatible default; reads <output-dir>/dockm8_results/)
    python analysis/literature_boxplots_norfscorevs.py --output-dir results/output

    # DEKOIS, generating the noRFScoreVS CSV from corrected aggregates
    python analysis/literature_boxplots_norfscorevs.py --dataset DEKOIS \
        --aggregated-dir /mnt/e/FINAL_RESULTS/corrected_final/DEKOIS_aggregated_results \
        --norf-csv /mnt/e/FINAL_RESULTS/corrected_final/results/dockm8_extracted_data/DEKOIS_dockm8_results_noRFScoreVS.csv \
        --out /mnt/e/FINAL_RESULTS/corrected_final/results/plots/DEKOIS/plots/literature_per_dataset_noRFScoreVS
"""
# ruff: noqa: D103, D205, D403, E402  (analysis runner; module documented)

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.config import NUMERIC_COLUMNS, get_dockm8_results_dir, get_literature_dir, get_output_dir
from analysis.data_loader import load_all_literature_datasets
from analysis.literature_boxplots import DATASET_FILES, generate_all_literature_plots
from analysis.plot_helpers import setup_plot_style

# noRFScoreVS results-CSV filename per dataset (matches the extraction output naming).
NORF_CSV_NAME = {
    "DEKOIS": "DEKOIS_dockm8_results_noRFScoreVS.csv",
    "DUD-E": "DUD-E_dockm8_results_noRFScoreVS.csv",
    "Lit-PCBA": "lit-pcba_dockm8_results_noRFScoreVS.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(NORF_CSV_NAME), default="DUD-E",
                        help="dataset to plot (default: DUD-E)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="results output root (default: results/output/); used to derive "
                             "--norf-csv and --out when those are not given")
    parser.add_argument("--norf-csv", type=Path, default=None,
                        help="explicit path to the noRFScoreVS results CSV "
                             "(default: <output-dir>/dockm8_results/<DATASET>_dockm8_results_noRFScoreVS.csv)")
    parser.add_argument("--aggregated-dir", type=Path, default=None,
                        help="dataset *_aggregated_results dir; if given and --norf-csv is missing, "
                             "the noRFScoreVS CSV is generated from it (RFScoreVS excluded)")
    parser.add_argument("--lit-metrics", type=Path, default=None,
                        help="literature complete-metrics CSV (default: from analysis/data/literature)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output plots dir (default: <output-dir>/plots/literature_per_dataset_noRFScoreVS)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output_dir = get_output_dir(args.output_dir)
    norf_csv = args.norf_csv or (get_dockm8_results_dir(output_dir) / NORF_CSV_NAME[args.dataset])

    # Generate the noRFScoreVS CSV on demand when absent and aggregates are available.
    if not norf_csv.is_file():
        if args.aggregated_dir is None:
            raise SystemExit(
                f"missing {norf_csv}; pass --aggregated-dir <dataset_aggregated_results> to generate it, "
                f"or run the extraction step that produces {NORF_CSV_NAME[args.dataset]}"
            )
        from analysis.fast_dockm8_extract import run_analysis_filtered_fast
        norf_csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"Generating {norf_csv.name} from {args.aggregated_dir} (RFScoreVS excluded)...")
        run_analysis_filtered_fast(
            data_dir=str(args.aggregated_dir),
            output_dir=str(norf_csv.parent),
            output_filename=norf_csv.name,
            excluded_sf="RFScoreVS",
        )

    lit_metrics = args.lit_metrics or (
        get_literature_dir() / "lit_calc_data_rounded" / DATASET_FILES[args.dataset]["Literature"]
    )
    dataset_info = {args.dataset: {"Literature": lit_metrics, "DockM8": norf_csv}}

    print(f"{args.dataset} noRFScoreVS literature boxplots\n  DockM8: {norf_csv}\n  Literature: {lit_metrics}")
    setup_plot_style(font_scale=1.4)
    datasets = load_all_literature_datasets(dataset_info, NUMERIC_COLUMNS)

    out_path = args.out or (output_dir / "plots" / "literature_per_dataset_noRFScoreVS")
    out_path.mkdir(parents=True, exist_ok=True)
    n = generate_all_literature_plots(datasets, out_path, args.dpi)
    print(f"\nGenerated {n} plot(s) in {out_path}")


if __name__ == "__main__":
    main()
