#!/usr/bin/env python3
"""Literature-comparison boxplots with BOTH DockM8 consensus variants in one plot.

Standard literature boxplots show DockM8-best/median computed WITH RFScoreVS. This runner
additionally overlays the no-RFScoreVS DockM8 consensus (DockM8-best/median recomputed with
every RFScoreVS workflow excluded) as extra bars in the SAME plot, labelled
"DockM8-best (no RFScoreVS)" / "DockM8-median (no RFScoreVS)" and drawn in a distinct colour
(see DOCKM8_NORF_HIGHLIGHT_COLOR in literature_boxplots), so the RFScoreVS impact on the
consensus is directly visible alongside the literature baselines and the @DockM8 single-SF bars.

Pass --norf-csv to include the no-RFScoreVS bars (DEKOIS/DUD-E). Omit it (Lit-PCBA) to just
regenerate the standard plot with the corrected highlighting.

Reads:  --dockm8-csv (standard {DATASET}_dockm8_results.csv)
        --norf-csv   (optional {DATASET}_dockm8_results_noRFScoreVS.csv)
        analysis/data/literature/lit_calc_data_rounded/<dataset>_complete_metrics.csv
Writes: <out>/<DATASET>_*_vertical_1*.png
"""
# ruff: noqa: D103, D205, D403, E402  (analysis runner; module documented)

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.config import NUMERIC_COLUMNS, get_literature_dir
from analysis.data_loader import load_all_literature_datasets
from analysis.literature_boxplots import DATASET_FILES, generate_all_literature_plots
from analysis.plot_helpers import setup_plot_style

CONSENSUS = {"DockM8-best": "DockM8-best (no RFScoreVS)",
             "DockM8-median": "DockM8-median (no RFScoreVS)"}


def build_combined_csv(dockm8_csv: Path, norf_csv: Path | None, out_csv: Path) -> Path:
    """Standard DockM8 rows + the no-RFScoreVS consensus rows (renamed), one CSV."""
    std = pd.read_csv(dockm8_csv)
    if norf_csv is not None:
        norf = pd.read_csv(norf_csv)
        extra = norf[norf["Model"].isin(CONSENSUS)].copy()
        extra["Model"] = extra["Model"].map(CONSENSUS)
        combined = pd.concat([std, extra], ignore_index=True)
    else:
        combined = std
    combined.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FILES))
    parser.add_argument("--dockm8-csv", type=Path, required=True,
                        help="standard {DATASET}_dockm8_results.csv")
    parser.add_argument("--norf-csv", type=Path, default=None,
                        help="optional {DATASET}_dockm8_results_noRFScoreVS.csv to overlay")
    parser.add_argument("--lit-metrics", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True, help="output plots dir")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if not args.dockm8_csv.is_file():
        raise SystemExit(f"missing {args.dockm8_csv}")
    if args.norf_csv is not None and not args.norf_csv.is_file():
        raise SystemExit(f"missing {args.norf_csv}")

    lit_metrics = args.lit_metrics or (
        get_literature_dir() / "lit_calc_data_rounded" / DATASET_FILES[args.dataset]["Literature"]
    )

    tmp = Path(tempfile.mkdtemp()) / f"{args.dataset}_dockm8_results_combined.csv"
    build_combined_csv(args.dockm8_csv, args.norf_csv, tmp)
    overlay = "with no-RFScoreVS overlay" if args.norf_csv else "standard only"
    print(f"{args.dataset} literature boxplots ({overlay})\n  DockM8: {args.dockm8_csv}"
          + (f"\n  +noRFScoreVS: {args.norf_csv}" if args.norf_csv else "")
          + f"\n  Literature: {lit_metrics}")

    setup_plot_style(font_scale=1.4)
    dataset_info = {args.dataset: {"Literature": lit_metrics, "DockM8": tmp}}
    datasets = load_all_literature_datasets(dataset_info, NUMERIC_COLUMNS)

    args.out.mkdir(parents=True, exist_ok=True)
    n = generate_all_literature_plots(datasets, args.out, args.dpi)
    print(f"\nGenerated {n} plot(s) in {args.out}")


if __name__ == "__main__":
    main()
