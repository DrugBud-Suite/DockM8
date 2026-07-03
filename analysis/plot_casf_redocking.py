"""
Summary figure for the CASF-2016 re-docking (pose-reproduction) benchmark.

Reads the outputs of analysis/casf_redocking_benchmark.py and produces a single
two-panel figure:
  (a) per-engine redocking success: top-scored ("native top-1") pose within 2 A
      of the crystal, with the best pose sampled (sampling ceiling) overlaid;
  (b) engine x pose-selection scoring-function heatmap of the same 2 A success
      rate. Scoring functions with possible PDBbind/CASF training overlap are
      marked with an asterisk.

Usage:
    conda activate dockm8_v1
    python analysis/plot_casf_redocking.py \
        --results analysis/data/casf_redocking_results.csv \
        --out analysis/data/figG1.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Display order/labels and leakage annotation.
ENGINES = ["SMINA", "GNINA_GPU", "PLANTS", "QVINA2", "QVINAW"]
ENGINE_LABELS = {"SMINA": "SMINA", "GNINA_GPU": "GNINA", "PLANTS": "PLANTS",
                 "QVINA2": "QVINA2", "QVINAW": "QVINA-W"}
# Scoring functions ordered leakage-free first, then those with possible
# PDBbind/CASF training overlap (marked '*').
SF_ORDER = ["AD4", "CHEMPLP", "PLP", "Vinardo", "LinF9", "KORP-PL", "ConvexPLR",
            "GNINA-Affinity", "CNN-Score", "CNN-Affinity", "NNScore", "RFScoreVS",
            "RTMScore", "GenScore-scoring", "GenScore-docking", "GenScore-balanced"]
LEAKAGE_PRONE = {"GNINA-Affinity", "CNN-Score", "CNN-Affinity", "NNScore", "RFScoreVS",
                 "RTMScore", "GenScore-scoring", "GenScore-docking", "GenScore-balanced"}


def per_engine_stats(df: pd.DataFrame) -> pd.DataFrame:
    gm = df["graph_match"].astype(str).str.lower().isin(["true", "1", "1.0"])
    ok = df[df["status"].astype(str).str.startswith("ok") & gm]
    rows = []
    for e in ENGINES:
        top1 = pd.to_numeric(ok.get(f"{e}__native_top1_rmsd"), errors="coerce").dropna()
        best = pd.to_numeric(ok.get(f"{e}__best_sampled_rmsd"), errors="coerce").dropna()
        rows.append({
            "engine": ENGINE_LABELS[e], "n": len(top1),
            "top1_2A": 100 * (top1 <= 2).mean(),
            "best_2A": 100 * (best <= 2).mean(),
            "median_rmsd": float(top1.median()),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Plot CASF-2016 redocking benchmark summary")
    p.add_argument("--results", type=Path, default=Path("analysis/data/casf_redocking_results.csv"))
    p.add_argument("--matrix", type=Path, default=None,
                   help="engine x SF success matrix CSV (default: <results>_matrix_success2A.csv)")
    p.add_argument("--out", type=Path, default=Path("analysis/data/figG1.png"))
    args = p.parse_args()
    matrix_path = args.matrix or args.results.with_name(args.results.stem + "_matrix_success2A.csv")

    df = pd.read_csv(args.results).drop_duplicates("id", keep="last")
    stats = per_engine_stats(df)
    mat = pd.read_csv(matrix_path, index_col=0)
    mat.index = [ENGINE_LABELS.get(i, i) for i in mat.index]
    mat = mat.reindex(index=[ENGINE_LABELS[e] for e in ENGINES],
                      columns=[c for c in SF_ORDER if c in mat.columns])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [0.85, 1.9]})

    # --- Panel (a): per-engine bars ---
    x = np.arange(len(stats))
    w = 0.38
    ax1.bar(x - w / 2, stats["top1_2A"], w, color="#2c6fbb", label="Top-scored pose")
    ax1.bar(x + w / 2, stats["best_2A"], w, color="#b6cce8", label="Best pose sampled")
    for xi, r in zip(x, stats.itertuples()):
        ax1.text(xi - w / 2, r.top1_2A + 1.2, f"{r.top1_2A:.0f}", ha="center", va="bottom", fontsize=8)
        ax1.text(xi + w / 2, r.best_2A + 1.2, f"{r.best_2A:.0f}", ha="center", va="bottom", fontsize=8, color="#444")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats["engine"], rotation=20, ha="right")
    ax1.set_ylabel("Poses within 2 Å of crystal (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("(a) Re-docking success per engine")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)

    # --- Panel (b): engine x SF heatmap ---
    data = mat.to_numpy(dtype=float)
    im = ax2.imshow(data, cmap="YlGnBu", vmin=0, vmax=max(60, np.nanmax(data)), aspect="auto")
    ax2.set_xticks(np.arange(mat.shape[1]))
    ax2.set_xticklabels(list(mat.columns), rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(np.arange(mat.shape[0]))
    ax2.set_yticklabels(mat.index, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=12,
                         color="white" if v > 0.6 * np.nanmax(data) else "black")
    ax2.set_title("(b) Pose-selection: top pose within 2 Å (%), per engine × scoring function")
    cb = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
    cb.set_label("% ≤ 2 Å", fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")
    print("\nPer-engine summary:")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
