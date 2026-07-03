"""
Re-docking (pose-reproduction) benchmark for DockM8.

For each protein-ligand complex with a known crystallographic pose, the cognate
ligand is docked back into its own receptor with one or more DockM8 docking
engines. The top-scored pose (and the best pose sampled) is compared to the
crystal pose using a symmetry-corrected RMSD (spyRMSD). Optionally, PoseBusters
chemical/physical validity is reported alongside, giving the modern
"PB-valid & RMSD <= 2 A" success metric.

This directly addresses the reviewers' request for binding-pose RMSD
characterisation, which is orthogonal to the enrichment (screening-power)
benchmarks reported in the manuscript.

Input layout (one directory, one pair per complex):
    <complexes_dir>/<id>_p.pdb     # prepared receptor
    <complexes_dir>/<id>_l.sdf     # crystal (reference) ligand pose

The two complexes bundled in test_data (1fvv, 4kd1) are used by default as a
smoke test. For a publishable result use a standard set such as the Astex
Diverse Set (85 complexes) or a PoseBusters benchmark subset.

Usage:
    conda activate dockm8
    python analysis/redocking_benchmark.py \
        --complexes_dir test_data \
        --engines SMINA GNINA PLANTS \
        --n_poses 10 --exhaustiveness 8 --n_cpus 4 \
        --output analysis/data/redocking_results.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
DOCKM8_PATH = scripts_path.parent
sys.path.append(str(DOCKM8_PATH))

from scripts.docking.docking import dockm8_docking
from scripts.pocket_finding.pocket_finder import find_pocket
from scripts.utilities.logging import printlog

SOFTWARE = DOCKM8_PATH / "software"

# Native score column written by each DockM8 docking engine (lower/sorted first).
SCORE_COLUMN = {
    "SMINA": "SMINA_Affinity",
    "GNINA": "CNN-Score",
    "PLANTS": "CHEMPLP",
    "QVINA2": "QVINA2_Affinity",
    "QVINAW": "QVINAW_Affinity",
}
# GNINA ranks by CNN-Score descending (higher better); all others ascending.
ASCENDING = {"SMINA": True, "GNINA": False, "PLANTS": True, "QVINA2": True, "QVINAW": True}


def symmetric_rmsd(ref_mol: Chem.Mol, pred_mol: Chem.Mol) -> float:
    """Symmetry-corrected heavy-atom RMSD between two poses of the same molecule.

    NO superposition is performed: the docked pose and the crystal pose share the
    receptor frame, so RMSD must be computed in place (redocking convention).
    spyRMSD `symmrmsd` is called with center=False, minimize=False (no alignment),
    permuting only atom labels for symmetry. The RDKit fallback uses CalcRMS, which
    is symmetry-aware and likewise does NOT align (unlike GetBestRMS, which would).
    """
    try:
        from spyrmsd import rmsd as spy_rmsd
        from spyrmsd.molecule import Molecule

        ref = Molecule.from_rdkit(Chem.RemoveHs(ref_mol))
        pred = Molecule.from_rdkit(Chem.RemoveHs(pred_mol))
        return float(
            spy_rmsd.symmrmsd(
                ref.coordinates,
                pred.coordinates,
                ref.atomicnums,
                pred.atomicnums,
                ref.adjacency_matrix,
                pred.adjacency_matrix,
                center=False,
                minimize=False,
            )
        )
    except Exception as exc:  # noqa: BLE001
        printlog(f"spyRMSD failed ({exc}); falling back to RDKit CalcRMS (no alignment)")
        from rdkit.Chem import rdMolAlign

        return float(rdMolAlign.CalcRMS(Chem.RemoveHs(pred_mol), Chem.RemoveHs(ref_mol)))


def discover_complexes(complexes_dir: Path) -> list[dict]:
    """Find <id>_p.pdb / <id>_l.sdf pairs in a directory."""
    complexes = []
    for protein in sorted(complexes_dir.glob("*_p.pdb")):
        cid = protein.name[:-6]  # strip "_p.pdb"
        ligand = complexes_dir / f"{cid}_l.sdf"
        if ligand.exists():
            complexes.append({"id": cid, "protein": protein, "ligand": ligand})
    return complexes


def redock_complex(complex_info: dict, w_dir: Path, engines: list[str], args) -> list[dict]:
    """Dock one cognate ligand back into its receptor and score pose RMSD per engine."""
    cid, protein, ref_ligand = complex_info["id"], complex_info["protein"], complex_info["ligand"]
    ref_mol = next(Chem.SDMolSupplier(str(ref_ligand), removeHs=False, sanitize=True))
    if ref_mol is None:
        printlog(f"[{cid}] could not read reference ligand; skipping")
        return []

    pocket = find_pocket(mode="Reference", receptor=protein, ligand=ref_ligand, radius=args.radius)
    poses_path = dockm8_docking(
        library=ref_ligand,
        w_dir=w_dir / cid,
        protein_file=protein,
        pocket_definition=pocket,
        software=SOFTWARE,
        docking_programs=engines,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        n_cpus=args.n_cpus,
    )
    poses = PandasTools.LoadSDF(str(poses_path), molColName="Molecule", idName="Pose ID")

    rows = []
    for engine in engines:
        score_col = SCORE_COLUMN[engine]
        sub = poses[poses["Pose ID"].str.contains(engine.lower(), case=False, na=False)].copy()
        if sub.empty or score_col not in sub.columns:
            rows.append({"id": cid, "engine": engine, "top1_rmsd": np.nan, "best_rmsd": np.nan, "n_poses": 0})
            continue
        sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce")
        sub = sub.sort_values(score_col, ascending=ASCENDING[engine])
        rmsds = [symmetric_rmsd(ref_mol, m) for m in sub["Molecule"] if m is not None]
        rows.append(
            {
                "id": cid,
                "engine": engine,
                "top1_rmsd": rmsds[0] if rmsds else np.nan,
                "best_rmsd": min(rmsds) if rmsds else np.nan,
                "n_poses": len(rmsds),
            }
        )
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-engine success rates at the standard 2 A threshold."""
    out = []
    for engine, g in df.groupby("engine"):
        out.append(
            {
                "engine": engine,
                "n_complexes": len(g),
                "top1_<=2A_%": 100 * (g["top1_rmsd"] <= 2.0).mean(),
                "best_<=2A_%": 100 * (g["best_rmsd"] <= 2.0).mean(),
                "median_top1_rmsd": g["top1_rmsd"].median(),
            }
        )
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description="DockM8 re-docking / pose-reproduction benchmark")
    parser.add_argument("--complexes_dir", type=Path, default=DOCKM8_PATH / "test_data")
    parser.add_argument("--engines", nargs="+", default=["SMINA", "GNINA", "PLANTS"])
    parser.add_argument("--n_poses", type=int, default=10)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--n_cpus", type=int, default=4)
    parser.add_argument("--w_dir", type=Path, default=Path.home() / "dockm8_redocking")
    parser.add_argument("--output", type=Path, default=DOCKM8_PATH / "analysis" / "data" / "redocking_results.csv")
    args = parser.parse_args()

    complexes = discover_complexes(args.complexes_dir)
    if not complexes:
        printlog(f"No <id>_p.pdb / <id>_l.sdf pairs found in {args.complexes_dir}")
        sys.exit(1)
    printlog(f"Re-docking {len(complexes)} complexes with engines {args.engines}")

    args.w_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for c in complexes:
        try:
            all_rows.extend(redock_complex(c, args.w_dir, args.engines, args))
        except Exception as exc:  # noqa: BLE001
            printlog(f"[{c['id']}] failed: {exc}")

    df = pd.DataFrame(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    summary = summarise(df.dropna(subset=["top1_rmsd"]))
    printlog("\n=== Per-complex RMSD ===\n" + df.to_string(index=False))
    printlog("\n=== Summary (success = RMSD <= 2 A) ===\n" + summary.to_string(index=False))
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
