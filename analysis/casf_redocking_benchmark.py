"""
CASF-2016 cognate re-docking (pose-reproduction) benchmark for DockM8.

Full DockM8 pipeline per complex:
  protein prep -> pocket from crystal ligand -> library prep (conformer
  REGENERATED, crystal coords discarded) -> docking (N engines) ->
  pose selection (M scoring functions) -> symmetry-corrected RMSD vs crystal.

For EACH engine independently (no pooling), each of the M pose-selection
scoring functions picks the best-SCORED pose among that engine's poses; the
RMSD of that selected pose (to the crystal) is recorded -> an engine x SF
matrix. Per-engine native top-1 and best-sampled (sampling ceiling) are also
recorded.

RMSD is symmetry-corrected and computed WITHOUT superposition (see
analysis/redocking_benchmark.symmetric_rmsd): the docked pose and the crystal
share the receptor frame.

Honest accounting: the selected pose is chosen on SCORE alone; its RMSD is
reported as-is (NaN if undefined). NaN RMSD is EXCLUDED from success-rate
denominators, never silently counted as a >2 A miss. Complexes whose regenerated
ligand graph/stereochemistry differs from the crystal (graph_match == False) are
excluded from headline rates and reported separately.

PARALLELISM: complexes are processed concurrently (--workers), each using a
single CPU internally (--n_cpus 1). GPU steps (GNINA_GPU docking, RTMScore and
GenScore rescoring) are gated by a shared semaphore (--gpu_slots) so at most
--gpu_slots complexes touch the GPU at once; everything else runs fully parallel.

Outputs (next to --output):
  <out>.csv  <out>_long.csv  <out>_matrix_success2A.csv  <out>_matrix_counts.csv
  <out>.run.json (provenance: args, seeds, versions, git SHA, counts)

RESUMABLE: re-invoking skips complexes already present with an "ok" status.

Reproducibility env (required):
    conda activate dockm8_v1
    export PYTHONNOUSERSITE=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

Full run:
    python analysis/casf_redocking_benchmark.py \
        --casf_dir /home/tony/Datasets/CASF-2016/coreset \
        --engines SMINA GNINA_GPU PLANTS QVINA2 QVINAW \
        --protonation protoss --workers 30 --gpu_slots 4 --n_cpus 1 \
        --output analysis/data/casf_redocking_results.csv
"""

import argparse
import contextlib
import importlib.metadata as importlib_metadata
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import PandasTools
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
DOCKM8_PATH = scripts_path.parent
sys.path.append(str(DOCKM8_PATH))

from scripts.docking.docking import dockm8_docking  # noqa: E402
from scripts.library_preparation.library_preparation import prepare_library  # noqa: E402
from scripts.pocket_finding.pocket_finder import find_pocket  # noqa: E402
from scripts.protein_preparation.protein_preparation import prepare_protein  # noqa: E402
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS, rescore_poses  # noqa: E402
from scripts.utilities.logging import printlog  # noqa: E402

from analysis.redocking_benchmark import symmetric_rmsd  # noqa: E402

SOFTWARE = DOCKM8_PATH / "software"


def _install_receptor_pdbqt_fallback():
    """Make receptor PDB->PDBQT robust for QVINA. DockM8 hardcodes MGLTools
    `prepare_receptor4.py -A bond_hydrogens`, which crashes (PyBabel IndexError in
    add_vinyl_hydrogens) on a subset of receptors regardless of input. This wraps
    convert_molecules so the pdb->pdbqt path falls back to OpenBabel (`-xr`) when
    prepare_receptor4 fails. Applied as an in-memory monkeypatch from the analysis
    layer; DockM8 source files on disk are unchanged."""
    import scripts.docking.qvina2_docking as _q2
    import scripts.docking.qvinaw_docking as _qw
    import scripts.utilities.molecule_conversion as _mc

    _orig = _mc.convert_molecules

    def _robust(input_file, output_file_or_path, input_format, output_format, *a, **k):
        if input_format == "pdb" and output_format == "pdbqt":
            out = Path(output_file_or_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = f"conda run -n mgltools prepare_receptor4.py -r {input_file} -o {out} -A bond_hydrogens"
            r = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
                return output_file_or_path
            subprocess.run(["obabel", str(input_file), "-O", str(out), "-xr"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if out.exists() and out.stat().st_size > 0:
                return output_file_or_path
            raise RuntimeError(f"receptor->PDBQT failed (MGLTools + OpenBabel) for {input_file}")
        return _orig(input_file, output_file_or_path, input_format, output_format, *a, **k)

    for _mod in (_mc, _q2, _qw):
        _mod.convert_molecules = _robust


_install_receptor_pdbqt_fallback()

# The 17 pose-selection scoring functions (exclude GNINA-All; its 3 component
# columns GNINA-Affinity/CNN-Score/CNN-Affinity are already separate entries).
DEFAULT_SF = [k for k in RESCORING_FUNCTIONS if k != "GNINA-All"]

# Scoring functions with no known training overlap with PDBbind/CASF-2016 core
# (classical / knowledge-based). Only this group may anchor a "best SF recovers
# poses" generalisation claim (see plan R7).
LEAKAGE_FREE = {"AD4", "CHEMPLP", "PLP", "Vinardo", "LinF9", "KORP-PL", "ConvexPLR"}

# Steps that use the GPU (gated by the shared semaphore). GNINA CNN rescoring runs
# --no_gpu (CPU); SCORCH is XGBoost/CPU. RTMScore/GenScore auto-select CUDA.
GPU_ENGINES = {"GNINA_GPU"}
GPU_SF = {"RTMScore", "GenScore-scoring", "GenScore-docking", "GenScore-balanced"}

# Fixed seeds (documented in provenance). PLANTS seed added to its config;
# GypsumDL conformer embedding seeded in software/gypsum_dl.../MyMol.py.
ENGINE_SEEDS = {"SMINA": 1, "GNINA": 1, "GNINA_GPU": 1, "QVINA2": 1, "QVINAW": 1,
                "PLANTS": 1, "GypsumDL_embed": 42}

NATIVE_SCORE = {
    "SMINA": ("SMINA_Affinity", True),
    "GNINA": ("CNN-Score", False),
    "GNINA_GPU": ("CNN-Score", False),
    "PLANTS": ("CHEMPLP", True),
    "QVINA2": ("QVINA2_Affinity", True),
    "QVINAW": ("QVINAW_Affinity", True),
}
# Pose ID is "<id>_<program>_<rank>"; the program token written by each engine:
PROGRAM_TOKEN = {
    "SMINA": "smina", "GNINA": "gnina", "GNINA_GPU": "gnina",
    "PLANTS": "plants", "QVINA2": "qvina2", "QVINAW": "qvinaw",
}

_UNCHARGER = rdMolStandardize.Uncharger()

# Set in the parent before forking workers; inherited by each worker process.
_GPU_SEM = None   # multiprocessing.Semaphore | None
_ARGS = None      # argparse.Namespace (worker-side handle to the run config)


@contextlib.contextmanager
def gpu_guard(gated: bool = True):
    """Hold a GPU slot for the duration of a GPU step (no-op in serial mode)."""
    if gated and _GPU_SEM is not None:
        _GPU_SEM.acquire()
        try:
            yield
        finally:
            _GPU_SEM.release()
    else:
        yield


def _as_bool(value) -> bool:
    """Robust truthiness for values round-tripped through CSV."""
    return str(value).strip().lower() in ("true", "1", "1.0", "yes")


def clean_reference(cid: str, casf_dir: Path, cdir: Path) -> Path | None:
    """CASF X-TOOL .sdf files often fail RDKit sanitization (valence errors).
    Convert the crystal .mol2 to SDF via OpenBabel (tolerant) for a robust
    RMSD reference and library-prep input. Returns the clean SDF path."""
    mol2 = casf_dir / cid / f"{cid}_ligand.mol2"
    out = cdir / f"{cid}_ref.sdf"
    src = mol2 if mol2.exists() else (casf_dir / cid / f"{cid}_ligand.sdf")
    # Primary: OpenBabel (tolerant of CASF X-TOOL .sdf valence quirks).
    try:
        subprocess.run(["obabel", str(src), "-O", str(out)], check=True, capture_output=True, text=True)
    except Exception:  # noqa: BLE001
        out = None
    if out is not None and out.exists() and out.stat().st_size > 0:
        if next(Chem.SDMolSupplier(str(out), removeHs=False, sanitize=True), None) is not None:
            return out
    # Fallback: RDKit's own mol2 reader (handles amidines etc. OpenBabel
    # mis-perceives, e.g. 4mme). Rewrite a sanitisable SDF for downstream use.
    if mol2.exists():
        m = Chem.MolFromMol2File(str(mol2), removeHs=False, sanitize=True)
        if m is not None:
            out = cdir / f"{cid}_ref.sdf"
            with Chem.SDWriter(str(out)) as w:
                w.write(m)
            return out
    return None


def load_crystal(ref_sdf: Path):
    """Read the crystal ligand as the RMSD reference, assigning stereochemistry
    from the 3D coordinates so the redocked molecule can be verified against it."""
    mol = next(Chem.SDMolSupplier(str(ref_sdf), removeHs=False, sanitize=True), None)
    if mol is not None:
        try:
            Chem.AssignStereochemistryFrom3D(mol)
        except Exception:  # noqa: BLE001
            pass
    return mol


def graph_key(mol) -> str | None:
    """Heavy-atom connectivity key: atomic numbers + bond topology, ignoring bond
    orders, formal charges, aromaticity, stereochemistry and hydrogens. Two
    molecules share this key iff spyRMSD can match them (same atoms, isomorphic
    adjacency) - which is exactly the condition under which the pose RMSD is well
    defined. Tautomer / protonation differences (e.g. GypsumDL's zwitterionic
    azoles `[nH+][n-]`) thus do NOT spuriously flag a graph mismatch."""
    if mol is None:
        return None
    try:
        rw = Chem.RWMol(Chem.RemoveHs(mol))
        for b in rw.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        for a in rw.GetAtoms():
            a.SetFormalCharge(0)
            a.SetIsAromatic(False)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
        return Chem.MolToSmiles(rw.GetMol(), isomericSmiles=False, canonical=True)
    except Exception:  # noqa: BLE001
        return None


def rmsd_to_crystal(ref_mol, pred_mol) -> float:
    """Symmetry-corrected, no-superposition RMSD; NaN if the pose graph cannot be
    matched to the reference (different heavy-atom count, or isomorphism failure)."""
    if pred_mol is None or ref_mol is None:
        return np.nan
    if Chem.RemoveHs(pred_mol).GetNumAtoms() != Chem.RemoveHs(ref_mol).GetNumAtoms():
        return np.nan
    try:
        return symmetric_rmsd(ref_mol, pred_mol)
    except Exception:  # noqa: BLE001
        return np.nan


def _load_engine_poses(cdir: Path, engines) -> pd.DataFrame:
    """Concatenate each engine's poses SDF (read per-engine, not via all_poses.sdf,
    so split CPU/GPU docking calls don't clobber each other)."""
    frames = []
    for eng in engines:
        f = cdir / eng.lower() / f"{eng.lower()}_poses.sdf"
        if f.exists() and f.stat().st_size > 0:
            frames.append(PandasTools.LoadSDF(str(f), molColName="Molecule", idName="Pose ID"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _complex_complete(cid: str, w_dir: Path, engines) -> bool:
    """A complex counts as done if it has a '.processed' marker (fully run under
    the current code) OR every requested engine has a non-empty poses file.
    The marker prevents endlessly re-processing complexes whose engine genuinely
    fails (e.g. PLANTS on amidine ligands), while still letting a stale run from
    older code (no marker, empty QVINA) be recovered once."""
    if (w_dir / cid / ".processed").exists():
        return True
    for eng in engines:
        f = w_dir / cid / eng.lower() / f"{eng.lower()}_poses.sdf"
        if not f.exists() or f.stat().st_size == 0:
            return False
    return True


def process_complex(cid: str, casf_dir: Path, w_dir: Path, engines, methods, args) -> dict:
    protein = casf_dir / cid / f"{cid}_protein.pdb"
    row: dict = {"id": cid, "status": "ok", "graph_match": False}
    cdir = w_dir / cid
    cdir.mkdir(parents=True, exist_ok=True)

    ref_sdf = clean_reference(cid, casf_dir, cdir)
    ref_mol = load_crystal(ref_sdf) if ref_sdf else None
    if ref_mol is None:
        return {"id": cid, "status": "crystal_read_fail", "graph_match": False}
    ref_key = graph_key(ref_mol)

    try:
        prepared_protein = prepare_protein(protein, output_dir=cdir, protonation_method=args.protonation)
    except Exception as exc:  # noqa: BLE001
        # Protoss (web service) intermittently rejects/rate-limits some receptors;
        # fall back to the offline PDBFixer so the complex still completes.
        if str(args.protonation).lower() == "pdbfixer":
            raise
        tqdm.write(f"[{cid}] protein prep ({args.protonation}) failed ({type(exc).__name__}); falling back to pdbfixer")
        prepared_protein = prepare_protein(protein, output_dir=cdir, protonation_method="pdbfixer")
    pocket = find_pocket(mode="Reference", receptor=prepared_protein, ligand=ref_sdf, radius=args.radius)

    # Library prep from the crystal-derived isomeric SMILES (graph + stereo, no
    # coords) so the conformer is regenerated from scratch (honest redocking).
    smiles = Chem.MolToSmiles(Chem.RemoveHs(ref_mol))
    lig_df = pd.DataFrame({"SMILES": [smiles], "ID": [cid]})
    prepared_lib = cdir / "prepared_ligand.sdf"
    # Reuse the cached conformer if present, so a re-processed complex (e.g. to
    # recover an empty QVINA docking) docks the SAME conformer the other engines
    # already used, and resume is faster.
    if not (prepared_lib.exists() and prepared_lib.stat().st_size > 0):
        prepare_library(
            lig_df, protonation="GypsumDL", conformers="GypsumDL",
            software=SOFTWARE, n_cpus=args.n_cpus, output_sdf=prepared_lib,
        )
    prep_mol = next(Chem.SDMolSupplier(str(prepared_lib), removeHs=False, sanitize=True), None) if prepared_lib.exists() else None
    row["graph_match"] = bool(prep_mol is not None and ref_key is not None and graph_key(prep_mol) == ref_key)

    # Re-dock any engine whose cached poses file is empty (0 bytes), so resume
    # actually retries failed dockings (DockM8 skips on file existence alone).
    for eng in engines:
        f = cdir / eng.lower() / f"{eng.lower()}_poses.sdf"
        if f.exists() and f.stat().st_size == 0:
            f.unlink()

    # Docking: GPU engines hold a GPU slot; all engines use the protonated
    # receptor. QVINA receptor->PDBQT falls back to OpenBabel via the installed
    # convert_molecules patch when MGLTools crashes.
    for grp, gated in (([e for e in engines if e in GPU_ENGINES], True),
                       ([e for e in engines if e not in GPU_ENGINES], False)):
        if not grp:
            continue
        with gpu_guard(gated):
            dockm8_docking(
                library=prepared_lib, w_dir=cdir, protein_file=prepared_protein,
                pocket_definition=pocket, software=SOFTWARE, docking_programs=grp,
                exhaustiveness=args.exhaustiveness, n_poses=args.n_poses, n_cpus=args.n_cpus,
            )
    poses = _load_engine_poses(cdir, engines)
    if poses.empty:
        row["status"] = "docking_empty"
        return row
    row["total_poses"] = len(poses)

    poses["__rmsd"] = [rmsd_to_crystal(ref_mol, m) for m in poses["Molecule"]]
    poses["__token"] = poses["Pose ID"].str.split("_").str[1].str.lower()

    # Per-engine native top-1 (selected on score; RMSD as-is) and sampling ceiling.
    for eng in engines:
        sub = poses[poses["__token"] == PROGRAM_TOKEN[eng]]
        eng_rmsd = sub["__rmsd"].dropna()
        row[f"{eng}__n_poses"] = len(sub)
        row[f"{eng}__best_sampled_rmsd"] = float(eng_rmsd.min()) if len(eng_rmsd) else np.nan
        ncol, asc = NATIVE_SCORE[eng]
        val = np.nan
        if ncol in sub.columns and len(sub):
            s = sub.assign(_n=pd.to_numeric(sub[ncol], errors="coerce")).dropna(subset=["_n"]).sort_values("_n", ascending=asc)
            if len(s):
                val = float(s["__rmsd"].iloc[0])
        row[f"{eng}__native_top1_rmsd"] = val

    # Rescore once per group (CPU SFs ungated, GPU SFs gated), merge score columns.
    # A rescoring failure must not void the engine-native results above.
    try:
        merged = poses[["Pose ID", "__token", "__rmsd"]].copy()
        for grp, gated in ((  [m for m in methods if m not in GPU_SF], False),
                           (  [m for m in methods if m in GPU_SF], True)):
            if not grp:
                continue
            with gpu_guard(gated):
                scored = rescore_poses(
                    protein_file=prepared_protein, pocket_definition=pocket, software=SOFTWARE,
                    poses=poses[["Pose ID", "ID", "Molecule"]].copy(), functions=grp, n_cpus=args.n_cpus,
                )
            cols = [RESCORING_FUNCTIONS[m]["column_name"] for m in grp
                    if RESCORING_FUNCTIONS[m]["column_name"] in scored.columns]
            merged = merged.merge(scored[["Pose ID", *cols]], on="Pose ID", how="left")

        for eng in engines:
            sub = merged[merged["__token"] == PROGRAM_TOKEN[eng]]
            for m in methods:
                col = RESCORING_FUNCTIONS[m]["column_name"]
                cell = np.nan
                if col in sub.columns and len(sub):
                    s = sub.assign(_v=pd.to_numeric(sub[col], errors="coerce")).dropna(subset=["_v"])
                    if len(s):
                        idx = s["_v"].idxmin() if RESCORING_FUNCTIONS[m]["best_value"] == "min" else s["_v"].idxmax()
                        cell = float(s.loc[idx, "__rmsd"])
                row[f"{eng}__{m}"] = cell
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"[{cid}] rescoring/selection failed ({type(exc).__name__}: {exc}); native results kept")
        row["status"] = "ok_no_selection"
    # Mark fully processed under current code so resume won't re-run it for an
    # engine that genuinely fails (e.g. PLANTS on amidine ligands).
    (cdir / ".processed").touch()
    return row


def _kill_complex_procs(cdir: Path, sweeps: int = 6, pause: float = 0.4):
    """Kill every process whose command line references this complex's working
    directory (the docked-pose / pocket paths). This reliably reaps a hung
    scorer's whole tree - including DataLoader workers that reparent to init or
    are detached into their own group by `conda run` - because they all inherit
    the same argv. The worker's own cmdline references --w_dir, not <w_dir>/<cid>,
    so it is never matched. Swept repeatedly to catch stragglers that the scorer
    spawns mid-kill, until a sweep finds nothing."""
    try:
        import time

        import psutil
    except Exception:  # noqa: BLE001
        return
    needle = str(cdir)
    me = os.getpid()
    for _ in range(sweeps):
        killed = 0
        for p in psutil.process_iter(["pid", "cmdline"]):
            if p.info["pid"] == me:
                continue
            try:
                if needle in " ".join(p.info["cmdline"] or []):
                    p.kill()
                    killed += 1
            except Exception:  # noqa: BLE001
                pass
        if killed == 0:
            break
        time.sleep(pause)


class _ComplexTimeout(BaseException):
    """Raised on the per-complex alarm. A BaseException (not Exception) so inner
    try/except Exception blocks (e.g. the rescoring guard) cannot swallow it -
    it must propagate to _run_one for subprocess cleanup."""


def _timeout_handler(signum, frame):
    raise _ComplexTimeout("complex exceeded time budget")


def _run_one(cid: str) -> dict:
    """Worker entry point. Reads the inherited run config / GPU semaphore globals.
    A per-complex wall-clock alarm guards against a hung external tool (e.g.
    RTMScore/GenScore deadlocking): on timeout the subprocess tree is killed and
    the GPU slot is released (via gpu_guard's finally), so the run continues."""
    budget = getattr(_ARGS, "complex_timeout", 0) or 0
    if budget > 0:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(budget)
    try:
        return process_complex(cid, _ARGS.casf_dir, _ARGS.w_dir, _ARGS.engines, _ARGS.selection_methods, _ARGS)
    except _ComplexTimeout:
        tqdm.write(f"[{cid}] TIMEOUT after {budget}s; killing subprocesses")
        _kill_complex_procs(_ARGS.w_dir / cid)
        return {"id": cid, "status": "timeout", "graph_match": False}
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"[{cid}] FAILED: {type(exc).__name__}: {exc}")
        _kill_complex_procs(_ARGS.w_dir / cid)
        return {"id": cid, "status": f"fail:{type(exc).__name__}", "graph_match": False}
    finally:
        if budget > 0:
            signal.alarm(0)


def build_long(df: pd.DataFrame, engines, methods) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        if not str(r.get("status", "")).startswith("ok"):
            continue
        gm = _as_bool(r.get("graph_match", False))
        for eng in engines:
            for m in methods:
                col = f"{eng}__{m}"
                if col in r:
                    rows.append({"id": r["id"], "engine": eng, "sf": m,
                                 "selected_rmsd": pd.to_numeric(r[col], errors="coerce"),
                                 "graph_match": gm, "leakage_free": m in LEAKAGE_FREE})
    return pd.DataFrame(rows)


def collect_provenance(args, ids, counts) -> dict:
    def _ver(pkg):
        try:
            return importlib_metadata.version(pkg)
        except Exception:  # noqa: BLE001
            return None

    try:
        git_sha = subprocess.run(["git", "-C", str(DOCKM8_PATH), "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        git_sha = None
    try:
        obabel_v = subprocess.run(["obabel", "-V"], capture_output=True, text=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        obabel_v = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_complexes_requested": len(ids),
        "counts": counts,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "engine_seeds": ENGINE_SEEDS,
        "gpu_gated": {"engines": sorted(GPU_ENGINES), "scoring_functions": sorted(GPU_SF)},
        "leakage_free_sfs": sorted(LEAKAGE_FREE),
        "git_sha": git_sha,
        "versions": {
            "rdkit": rdBase.rdkitVersion,
            "spyrmsd": _ver("spyrmsd"),
            "numpy": _ver("numpy"),
            "pandas": _ver("pandas"),
            "posebusters": _ver("posebusters"),
            "openbabel": obabel_v,
            "tqdm": _ver("tqdm"),
        },
    }


def write_summary(df: pd.DataFrame, args) -> dict:
    status = df.get("status", pd.Series(dtype=str)).astype(str)
    gm = df.get("graph_match", pd.Series([False] * len(df))).apply(_as_bool)
    counts = {
        "total": int(len(df)),
        "ok": int(status.str.startswith("ok").sum()),
        "ok_graph_match": int((status.str.startswith("ok") & gm).sum()),
        "graph_mismatch": int((status.str.startswith("ok") & ~gm).sum()),
        "ok_no_selection": int((status == "ok_no_selection").sum()),
        "crystal_read_fail": int((status == "crystal_read_fail").sum()),
        "docking_empty": int((status == "docking_empty").sum()),
        "timeout": int((status == "timeout").sum()),
        "failed": int(status.str.startswith("fail").sum()),
    }

    long_df = build_long(df, args.engines, args.selection_methods)
    if len(long_df):
        long_df.to_csv(args.output.with_name(args.output.stem + "_long.csv"), index=False)
        ev = long_df[long_df["graph_match"] & long_df["selected_rmsd"].notna()]
        if len(ev):
            succ = (ev.assign(hit=ev["selected_rmsd"] <= 2.0)
                    .groupby(["engine", "sf"])["hit"].mean().mul(100).round(1).unstack("sf"))
            cnt = ev.groupby(["engine", "sf"])["selected_rmsd"].size().unstack("sf")
            succ.to_csv(args.output.with_name(args.output.stem + "_matrix_success2A.csv"))
            cnt.to_csv(args.output.with_name(args.output.stem + "_matrix_counts.csv"))
            lf = sorted(c for c in succ.columns if c in LEAKAGE_FREE)
            printlog("\n=== Engine x SF: % selected pose <= 2 A (evaluable complexes) ===")
            printlog("[leakage-free SFs only]\n" + succ[lf].to_string())
            printlog("\n[all SFs]\n" + succ.to_string())

    printlog("\n=== Counts ===\n" + "\n".join(f"  {k}: {v}" for k, v in counts.items()))
    okgm = df[status.str.startswith("ok") & gm]
    for eng in args.engines:
        vals = pd.to_numeric(okgm.get(f"{eng}__native_top1_rmsd", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(vals):
            printlog(f"  {eng}: native top-1 <=2A = {100 * (vals <= 2).mean():.0f}%  "
                     f"(median {vals.median():.2f} A, n={len(vals)})")
    return counts


def main():
    global _GPU_SEM, _ARGS
    p = argparse.ArgumentParser(description="CASF-2016 DockM8 re-docking RMSD benchmark")
    p.add_argument("--casf_dir", type=Path, default=Path("/home/tony/Datasets/CASF-2016/coreset"))
    p.add_argument("--ids", nargs="+", default=None, help="Specific complex IDs (default: all)")
    p.add_argument("--engines", nargs="+", default=["SMINA", "GNINA_GPU", "PLANTS", "QVINA2", "QVINAW"])
    p.add_argument("--selection_methods", nargs="+", default=DEFAULT_SF,
                   help="Pose-selection scoring functions (default: all 17)")
    p.add_argument("--protonation", default="protoss", help="protoss | pdbfixer | <pH float>")
    p.add_argument("--n_poses", type=int, default=10)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--radius", type=int, default=10)
    p.add_argument("--n_cpus", type=int, default=1, help="CPUs PER complex (intra-complex). Keep 1 with many --workers.")
    p.add_argument("--workers", type=int, default=30, help="Complexes processed in parallel.")
    p.add_argument("--gpu_slots", type=int, default=4, help="Max complexes using the GPU concurrently.")
    p.add_argument("--complex_timeout", type=int, default=1800,
                   help="Per-complex wall-clock budget (s); a hung complex is killed and marked 'timeout'. 0 disables.")
    p.add_argument("--w_dir", type=Path, default=Path.home() / "dockm8_casf_redocking")
    p.add_argument("--output", type=Path, default=DOCKM8_PATH / "analysis" / "data" / "casf_redocking_results.csv")
    args = p.parse_args()

    tokens = [PROGRAM_TOKEN[e] for e in args.engines]
    if len(set(tokens)) != len(tokens):
        sys.exit(f"ERROR: engines {args.engines} map to colliding Pose ID tokens {tokens}; choose one GNINA variant.")
    unknown = [m for m in args.selection_methods if m not in RESCORING_FUNCTIONS]
    if unknown:
        sys.exit(f"ERROR: unknown selection methods: {unknown}")

    ids = args.ids or sorted([d.name for d in args.casf_dir.iterdir() if d.is_dir()])
    args.w_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    done: set[str] = set()
    if args.output.exists():
        prev = pd.read_csv(args.output).drop_duplicates("id", keep="last")
        rows = prev.to_dict("records")
        ok_ids = set(prev.loc[prev["status"].astype(str).str.startswith("ok"), "id"].astype(str))
        # 'done' = ok AND every engine produced poses; the rest is re-processed
        # (recovers empty/missing dockings such as the QVINA failures).
        done = {c for c in ok_ids if _complex_complete(c, args.w_dir, args.engines)}
    todo = [c for c in ids if c not in done]
    printlog(f"CASF re-docking: total={len(ids)} done={len(done)} todo={len(todo)} | "
             f"workers={args.workers} gpu_slots={args.gpu_slots} n_cpus={args.n_cpus} | "
             f"engines={args.engines} | {len(args.selection_methods)} SFs")

    n_ok = sum(1 for r in rows if str(r.get("status", "")).startswith("ok"))
    n_fail = len(rows) - n_ok

    def _record(row):
        nonlocal n_ok, n_fail
        rows.append(row)
        if str(row.get("status", "")).startswith("ok"):
            n_ok += 1
        else:
            n_fail += 1
        pd.DataFrame(rows).drop_duplicates("id", keep="last").to_csv(args.output, index=False)  # single-writer checkpoint

    # Set inherited globals BEFORE forking workers (avoids pickling the semaphore).
    _ARGS = args
    if args.workers > 1 and len(todo) > 1:
        ctx = mp.get_context("fork")
        _GPU_SEM = ctx.Semaphore(max(1, args.gpu_slots))
        loop = tqdm(total=len(todo), desc="CASF redocking", unit="cplx")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futures = {ex.submit(_run_one, cid): cid for cid in todo}
            for fut in as_completed(futures):
                _record(fut.result())
                loop.update(1)
                loop.set_postfix(ok=n_ok, fail=n_fail)
        loop.close()
    else:
        _GPU_SEM = None  # serial: GPU guard is a no-op
        for cid in tqdm(todo, desc="CASF redocking", unit="cplx"):
            _record(_run_one(cid))

    df = pd.DataFrame(rows).drop_duplicates("id", keep="last")
    counts = write_summary(df, args)
    with open(args.output.with_suffix(".run.json"), "w") as fh:
        json.dump(collect_provenance(args, ids, counts), fh, indent=2)


if __name__ == "__main__":
    main()
