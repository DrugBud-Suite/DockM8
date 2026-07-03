import sys
import warnings
from pathlib import Path

import numpy as np
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog
from scripts.utilities.utilities import load_molecule

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


def extract_pocket(pocket_definition, protein_file: Path, ligand: Path | None = None, cutoff: float = 10.0):
    """Extract the binding pocket as WHOLE residues.

    A residue is kept in full if ANY of its atoms lies within ``cutoff`` Å of the
    reference: the reference ligand's atoms when a ``ligand`` is given (Reference/RoG
    modes), otherwise the docking-box centre (Manual/Dogsitescorer modes).

    This replaces the previous per-atom box slicing, which kept only the near atoms of
    each residue and produced *truncated* residues. RTMScore and GenScore build a graph
    from the pocket residues, so truncated residues corrupted their scores; classical
    scoring functions use the full receptor and were unaffected. The docking box
    (``pocket_definition`` center/size) is not used for docking here and is left
    untouched.

    The pocket PDB is always (re)generated, so a stale box pocket from a prior run can
    never be reused and mask the correction.

    Args:
        pocket_definition (dict): docking-box ``center``/``size`` (only ``center``
            (and ``size`` for the box radius) is used, and only when ``ligand`` is None).
        protein_file (Path): receptor PDB.
        ligand (Path | None): reference ligand file (Reference/RoG). When None, residues
            are taken around the box centre instead.
        cutoff (float): distance in Å around the reference ligand atoms. Ignored for the
            ligand-less case, which uses the box half-size.

    Returns:
        Path | None: the ``<protein>_pocket.pdb`` path, or None if the pocket is empty.
    """
    protein_file = Path(protein_file)
    output_file = protein_file.with_name(protein_file.stem + "_pocket.pdb")

    ppdb = PandasPdb().read_pdb(str(protein_file))
    atoms = ppdb.df["ATOM"]
    protein_xyz = atoms[["x_coord", "y_coord", "z_coord"]].to_numpy()

    if ligand is not None:
        printlog(f"Extracting whole-residue pocket from {protein_file.stem} within {cutoff} Å of {Path(ligand).stem}")
        reference_xyz = load_molecule(str(ligand)).GetConformer().GetPositions()
        radius = cutoff
    else:
        radius = float(pocket_definition["size"][0]) / 2.0
        printlog(f"Extracting whole-residue pocket from {protein_file.stem} within {radius} Å of the box centre")
        reference_xyz = np.asarray([pocket_definition["center"]], dtype=float)

    min_distance = cdist(protein_xyz, reference_xyz).min(axis=1)
    residue_key = ["chain_id", "residue_number", "insertion"]
    near_residues = atoms.loc[min_distance < radius, residue_key].drop_duplicates()

    if near_residues.empty:
        printlog(f"Warning: no residues within {radius} Å for {protein_file.stem}. The pocket might be empty.")
        return None

    pocket_atoms = atoms.merge(near_residues, on=residue_key)
    ppdb.df["ATOM"] = pocket_atoms
    ppdb.to_pdb(path=str(output_file), records=["ATOM"])
    printlog(f"Finished extracting pocket from {protein_file.stem} ({len(near_residues)} residues)")
    return output_file
