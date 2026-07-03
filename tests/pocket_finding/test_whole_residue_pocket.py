# Regression: extract_pocket must keep WHOLE residues. A residue with any atom within
# `cutoff` of the reference ligand is included in full -- including atoms that are
# themselves beyond the cutoff. The old per-atom slicing truncated residues, which
# corrupted RTMScore/GenScore (they build a graph from the pocket residues).
from pathlib import Path

from biopandas.pdb import PandasPdb
from rdkit import Chem

from scripts.pocket_finding.utils import extract_pocket


def _atom(serial, name, resname, chain, resseq, x, y, z, elem):
    nm = f"{name:<4}" if len(name) >= 4 else " " + f"{name:<3}"
    return (
        "ATOM  " + f"{serial:>5}" + " " + nm + " " + f"{resname:>3}" + " "
        + chain + f"{resseq:>4}" + " " + "   "
        + f"{x:>8.3f}" + f"{y:>8.3f}" + f"{z:>8.3f}"
        + f"{1.0:>6.2f}" + f"{0.0:>6.2f}" + " " * 10 + f"{elem:>2}"
    )


def _write_protein(path: Path):
    # Residue A (resseq 1): N at x=9 (within 10 A of origin), CA at 9.5, C at 15 (BEYOND
    #   the 10 A cutoff). The residue has an in-range atom, so it must be kept WHOLE --
    #   including C. Residue B (resseq 2): all atoms at x>=30, entirely excluded.
    lines = [
        _atom(1, "N", "ALA", "A", 1, 9.0, 0.0, 0.0, "N"),
        _atom(2, "CA", "ALA", "A", 1, 9.5, 0.0, 0.0, "C"),
        _atom(3, "C", "ALA", "A", 1, 15.0, 0.0, 0.0, "C"),
        _atom(4, "N", "GLY", "A", 2, 30.0, 0.0, 0.0, "N"),
        _atom(5, "CA", "GLY", "A", 2, 31.0, 0.0, 0.0, "C"),
        _atom(6, "C", "GLY", "A", 2, 32.0, 0.0, 0.0, "C"),
    ]
    path.write_text("\n".join(lines) + "\nEND\n")


def _write_ligand(path: Path):
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom(6))
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))  # single atom at the origin
    rw.AddConformer(conf)
    Chem.MolToMolFile(rw.GetMol(), str(path))


def test_pocket_keeps_whole_residues(tmp_path):
    protein = tmp_path / "prot_protein.pdb"
    ligand = tmp_path / "ref.mol"
    _write_protein(protein)
    _write_ligand(ligand)
    # sanity: the input parses to 6 atoms across 2 residues
    assert len(PandasPdb().read_pdb(str(protein)).df["ATOM"]) == 6

    out = extract_pocket({"center": [0.0, 0.0, 0.0], "size": [20.0, 20.0, 20.0]}, protein, ligand=ligand, cutoff=10.0)
    assert out is not None and Path(out).exists()

    pocket = PandasPdb().read_pdb(str(out)).df["ATOM"]
    residues = set(zip(pocket["residue_name"], pocket["residue_number"]))
    assert ("ALA", 1) in residues, "residue with an in-range atom must be in the pocket"
    assert ("GLY", 2) not in residues, "residue entirely beyond cutoff must be excluded"

    # Whole residue: ALL 3 atoms of ALA/1 are present, including C at x=15 (beyond the
    # 10 A cutoff) -- the old per-atom slicing would have dropped it.
    ala_atoms = pocket[(pocket["residue_name"] == "ALA") & (pocket["residue_number"] == 1)]
    assert len(ala_atoms) == 3, "residue must be kept whole, not truncated to near atoms"
    assert 15.0 in set(ala_atoms["x_coord"].round(3)), "the far atom of an in-range residue must be retained"


def test_ligandless_uses_box_centre(tmp_path):
    # Manual/Dogsitescorer path: no ligand -> whole residues within box half-size of the
    # centre. Box size 20 -> radius 10; only residue A (min dist 9) qualifies.
    protein = tmp_path / "prot_protein.pdb"
    _write_protein(protein)
    out = extract_pocket({"center": [0.0, 0.0, 0.0], "size": [20.0, 20.0, 20.0]}, protein, ligand=None)
    pocket = PandasPdb().read_pdb(str(out)).df["ATOM"]
    residues = set(zip(pocket["residue_name"], pocket["residue_number"]))
    assert residues == {("ALA", 1)}
    assert len(pocket) == 3
