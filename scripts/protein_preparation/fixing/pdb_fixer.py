import sys
from pathlib import Path

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from openmm.app import PDBFile
from pdbfixer import PDBFixer

from scripts.utilities.utilities import printlog


def fix_pdb_file(input_pdb_file: Path,
		output_dir: Path = None,
		fix_nonstandard_residues: bool = True,
		fix_missing_residues: bool = True,
		add_missing_hydrogens_pH: float = 7.0,
		remove_hetero: bool = True,
		remove_water: bool = True,
	):
	"""
    Fixes a PDB file by performing various modifications.

    Args:
        input_pdb_file (Path): The path to the input PDB file.
        output_dir (Path, optional): The directory where the fixed PDB file will be saved. If not provided, the same directory as the input file will be used.
        fix_nonstandard_residues (bool, optional): Whether to fix nonstandard residues. Defaults to True.
        fix_missing_residues (bool, optional): Whether to fix missing residues. Defaults to True.
        add_missing_hydrogens_pH (float, optional): The pH value for adding missing hydrogens. Defaults to 7.0.
        remove_hetero (bool, optional): Whether to remove heteroatoms. Defaults to True.
        remove_water (bool, optional): Whether to remove water molecules. Defaults to True.

    Returns:
        Path: The path to the fixed PDB file.
    """
	output_dir = output_dir or input_pdb_file.parent

	printlog(f"Fixing PDB file: {input_pdb_file}")

	fixer = PDBFixer(filename=str(input_pdb_file))
	if fix_nonstandard_residues:
		printlog("Fixing nonstandard residues")
		fixer.findNonstandardResidues()
		fixer.replaceNonstandardResidues()
	if fix_missing_residues:
		printlog("Fixing missing residues")
		fixer.findMissingResidues()
		fixer.findMissingAtoms()
		fixer.addMissingAtoms()
	if add_missing_hydrogens_pH is not None or add_missing_hydrogens_pH == 0.0:
		printlog(f"Adding missing hydrogens at pH {add_missing_hydrogens_pH}")
		fixer.addMissingHydrogens(add_missing_hydrogens_pH)
	if remove_hetero and remove_water:
		printlog("Removing heteroatoms and water molecules")
		fixer.removeHeterogens(keepWater=False)
	if remove_hetero and not remove_water:
		printlog("Removing heteroatoms and leaving water")
		fixer.removeHeterogens(keepWater=True)
	printlog("Writing fixed PDB file")
	output_file = output_dir / (input_pdb_file.stem + "_fixed.pdb")
	PDBFile.writeFile(fixer.topology, fixer.positions, open(output_file, "w"))
	return output_file
