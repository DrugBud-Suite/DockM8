import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple

import MDAnalysis as mda
import pandas as pd
from openmm import app
from rdkit import Chem
from rdkit.Chem import ChemicalForceFields, PandasTools, rdmolops
from rdkit.Chem.rdchem import AtomPDBResidueInfo
from rdkit.Geometry import Point3D

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.parallel_executor import parallel_executor

# SMARTS patterns for identifying polar hydrogen donors and unwanted hydrogens
HDONER = "[$([O,S;+0]),$([N;$(Na),$(NC=[O,S]);H2]),$([N;$(N[S,P]=O)]);!H0]"
UNWANTED_H = "[#1;$([#1][N;+1;H2]),$([#1][N;!H2]a)]"


def convert_to_pdb(sdf: str) -> str:
	"""
    Convert an SDF file to a PDB file using Open Babel.

    Parameters:
    sdf (str): Path to the input SDF file.

    Returns:
    str: Path to the temporary PDB file.
    """
	with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as temp_pdb_file:
		temp_pdb_file_path = temp_pdb_file.name
	# Use Open Babel to convert the SDF file to PDB format
	cmd = f"obabel {sdf} -O {temp_pdb_file_path}"
	subprocess.call(cmd, shell=True)
	return temp_pdb_file_path


def constrain_minimize(mol: Chem.Mol, constrain_list: List[int]) -> Chem.Mol:
	"""
    Minimize a molecule with positional constraints on specified atoms.

    Parameters:
    mol (rdkit.Chem.Mol): The molecule to be minimized.
    constrain_list (list): List of atom indices to constrain during minimization.

    Returns:
    rdkit.Chem.Mol: The minimized molecule.
    """
	# Set up the force field for minimization
	ff_property = ChemicalForceFields.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
	ff = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ff_property, confId=0, ignoreInterfragInteractions=False)

	# Add positional constraints to specified atoms
	for query_atom_idx in constrain_list:
		ff.MMFFAddPositionConstraint(query_atom_idx, 0.0, 1000)

	ff.Initialize()

	max_minimize_iteration = 10
	# Perform energy minimization
	for _ in range(max_minimize_iteration):
		minimize_seed = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
		if minimize_seed == 0:
			break

	return mol


def preprocess_protein(pdb_path: str) -> Tuple[mda.Universe, Callable]:
	"""
    Preprocess the protein PDB file to obtain a MDAnalysis Universe and RDKit converter.

    Parameters:
    pdb_path (str): Path to the input PDB file.

    Returns:
    tuple: MDAnalysis Universe object and RDKit converter function.
    """
	pdbfile = app.PDBFile(str(pdb_path))
	protein_universe = mda.Universe(pdbfile)
	mda_to_rdkit = mda._CONVERTERS['RDKIT']().convert
	return protein_universe, mda_to_rdkit


def minimize_polar_hydrogens(ligand: Chem.Mol, protein_universe: mda.Universe, mda_to_rdkit: Callable) -> Chem.Mol:
	"""
    Minimize the polar hydrogens of a ligand in the context of a protein pocket.

    Parameters:
    ligand (rdkit.Chem.Mol): The ligand molecule.
    protein_universe (MDAnalysis.Universe): The preprocessed protein universe.
    mda_to_rdkit (function): MDAnalysis to RDKit converter function.

    Returns:
    rdkit.Chem.Mol: The minimized ligand molecule.
    """
	# Check if the ligand has polar hydrogens to minimize
	ligand_pattern = Chem.MolFromSmarts(HDONER)
	match = ligand.HasSubstructMatch(ligand_pattern)

	if not match:
		return ligand

	# Create a temporary SDF file for the ligand
	with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as temp_sdf_file:
		temp_sdf_file_path = temp_sdf_file.name
		writer = Chem.SDWriter(temp_sdf_file_path)
		writer.write(ligand)
		writer.close()

	# Convert the temporary SDF file to a PDB file
	temp_pdb_path = convert_to_pdb(temp_sdf_file_path)
	ligand_universe = mda.Universe(temp_pdb_path)

	# Merge the protein and ligand into a single MDAnalysis universe
	merge_pdb = mda.Merge(protein_universe.atoms, ligand_universe.atoms)
	pro_pocket = merge_pdb.select_atoms('byres protein and around 4.0 (resname UNL)')
	pro_pocket_mol = mda_to_rdkit(pro_pocket)

	# Set residue names for the protein pocket and ligand atoms
	for atom in pro_pocket_mol.GetAtoms():
		atom.GetMonomerInfo().SetResidueName("PRO")

	for atom in ligand.GetAtoms():
		monomer_info = atom.GetMonomerInfo()
		if monomer_info is None:
			residue_info = AtomPDBResidueInfo()
			residue_info.SetResidueName("LIG")
			atom.SetMonomerInfo(residue_info)
		else:
			monomer_info.SetResidueName("LIG")

	# Combine the ligand and protein pocket into a single molecule
	complex = rdmolops.CombineMols(ligand, pro_pocket_mol)
	constrain_list = []
	unwanted_H_pattern = Chem.MolFromSmarts(UNWANTED_H)
	unwanted_H_atom_idx_list = list(complex.GetSubstructMatches(unwanted_H_pattern))
	unwanted_H_atom_idx_list = [unwanted_H_atom_idx_tuple[0] for unwanted_H_atom_idx_tuple in unwanted_H_atom_idx_list]

	# Create a list of atom indices to constrain during minimization
	for atom in complex.GetAtoms():
		if atom.GetMonomerInfo().GetResidueName() == "PRO":
			constrain_list.append(atom.GetIdx())
		if atom.GetMonomerInfo().GetResidueName() == "LIG":
			if atom.GetSymbol() != "H":
				constrain_list.append(atom.GetIdx())
			if atom.GetIdx() in unwanted_H_atom_idx_list:
				constrain_list.append(atom.GetIdx())

	# Minimize the complex with constraints
	complex_min = constrain_minimize(complex, constrain_list)
	coord_dict = {}
	for atom in complex_min.GetAtoms():
		if atom.GetMonomerInfo().GetResidueName() == "LIG":
			coord_dict[atom.GetIdx()] = complex_min.GetConformer().GetAtomPosition(atom.GetIdx())

	# Assign minimized coordinates back to the ligand
	lig_conf = Chem.Conformer()
	for idx in range(len(ligand.GetAtoms())):
		atom_coords = coord_dict[idx]
		atom_coords_point_3D = Point3D(atom_coords[0], atom_coords[1], atom_coords[2])
		lig_conf.SetAtomPosition(idx, atom_coords_point_3D)

	ligand.RemoveAllConformers()
	ligand.AddConformer(lig_conf)

	# Clean up temporary files
	os.remove(temp_sdf_file_path)
	os.remove(temp_pdb_path)

	return ligand


def minimize_all_ligands(pdb_path: str, ligfile_path: str, n_cpus: int = int(os.cpu_count() * 0.9)) -> pd.DataFrame:
	"""
    Minimize all ligands in an SDF file in the context of a protein pocket.

    Parameters:
    pdb_path (str): Path to the input protein PDB file.
    ligfile_path (str): Path to the input SDF file containing ligands.
    outfile_path (str): Path to the output SDF file for minimized ligands.
    n_cpus (int): Number of CPUs to use for parallel processing.
    """
	# Read the ligands from the SDF file
	dataframe = PandasTools.LoadSDF(ligfile_path, molColName='Molecule', idName='Pose ID')

	ligands = dataframe['Molecule']

	minimized_ligands = []

	# Preprocess the protein once
	protein_universe, mda_to_rdkit = preprocess_protein(pdb_path)

	# Parallelize ligand minimization using the provided parallel_executor function
	minimized_ligands = parallel_executor(minimize_polar_hydrogens,
				list(ligands),
				n_cpus,
				job_manager="concurrent_process",
				protein_universe=protein_universe,
				mda_to_rdkit=mda_to_rdkit)

	# Add the minimized ligands to the dataframe

	dataframe['Molecule'] = minimized_ligands

	return dataframe
