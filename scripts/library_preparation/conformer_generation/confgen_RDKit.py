import sys
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def conf_gen_RDKit(molecule: Chem.Mol, forcefield: str):
	"""
    Generates 3D conformers using RDKit.

    Args:
        molecule (RDKit molecule): The input molecule.
        forcefield (str): The force field to be used for optimization. Valid options are 'MMFF' and 'UFF'.

    Returns:
        molecule (RDKit molecule): The molecule with 3D conformers.
    """
	try:
		if not molecule.GetConformer().Is3D():
			molecule = Chem.AddHs(molecule)               # Add hydrogens to the molecule
			AllChem.EmbedMolecule(molecule, AllChem.ETKDGv3())
			if forcefield == 'MMFF':                      # Generate initial 3D coordinates for the molecule
				AllChem.MMFFOptimizeMolecule(molecule)
			elif forcefield == 'UFF':
				AllChem.UFFOptimizeMolecule(molecule)        # Optimize the 3D coordinates using the MMFF force field
			AllChem.SanitizeMol(molecule)                 # Sanitize the molecule to ensure it is chemically valid
		return molecule
	except Exception as e:
		printlog(f"Error generating conformer: {str(e)}")
		return None


def generate_conformers_RDKit(df: pd.DataFrame, n_cpus: int, forcefield: str) -> pd.DataFrame:
	"""
    Generates 3D conformers using RDKit.

    Args:
        df (pd.DataFrame): The input DataFrame containing the molecules.
        n_cpus (int): Number of CPUs to use for parallel processing.
        forcefield (str): The forcefield to use for conformer generation.

    Returns:
        pd.DataFrame: The DataFrame with generated conformers.
    """
	printlog("Generating 3D conformers using RDKit...")

	try:
		n_cpds_start = len(df)
		# Generate conformers for each molecule in parallel using the conf_gen_RDKit function
		results = parallel_executor(conf_gen_RDKit,
				df['Molecule'].tolist(),
				n_cpus,
				'concurrent_process',
				forcefield=forcefield)

		df['Molecule'] = results
		# Remove molecules where conformer generation failed
		df = df.dropna(subset=['Molecule'])
		n_cpds_end = len(df)

		# Check if the number of compounds matches the input
		if n_cpds_start != n_cpds_end:
			printlog(
				f"Conformer generation failed for {n_cpds_start - n_cpds_end} compounds. These compounds have been removed from the library."
			)
	except Exception as e:
		printlog(f"ERROR: Failed to generate conformers using RDKit! {str(e)}")

	return df
