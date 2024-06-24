import sys
import warnings
from pathlib import Path
import pandas as pd
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.library_preparation.conformer_generation.confgen_GypsumDL import generate_conformers_GypsumDL
from scripts.library_preparation.conformer_generation.confgen_RDKit import generate_conformers_RDKit
from scripts.library_preparation.protonation.protgen_GypsumDL import protonate_GypsumDL
from scripts.library_preparation.standardisation.standardise import standardize_library
from scripts.utilities.utilities import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROTONATION_OPTIONS = ["GypsumDL", "None"]
CONFORMER_OPTIONS = ["RDKit", "MMFF", "GypsumDL"]


def prepare_library(input_sdf: str,
					output_dir: Path,
					id_column: str,
					protonation: str,
					conformers: str,
					software: Path,
					n_cpus: int,
					n_conformers: int = 1,
					):
	"""
    Prepares a docking library for further analysis.

    Args:
        input_sdf (str): The path to the input SDF file containing the docking library.
        output_dir (Path): The directory to save output files.
        id_column (str): The name of the column in the SDF file that contains the compound IDs.
        protonation (str): The method to use for protonation. Can be 'GypsumDL', or 'None' for no protonation.
        conformers (str): The method to use for conformer generation. Can be 'RDKit', 'MMFF', or 'GypsumDL'.
        software (Path): The path to the GypsumDL software.
        n_cpus (int): The number of CPUs to use for parallelization.
        n_conformers (int): The number of conformers to generate (default is 1).
    """
	# Load input SDF file into a DataFrame
	input_df = PandasTools.LoadSDF(input_sdf, molColName="Molecule", idName=id_column)

	# Standardization
	standardized_df = standardize_library(input_df, id_column=id_column, smiles_column=None, n_cpus=n_cpus)

	# Protonation
	if protonation == "GypsumDL":
		protonated_df = protonate_GypsumDL(standardized_df, software, n_cpus)
	elif protonation == "None":
		protonated_df = standardized_df
	else:
		raise ValueError(f'Invalid protonation method specified: {protonation}. Must be either "None" or "GypsumDL".')

	# Conformer generation
	if conformers == "MMFF":
		final_df = generate_conformers_RDKit(protonated_df, n_cpus, forcefield='MMFF')
	elif conformers == "UFF":
		final_df = generate_conformers_RDKit(protonated_df, n_cpus, forcefield='MMFF')
	elif conformers == "GypsumDL":
		final_df = generate_conformers_GypsumDL(protonated_df, software, n_cpus)
	else:
		raise ValueError(
			f'Invalid conformer method specified: {conformers}. Must be either "RDKit", "MMFF" or "GypsumDL".')

	printlog("Preparing final output...")

	# Keep only 'Molecule' and 'ID' columns
	final_df = final_df[["Molecule", "ID"]]

	# Save the final DataFrame as an SDF file
	final_sdf_path = output_dir / "final_library.sdf"
	PandasTools.WriteSDF(final_df, str(final_sdf_path), molColName="Molecule", idName="ID")

	printlog("Preparation of compound library finished.")
	return final_sdf_path
