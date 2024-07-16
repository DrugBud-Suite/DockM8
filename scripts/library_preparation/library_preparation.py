import sys
import warnings
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.library_preparation.conformer_generation.confgen_GypsumDL import generate_conformers_GypsumDL
from scripts.library_preparation.conformer_generation.confgen_RDKit import generate_conformers_RDKit
from scripts.library_preparation.protonation.protgen_GypsumDL import protonate_GypsumDL
from scripts.library_preparation.standardisation.standardise import standardize_library
from scripts.utilities.logging import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROTONATION_OPTIONS = ["GypsumDL", "None"]
CONFORMER_OPTIONS = ["RDKit", "MMFF", "GypsumDL"]


def prepare_library(input_data: Union[pd.DataFrame, Path],
					protonation: str,
					conformers: str,
					software: Path,
					n_cpus: int,
					n_conformers: int = 1,
					standardize_ids: bool = True,
					standardize_tautomers: bool = True,
					remove_salts: bool = True,
					min_ph: float = 6.4,
					max_ph: float = 8.4,
					pka_precision: float = 1.0,
					output_sdf: Optional[Path] = None) -> pd.DataFrame:
	"""
    Prepares a docking library for further analysis.

    Args:
        input_data (Union[pd.DataFrame, Path]): The input data, either a DataFrame or path to an SDF file.
        id_column (str): The name of the column that contains the compound IDs.
        protonation (str): The method to use for protonation.
        conformers (str): The method to use for conformer generation.
        software (Path): The path to the required software.
        n_cpus (int): The number of CPUs to use for parallelization.
        n_conformers (int): The number of conformers to generate (default is 1).
        output_sdf (Union[Path, None]): Path to save the output SDF file (optional).

    Returns:
        pd.DataFrame: A DataFrame containing the prepared library.
    """
	# Load input data into a DataFrame
	if isinstance(input_data, pd.DataFrame):
		input_df = input_data
		if 'Molecule' not in input_df.columns:
			try:
				input_df['Molecule'] = input_df['SMILES'].apply(Chem.MolFromSmiles)
			except Exception as e:
				raise ValueError(f"Failed to find a column containing molecule data: {str(e)}")

	elif isinstance(input_data, Path):
		input_df = PandasTools.LoadSDF(str(input_data), molColName="Molecule", idName='ID')
	else:
		raise ValueError("input_data must be either a pandas DataFrame or a Path to an SDF file.")

	# Standardization
	standardized_df = standardize_library(input_df,
											smiles_column=None,
											remove_salts=remove_salts,
											standardize_tautomers=standardize_tautomers,
											standardize_ids_flag=standardize_ids,
											n_cpus=n_cpus)

	# Protonation
	if protonation == "GypsumDL":
		protonated_df = protonate_GypsumDL(standardized_df, software, n_cpus, min_ph, max_ph, pka_precision)
	elif protonation == "None":
		protonated_df = standardized_df
	else:
		raise ValueError(f'Invalid protonation method specified: {protonation}. Must be either "None" or "GypsumDL".')

	# Conformer generation
	if conformers == "MMFF":
		final_df = generate_conformers_RDKit(protonated_df, n_cpus, forcefield='MMFF')
	elif conformers == "UFF":
		final_df = generate_conformers_RDKit(protonated_df, n_cpus, forcefield='UFF')
	elif conformers == "GypsumDL":
		final_df = generate_conformers_GypsumDL(protonated_df, software, n_cpus)
	else:
		raise ValueError(
			f'Invalid conformer method specified: {conformers}. Must be either "MMFF", "UFF" or "GypsumDL".')

	# Keep only 'Molecule' and 'ID' columns
	final_df = final_df[["Molecule", "ID"]]

	# Save output to SDF if specified
	if output_sdf:
		PandasTools.WriteSDF(final_df,
				str(output_sdf),
				molColName="Molecule",
				idName="ID",
				properties=list(final_df.columns))
		return output_sdf
	else:
		return final_df
