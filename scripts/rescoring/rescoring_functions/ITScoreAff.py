import subprocess
import sys
import time
import traceback
from pathlib import Path
import os
from typing import List

import pandas as pd
from rdkit.Chem import PandasTools

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.setup.software_manager import ensure_software_installed
from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.parallel_executor import parallel_executor


class ITScoreAff(ScoringFunction):

	"""
    ITScoreAff scoring function implementation.
    """

	def __init__(self, software_path: Path):
		super().__init__("ITScoreAff", "ITScoreAff", "min", (-200, 100), software_path)
		self.software_path = software_path
		ensure_software_installed("IT_SCORE_AFF", software_path)

	def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
		"""
        Rescore the molecules in the given SDF file using the ITScoreAff scoring function.

        Args:
            sdf_file (str): The path to the SDF file.
            n_cpus (int): The number of CPUs to use for parallel processing.
            protein_file (str): The path to the protein file.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: A DataFrame containing the rescored molecules.
        """
		start_time = time.perf_counter()

		temp_dir = self.create_temp_dir()
		try:
			split_files_folder = split_sdf_str(Path(temp_dir), sdf_file, n_cpus)
			split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

			protein_mol2 = temp_dir / 'protein.mol2'
			try:
				convert_molecules(Path(protein_file), protein_mol2, 'pdb', 'mol2')
			except Exception as e:
				printlog(f"Error converting protein file to .mol2:")
				printlog(traceback.format_exc())
				return pd.DataFrame()

			rescoring_results = parallel_executor(self._rescore_split_file,
													split_files_sdfs,
													n_cpus,
													display_name=self.name,
													protein_mol2=protein_mol2)

			itscoreaff_rescoring_results = self._combine_rescoring_results(rescoring_results)

			end_time = time.perf_counter()
			printlog(f"Rescoring with ITScoreAff complete in {end_time - start_time:.4f} seconds!")
			return itscoreaff_rescoring_results
		except Exception as e:
			printlog(f"ERROR: An unexpected error occurred during ITScoreAff rescoring:")
			printlog(traceback.format_exc())
			return pd.DataFrame()
		finally:
			self.remove_temp_dir(temp_dir)

	def _rescore_split_file(self, split_file: Path, protein_mol2: Path) -> Path:
		"""
        Rescore a single split SDF file.

        Args:
            split_file (Path): The path to the split SDF file.
            protein_mol2 (Path): The path to the protein MOL2 file.

        Returns:
            Path: The path to the rescored CSV file.
        """
		try:
			df = PandasTools.LoadSDF(str(split_file), idName="Pose ID", molColName=None)
			df = df[["Pose ID"]]

			ligand_mol2 = split_file.with_suffix('.mol2')
			convert_molecules(split_file, ligand_mol2, 'sdf', 'mol2')

			itscoreaff_cmd = (f"{self.software_path}/ITScoreAff_v1.0/ITScoreAff"
				f" {protein_mol2.name}"
				f" {ligand_mol2.name}")

			result = subprocess.run(itscoreaff_cmd,
				shell=True,
				capture_output=True,
				text=True,
				check=True,
				cwd=split_file.parent)

			scores = []
			for line in result.stdout.splitlines()[1:]:
				parts = line.split()
				if len(parts) >= 3:
					try:
						score = round(float(parts[2]), 2)
						scores.append(score)
					except ValueError:
						printlog(
							f"Warning: Could not convert '{parts[2]}' to float for file {split_file}. Skipping this score."
						)
						scores.append(None)

			df[self.column_name] = scores
			output_csv = split_file.parent / f"{split_file.stem}_scores.csv"
			df.to_csv(output_csv, index=False)
			return output_csv
		except Exception as e:
			printlog(f"ITScoreAff rescoring failed for {split_file}:")
			printlog(traceback.format_exc())
			printlog(f"stdout: {result.stdout}")
			return None

	def _combine_rescoring_results(self, result_files: List[Path]) -> pd.DataFrame:
		"""
        Combine rescoring results from multiple CSV files.

        Args:
            result_files (List[Path]): List of paths to rescored CSV files.

        Returns:
            pd.DataFrame: Combined DataFrame with rescoring results.
        """
		try:
			dataframes = []
			for file in result_files:
				if file and file.is_file():
					df = pd.read_csv(file)
					dataframes.append(df)

			if not dataframes:
				printlog("No valid CSV files found with ITScoreAff scores.")
				return pd.DataFrame()

			combined_results = pd.concat(dataframes, ignore_index=True)
			return combined_results[["Pose ID", self.column_name]]
		except Exception as e:
			printlog(f"ERROR: Could not combine {self.column_name} rescored poses")
			printlog(traceback.format_exc())
			return pd.DataFrame()
