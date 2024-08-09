import subprocess
import sys
import time
import traceback
from pathlib import Path
import os
from typing import List, Dict

import pandas as pd
from rdkit.Chem import PandasTools

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.setup.software_manager import ensure_software_installed


class PANTHER(ScoringFunction):

	"""
    PANTHER scoring function implementation.
    """

	@ensure_software_installed("PANTHER")
	def __init__(self, score_type: str, software_path: Path):
		if score_type == "PANTHER":
			super().__init__("PANTHER", "PANTHER", "max", (0, 10), software_path)
		elif score_type == "PANTHER-ESP":
			super().__init__("PANTHER-ESP", "PANTHER-ESP", "max", (0, 10), software_path)
		elif score_type == "PANTHER-Shape":
			super().__init__("PANTHER-Shape", "PANTHER-Shape", "max", (0, 10), software_path)
		else:
			raise ValueError(f"Invalid PANTHER score type: {score_type}")

	def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
		"""
        Rescore the molecules in the given SDF file using the PANTHER scoring function.

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
			pocket_definition = kwargs.get("pocket_definition")
			negative_image = self._generate_negative_image(temp_dir, protein_file, pocket_definition)

			split_files_folder = split_sdf_str(Path(temp_dir), sdf_file, n_cpus)
			split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

			rescoring_results = parallel_executor(self._rescore_split_file,
						split_files_sdfs,
						n_cpus,
						display_name=self.name,
						negative_image=negative_image)

			panther_dataframes = self._load_rescoring_results(rescoring_results)
			panther_rescoring_results = self._combine_rescoring_results(panther_dataframes)

			end_time = time.perf_counter()
			printlog(f"Rescoring with {self.name} complete in {end_time - start_time:.4f} seconds!")
			return panther_rescoring_results
		except Exception as e:
			printlog(f"ERROR: An unexpected error occurred during {self.name} rescoring:")
			printlog(traceback.format_exc())
			return pd.DataFrame()
		finally:
			self.remove_temp_dir(temp_dir)

	def _generate_negative_image(self, temp_dir: Path, protein_file: str, pocket_definition: Dict) -> Path:
		"""
        Generate a negative image of the protein binding site.

        Args:
            temp_dir (Path): The temporary directory path.
            protein_file (str): The path to the protein file.
            pocket_definition (Dict): The pocket definition dictionary.

        Returns:
            Path: The path to the generated negative image file.
        """
		try:
			default_in = self.software_path / "default.in"
			panther_input = temp_dir / "panther_input.in"
			negative_image = temp_dir / "negative_image.mol2"

			with default_in.open('r') as f_in, panther_input.open('w') as f_out:
				for line in f_in:
					if line.startswith("1-Pdb file"):
						f_out.write(f"1-Pdb file (-pfil):: {protein_file}\n")
					elif line.startswith("2-Radius library"):
						f_out.write(f"2-Radius library (-rlib):: {self.software_path}/panther/rad.lib\n")
					elif line.startswith("3-Angle library"):
						f_out.write(f"3-Angle library (-alib):: {self.software_path}/panther/angles.lib\n")
					elif line.startswith("4-Charge library file"):
						f_out.write(f"4-Charge library file (-chlib):: {self.software_path}/panther/charges.lib\n")
					elif line.startswith("5-Center(s)"):
						center = pocket_definition['center']
						f_out.write(f"5-Center(s) (-cent):: {center[0]} {center[1]} {center[2]}\n")
					elif line.startswith("9-Box radius"):
						box_size = pocket_definition['size'][0] // 2
						f_out.write(f"9-Box radius (-brad):: {box_size}\n")
					else:
						f_out.write(line)

			panther_cmd = (f"conda run -n panther python {self.software_path}/panther/panther.py"
				f" {panther_input}"
				f" {negative_image}")

			result = subprocess.run(panther_cmd, shell=True, capture_output=True, text=True)

			mol2_start = result.stdout.find("@<TRIPOS>MOLECULE")
			mol2_end = result.stdout.rfind("INFO:")
			if mol2_start != -1 and mol2_end != -1:
				mol2_data = result.stdout[mol2_start:mol2_end].strip()
				with negative_image.open('w') as f:
					f.write(mol2_data)
				printlog(f"Negative image written to {negative_image}")
			else:
				raise RuntimeError("PANTHER failed to generate mol2 file")

			return negative_image
		except Exception as e:
			printlog(f"PANTHER Negative Image Generation failed:")
			printlog(traceback.format_exc())
			raise

	def _rescore_split_file(self, split_file: Path, negative_image: Path) -> Path:
		"""
        Rescore a single split SDF file.

        Args:
            split_file (Path): The path to the split SDF file.
            negative_image (Path): The path to the negative image file.

        Returns:
            Path: The path to the rescored SDF file.
        """
		try:
			mol2_file = split_file.with_suffix('.mol2')
			obabel_cmd = (f"obabel -isdf {split_file}"
							f" -O {mol2_file}"
							" --partialcharge mmff94")
			subprocess.run(obabel_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

			shaep_output_sdf = split_file.parent / f"{split_file.stem}_{self.column_name}.sdf"
			shaep_output_txt = split_file.parent / f"{split_file.stem}_{self.column_name}.txt"

			shaep_cmd = (f"{self.software_path}/shaep"
							f" -q {negative_image}"
							f" {mol2_file}"
							f" -s {shaep_output_sdf}"
							f" --output-file {shaep_output_txt}"
							" --noOptimization")
			subprocess.run(shaep_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

			os.remove(mol2_file)
			return shaep_output_sdf
		except Exception as e:
			printlog(f"SHAEP similarity calculation to negative image failed for {split_file}:")
			printlog(traceback.format_exc())
			return None

	def _load_rescoring_results(self, result_files: List[Path]) -> List[pd.DataFrame]:
		"""
        Load rescoring results from SDF files.

        Args:
            result_files (List[Path]): List of paths to rescored SDF files.

        Returns:
            List[pd.DataFrame]: List of DataFrames containing the rescoring results.
        """
		dataframes = []
		for file in result_files:
			if file and file.is_file():
				try:
					df = PandasTools.LoadSDF(str(file),
							idName="Pose ID",
							molColName=None,
							includeFingerprints=False,
							embedProps=False)
					dataframes.append(df)
				except Exception as e:
					printlog(f"ERROR: Failed to Load {self.column_name} rescoring SDF file: {file}")
					printlog(traceback.format_exc())
		return dataframes

	def _combine_rescoring_results(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
		"""
        Combine rescoring results from multiple DataFrames.

        Args:
            dataframes (List[pd.DataFrame]): List of DataFrames containing rescoring results.

        Returns:
            pd.DataFrame: Combined DataFrame with rescoring results.
        """
		try:
			combined_results = pd.concat(dataframes, ignore_index=True)
			combined_results = combined_results[["Pose ID", "Similarity_best", "Similarity_ESP", "Similarity_shape"]]
			combined_results.rename(columns={
				"Similarity_best": "PANTHER", "Similarity_ESP": "PANTHER-ESP", "Similarity_shape": "PANTHER-Shape"},
					inplace=True)
			return combined_results[["Pose ID", self.column_name]]
		except Exception as e:
			printlog(f"ERROR: Could not combine {self.column_name} rescored poses")
			printlog(traceback.format_exc())
			return pd.DataFrame()
