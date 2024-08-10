import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.setup.software_manager import ensure_software_installed


class CHEMPLP(ScoringFunction):

	"""
    CHEMPLP scoring function implementation.
    """

	@ensure_software_installed("PLANTS")
	def __init__(self, software_path: Path):
		super().__init__("CHEMPLP", "CHEMPLP", "min", (200, -200), software_path)

	def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
		"""
        Rescore the molecules in the given SDF file using the CHEMPLP scoring function.

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
			plants_protein_mol2 = temp_dir / "protein.mol2"
			plants_ligands_mol2 = temp_dir / "ligands.mol2"

			try:
				convert_molecules(protein_file, plants_protein_mol2, "pdb", "mol2")
				convert_molecules(sdf_file, plants_ligands_mol2, "sdf", "mol2")
			except Exception as e:
				printlog("Error converting molecules:")
				printlog(traceback.format_exc())
				return pd.DataFrame()

			config_file = self._create_config_file(temp_dir, plants_protein_mol2, plants_ligands_mol2)

			chemplp_cmd = (f"{self.software_path}/PLANTS"
				f" --mode rescore"
				f" {config_file}")

			try:
				subprocess.run(chemplp_cmd,
					shell=True,
					check=True,
					stdout=subprocess.DEVNULL,
					stderr=subprocess.DEVNULL)
			except subprocess.CalledProcessError as e:
				printlog("Error running PLANTS docking:")
				printlog(traceback.format_exc())
				return pd.DataFrame()

			results_csv = temp_dir / "results" / "ranking.csv"
			if not results_csv.exists():
				printlog(f"Results file not found: {results_csv}")
				return pd.DataFrame()

			chemplp_results = pd.read_csv(results_csv, sep=",", header=0)
			chemplp_results.rename(columns={"TOTAL_SCORE": self.column_name}, inplace=True)
			chemplp_results["Pose ID"] = chemplp_results["LIGAND_ENTRY"].apply(lambda x: "_".join(x.split("_")[:3]))
			chemplp_rescoring_results = chemplp_results[["Pose ID", self.column_name]]

			end_time = time.perf_counter()
			printlog(f"Rescoring with CHEMPLP complete in {end_time - start_time:.4f} seconds!")
			return chemplp_rescoring_results
		except Exception as e:
			printlog(f"ERROR: An unexpected error occurred during CHEMPLP rescoring:")
			printlog(traceback.format_exc())
			return pd.DataFrame()
		finally:
			self.remove_temp_dir(temp_dir)

	def _create_config_file(self, temp_dir: Path, protein_file: Path, ligands_file: Path) -> Path:
		"""
        Create a configuration file for PLANTS.

        Args:
            temp_dir (Path): The temporary directory path.
            protein_file (Path): The path to the protein file.
            ligands_file (Path): The path to the ligands file.

        Returns:
            Path: The path to the created configuration file.
        """
		config_content = [
			"# search algorithm",
			"search_speed speed1",
			"aco_ants 20",
			"flip_amide_bonds 0",
			"flip_planar_n 1",
			"force_flipped_bonds_planarity 0",
			"force_planar_bond_rotation 1",
			"rescore_mode simplex",
			"flip_ring_corners 0",
			"# scoring functions",
			"scoring_function chemplp",
			"outside_binding_site_penalty 50.0",
			"enable_sulphur_acceptors 1",
			"ligand_intra_score clash2",
			"chemplp_clash_include_14 1",
			"chemplp_clash_include_HH 0",
			f"protein_file {protein_file}",
			f"ligand_file {ligands_file}",
			f"output_dir {temp_dir / 'results'}",
			"write_multi_mol2 1",
			"cluster_structures 10",
			"cluster_rmsd 2.0",
			"write_ranking_links 0",
			"write_protein_bindingsite 0",
			"write_protein_conformations 0",
			"write_protein_splitted 0",
			"write_merged_protein 0", ]

		config_file = temp_dir / "config.txt"
		with config_file.open("w") as f:
			f.write("\n".join(config_content))

		return config_file
