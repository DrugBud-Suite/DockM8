import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class CHEMPLP(ScoringFunction):

	def __init__(self):
		super().__init__("CHEMPLP", "CHEMPLP", "min", (200, -200))

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		plants_search_speed = "speed1"
		ants = "20"

		temp_dir = self.create_temp_dir()
		try:
			# Convert protein file to .mol2 using open babel
			plants_protein_mol2 = Path(temp_dir) / "protein.mol2"
			convert_molecules(protein_file, plants_protein_mol2, "pdb", "mol2")

			# Convert prepared ligand file to .mol2 using open babel
			plants_ligands_mol2 = Path(temp_dir) / "ligands.mol2"
			convert_molecules(sdf, plants_ligands_mol2, "sdf", "mol2")

			chemplp_rescoring_config_path_txt = Path(temp_dir) / "config.txt"
			chemplp_config = [
				"# search algorithm\n",
				f"search_speed {plants_search_speed}\n",
				f"aco_ants {ants}\n",
				"flip_amide_bonds 0\n",
				"flip_planar_n 1\n",
				"force_flipped_bonds_planarity 0\n",
				"force_planar_bond_rotation 1\n",
				"rescore_mode simplex\n",
				"flip_ring_corners 0\n",
				"# scoring functions\n",
				"# Intermolecular (protein-ligand interaction scoring)\n",
				"scoring_function chemplp\n",
				"outside_binding_site_penalty 50.0\n",
				"enable_sulphur_acceptors 1\n",
				"# Intramolecular ligand scoring\n",
				"ligand_intra_score clash2\n",
				"chemplp_clash_include_14 1\n",
				"chemplp_clash_include_HH 0\n",
				"# input\n",
				f"protein_file {plants_protein_mol2}\n",
				f"ligand_file {plants_ligands_mol2}\n",
				"# output\n",
				f"output_dir {Path(temp_dir) / 'results'}\n",
				"# write single mol2 files (e.g. for RMSD calculation)\n",
				"write_multi_mol2 1\n",
				"# binding site definition\n",
				"# cluster algorithm\n",
				"cluster_structures 10\n",
				"cluster_rmsd 2.0\n",
				"# write\n",
				"write_ranking_links 0\n",
				"write_protein_bindingsite 1\n",
				"write_protein_conformations 1\n",
				"write_protein_splitted 1\n",
				"write_merged_protein 0\n",
				"####\n", ]

			# Write config file
			chemplp_rescoring_config_path_config = chemplp_rescoring_config_path_txt.with_suffix(".config")
			with chemplp_rescoring_config_path_config.open("w") as configwriter:
				configwriter.writelines(chemplp_config)

			# Run PLANTS docking
			chemplp_rescoring_command = f"{software}/PLANTS --mode rescore {chemplp_rescoring_config_path_config}"
			subprocess.call(chemplp_rescoring_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

			# Fetch results
			results_csv_location = Path(temp_dir) / "results" / "ranking.csv"
			chemplp_results = pd.read_csv(results_csv_location, sep=",", header=0)
			chemplp_results.rename(columns={"TOTAL_SCORE": self.column_name}, inplace=True)
			for i, row in chemplp_results.iterrows():
				split = row["LIGAND_ENTRY"].split("_")
				chemplp_results.loc[i, ["Pose ID"]] = f"{split[0]}_{split[1]}_{split[2]}"
			chemplp_rescoring_results = chemplp_results[["Pose ID", self.column_name]]

			toc = time.perf_counter()
			printlog(f"Rescoring with CHEMPLP complete in {toc-tic:0.4f}!")
			return chemplp_rescoring_results
		finally:
			self.remove_temp_dir(temp_dir)


# Usage:
# chemplp = CHEMPLP()
# results = chemplp.rescore(sdf_file, n_cpus, software=software_path, protein_file=protein_file_path)
