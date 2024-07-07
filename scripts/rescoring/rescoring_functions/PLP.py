import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from pandas import DataFrame

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.utilities import delete_files

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class PLP(ScoringFunction):

	def __init__(self):
		super().__init__("PLP", "PLP", "min", (200, -200))

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> DataFrame:
		tic = time.perf_counter()
		rescoring_folder = kwargs.get("rescoring_folder")
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		plants_search_speed = "speed1"
		ants = "20"
		plp_rescoring_folder = Path(rescoring_folder) / f"{self.column_name}_rescoring"
		plp_rescoring_folder.mkdir(parents=True, exist_ok=True)
		# Convert protein file to .mol2 using open babel
		plants_protein_mol2 = plp_rescoring_folder / "protein.mol2"
		convert_molecules(protein_file, plants_protein_mol2, "pdb", "mol2")
		# Convert prepared ligand file to .mol2 using open babel
		plants_ligands_mol2 = plp_rescoring_folder / "ligands.mol2"
		convert_molecules(sdf, plants_ligands_mol2, "sdf", "mol2")

		# Generate plants config file
		plp_rescoring_config_path_txt = plp_rescoring_folder / "config.txt"
		plp_config = [
			"# search algorithm\n",
			"search_speed " + plants_search_speed + "\n",
			"aco_ants " + ants + "\n",
			"flip_amide_bonds 0\n",
			"flip_planar_n 1\n",
			"force_flipped_bonds_planarity 0\n",
			"force_planar_bond_rotation 1\n",
			"rescore_mode simplex\n",
			"flip_ring_corners 0\n",
			"# scoring functions\n",
			"# Intermolecular (protein-ligand interaction scoring)\n",
			"scoring_function plp\n",
			"outside_binding_site_penalty 50.0\n",
			"enable_sulphur_acceptors 1\n",
			"# Intramolecular ligand scoring\n",
			"ligand_intra_score clash2\n",
			"chemplp_clash_include_14 1\n",
			"chemplp_clash_include_HH 0\n",
			"# input\n",
			"protein_file " + str(plants_protein_mol2) + "\n",
			"ligand_file " + str(plants_ligands_mol2) + "\n",
			"# output\n",
			"output_dir " + str(plp_rescoring_folder / "results") + "\n",
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
		plp_rescoring_config_path_config = plp_rescoring_config_path_txt.with_suffix(".config")
		with plp_rescoring_config_path_config.open("w") as configwriter:
			configwriter.writelines(plp_config)

		# Run PLANTS docking
		plp_rescoring_command = f"{software}/PLANTS --mode rescore {plp_rescoring_config_path_config}"
		subprocess.call(plp_rescoring_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

		# Fetch results
		results_csv_location = plp_rescoring_folder / "results" / "ranking.csv"
		plp_results = pd.read_csv(results_csv_location, sep=",", header=0)
		plp_results.rename(columns={"TOTAL_SCORE": self.column_name}, inplace=True)
		for i, row in plp_results.iterrows():
			split = row["LIGAND_ENTRY"].split("_")
			plp_results.loc[i, ["Pose ID"]] = f"{split[0]}_{split[1]}_{split[2]}"
		PLP_rescoring_results = plp_results[["Pose ID", self.column_name]]
		PLP_rescoring_results.to_csv(rescoring_folder / f"{self.column_name}_rescoring" /
										f"{self.column_name}_scores.csv",
										index=False)

		# Remove files
		plants_ligands_mol2.unlink()
		delete_files(rescoring_folder / f"{self.column_name}_rescoring", f"{self.column_name}_scores.csv")
		toc = time.perf_counter()
		printlog(f"Rescoring with PLP complete in {toc-tic:0.4f}!")
		return PLP_rescoring_results
