import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import PandasTools

from scripts.docking.docking_function import DockingFunction
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules


class PlantsDocking(DockingFunction):

	def __init__(self, software_path: Path):
		super().__init__("PLANTS", software_path)

	def dock_batch(self,
					batch_file: Path,
					protein_file: Path,
					pocket_definition: Dict[str, list],
					exhaustiveness: int,
					n_poses: int) -> Path:
		RDLogger.DisableLog("rdApp.*")
		temp_dir = self.create_temp_dir()
		results_folder = temp_dir / "results"

		# Convert molecules to mol2 format
		plants_protein_mol2 = temp_dir / "protein.mol2"
		try:
			convert_molecules(protein_file, plants_protein_mol2, "pdb", "mol2", self.software_path)
		except Exception as e:
			printlog(f"ERROR: Failed to convert protein file to mol2: {str(e)}")
			self.remove_temp_dir(temp_dir)
			return None

		plants_ligands_mol2 = temp_dir / f"{batch_file.stem}.mol2"
		try:
			convert_molecules(batch_file, plants_ligands_mol2, "sdf", "mol2", self.software_path)
		except Exception as e:
			printlog(f"ERROR: Failed to convert ligands file to mol2: {str(e)}")
			self.remove_temp_dir(temp_dir)
			return None

		# Generate PLANTS config file
		plants_docking_config_path = temp_dir / "plants_config.txt"
		self.generate_plants_config(plants_protein_mol2,
									plants_ligands_mol2,
									pocket_definition,
									n_poses,
									results_folder,
									plants_docking_config_path)

		# Run PLANTS docking
		try:
			plants_docking_command = f'{self.software_path / "PLANTS"} --mode screen {plants_docking_config_path}'
			subprocess.run(plants_docking_command,
							shell=True,
							check=True,
							stdout=subprocess.DEVNULL,
							stderr=subprocess.STDOUT)
		except subprocess.CalledProcessError as e:
			printlog(f"ERROR: PLANTS docking command failed: {str(e)}")
			self.remove_temp_dir(temp_dir)
			return None

		# Post-process results
		results_mol2 = results_folder / "docked_ligands.mol2"
		results_sdf = results_mol2.with_suffix(".sdf")
		try:
			convert_molecules(results_mol2, results_sdf, "mol2", "sdf", self.software_path)
		except Exception as e:
			printlog(f"ERROR: Failed to convert PLANTS poses file to .sdf: {str(e)}")
			self.remove_temp_dir(temp_dir)
			return None

		return results_sdf

	def process_docking_result(self, result_file: Path, n_poses: int) -> pd.DataFrame:
		RDLogger.DisableLog("rdApp.*")
		try:
			plants_poses = PandasTools.LoadSDF(str(result_file), idName="ID", molColName="Molecule")
			plants_scores = pd.read_csv(str(result_file.parent / "ranking.csv")).rename(columns={
				"LIGAND_ENTRY": "ID", "TOTAL_SCORE": "CHEMPLP"})[["ID", "CHEMPLP"]]
			df = pd.merge(plants_scores, plants_poses, on="ID")
			df["ID"] = df["ID"].str.split("_").str[0]
			df["Pose Rank"] = df.groupby("ID").cumcount() + 1
			df = df[df["Pose Rank"] <= n_poses]
			df["Pose ID"] = df.apply(lambda row: f'{row["ID"]}_PLANTS_{row["Pose Rank"]}', axis=1)
			return df

		except Exception as e:
			printlog(f"ERROR: Failed to process PLANTS result file {result_file}: {str(e)}")
			return pd.DataFrame()     # Return an empty DataFrame on error
		finally:
			self.remove_temp_dir(result_file.parent)

	def generate_plants_config(self,
								protein_mol2: Path,
								ligands_mol2: Path,
								pocket_definition: dict,
								n_poses: int,
								output_dir: Path,
								config_path: Path):
		config_lines = [
			"# search algorithm\n",
			"search_speed speed1\n",
			"aco_ants 20\n",
			"flip_amide_bonds 0\n",
			"flip_planar_n 1\n",
			"force_flipped_bonds_planarity 0\n",
			"force_planar_bond_rotation 1\n",
			"rescore_mode simplex\n",
			"flip_ring_corners 0\n",
			"# scoring functions\n",
			"scoring_function chemplp\n",
			"outside_binding_site_penalty 50.0\n",
			"enable_sulphur_acceptors 1\n",
			"# Intramolecular ligand scoring\n",
			"ligand_intra_score clash2\n",
			"chemplp_clash_include_14 1\n",
			"chemplp_clash_include_HH 0\n",
			"# input\n",
			f"protein_file {protein_mol2}\n",
			f"ligand_file {ligands_mol2}\n",
			"# output\n",
			f"output_dir {output_dir}\n",
			"# write single mol2 files\n",
			"write_multi_mol2 1\n",
			"# binding site definition\n",
			f'bindingsite_center {pocket_definition["center"][0]} {pocket_definition["center"][1]} {pocket_definition["center"][2]}\n',
			f'bindingsite_radius {pocket_definition["size"][0] / 2}\n',
			"# cluster algorithm\n",
			f"cluster_structures {n_poses}\n",
			"cluster_rmsd 2.0\n",
			"# write\n",
			"write_ranking_links 0\n",
			"write_protein_bindingsite 0\n",
			"write_protein_conformations 0\n",
			"write_protein_splitted 0\n",
			"write_merged_protein 0\n",
			"####\n", ]
		with open(config_path, "w") as config_writer:
			config_writer.writelines(config_lines)
