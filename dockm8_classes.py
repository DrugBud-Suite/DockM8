import argparse
import json
import os
import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import required modules
from scripts.consensus import consensus
from scripts.docking.docking import dockm8_docking
from scripts.docking_postprocessing import docking_postprocessing
from scripts.library_preparation import library_preparation
from scripts.performance_calculation import determine_optimal_conditions
from scripts.pocket_finding.pocket_finder import PocketFinder
from scripts.pose_selection import pose_selection
from scripts.protein_preparation.protein import Protein
from scripts.rescoring import rescoring
from scripts.results import DockM8Results
from scripts.utilities.config_parser import check_config, DockM8Error, DockM8Warning
from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from software.DeepCoy.generate_decoys import generate_decoys

BATCH_SIZES = {100000: 1, 500000: 5, 1000000: 10}


def count_molecules(sdf_file: Path) -> int:
	# Count the number of molecules in an SDF file
	with open(sdf_file, 'r') as f:
		return f.read().count('$$$$')


class DockM8Base(ABC):

	def __init__(self, config_input: Union[str, Path]):
		# Check if the input is a dictionary or a file path
		if isinstance(config_input, Union[str, Path]):
			self.config = check_config(Path(config_input))
		else:
			raise DockM8Error("Invalid configuration input. Expected a str or a file path.")
		self.software = Path(self.config['general']['software'])
		self.w_dir = self._setup_working_directory()
		self.n_cpus = self.config['general']['n_cpus']

	def _setup_working_directory(self) -> Path:
		# Set up the working directory
		w_dir = Path(self.config['receptor(s)'][0]).parent / Path(self.config['receptor(s)'][0]).stem
		w_dir.mkdir(exist_ok=True)
		printlog(f"Working directory set to: {w_dir}")
		return w_dir

	def prepare_protein(self) -> Path:
		# Prepare the protein for docking
		protein = Protein(str(self.config['receptor(s)'][0]), self.w_dir)
		if not protein.is_prepared:
			protein.prepare_protein(**self.config['protein_preparation'])
		return protein.pdb_file

	def determine_pocket(self, prepared_receptor: Path) -> Dict:
		# Determine the binding pocket for docking
		pocket_finder = PocketFinder(software_path=self.software)
		return pocket_finder.find_pocket(
			mode=self.config['pocket_detection']['method'],
			receptor=prepared_receptor,
			ligand=Path(self.config['pocket_detection'].get('reference_ligand(s)', [None])[0]),
			radius=self.config['pocket_detection'].get('radius', 10),
			manual_pocket=self.config['pocket_detection'].get('manual_pocket'),
			dogsitescorer_method=self.config['pocket_detection'].get('dogsitescorer_method', 'Volume'))

	def prepare_ligands(self, input_library: Path, output_dir: Path) -> Path:
		# Prepare ligands for docking
		prepared_library = output_dir / "prepared_library.sdf"
		if not prepared_library.exists():
			prepared_library = library_preparation.prepare_library(input_data=input_library,
																	**self.config['ligand_preparation'],
																	software=self.software,
																	n_cpus=self.n_cpus,
																	output_sdf=prepared_library)
		return prepared_library

	def _run_single_docking(self,
							prepared_library: Path,
							prepared_receptor: Path,
							pocket_definition: Dict,
							output_dir: Path) -> Tuple[Path, Dict[str, Path]]:
		all_poses_path = dockm8_docking(library=prepared_library,
										w_dir=output_dir,
										protein_file=prepared_receptor,
										pocket_definition=pocket_definition,
										software=self.software,
										docking_programs=self.config['docking']['docking_programs'],
										exhaustiveness=self.config['docking']['exhaustiveness'],
										n_poses=self.config['docking']['n_poses'],
										n_cpus=self.n_cpus)
		docked_poses = {
			program: output_dir / program.lower() / f"{program.lower()}_poses.sdf"
			for program in self.config['docking']['docking_programs']}
		return all_poses_path, docked_poses

	def post_process(self, docked_poses: Path, prepared_receptor: Path, output_dir: Path) -> Path:
		# Post-process docking results
		processed_poses = output_dir / "processed_poses.sdf"
		if not processed_poses.exists():
			processed_poses = docking_postprocessing.docking_postprocessing(input_data=docked_poses,
																			protein_file=prepared_receptor,
																			**self.config['post_docking'],
																			n_cpus=self.n_cpus,
																			output_sdf=processed_poses)
		return processed_poses

	def select_poses(self, processed_poses: Path, prepared_receptor: Path, pocket_definition: Dict,
						output_dir: Path) -> Dict[str, Path]:
		# Select poses based on specified methods
		selected_poses = {}
		for method in self.config['pose_selection']['pose_selection_method']:
			output_file = output_dir / f"{method}_selected.sdf"
			selected_poses[method] = pose_selection.select_poses(
				poses=processed_poses,
				selection_method=method,
				clustering_method=self.config['pose_selection']['clustering_method'],
				pocket_definition=pocket_definition,
				protein_file=prepared_receptor,
				software=self.software,
				n_cpus=self.n_cpus,
				output_file=output_file)
		return selected_poses

	def rescore_poses(self,
						selected_poses: Dict[str, Path],
						prepared_receptor: Path,
						pocket_definition: Dict,
						output_dir: Path) -> Dict[str, Path]:
		# Rescore selected poses
		rescored_poses = {}
		for method, poses in selected_poses.items():
			output_file = output_dir / f"{method}_rescored.csv"
			rescored_poses[method] = rescoring.rescore_poses(protein_file=prepared_receptor,
																pocket_definition=pocket_definition,
																software=self.software,
																poses=poses,
																functions=self.config['rescoring'],
																n_cpus=self.n_cpus,
																output_file=output_file)
		return rescored_poses

	def apply_consensus(self, rescored_poses: Path, method) -> Path:
		# Apply consensus methods to rescored poses
		output_path = self.w_dir / f"consensus/{method}_rescored_consensus.csv"
		return consensus.apply_consensus_methods(poses_input=rescored_poses,
													consensus_methods=self.config['consensus'],
													standardization_type="min_max",
													output_path=output_path)

	def create_batches(self, input_file: Path) -> List[Path]:
		# Create batches for processing
		num_molecules = count_molecules(input_file)

		# Determine batch size using the BATCH_SIZES dictionary
		batch_size = 0
		for threshold, size in sorted(BATCH_SIZES.items(), reverse=True):
			if num_molecules >= threshold:
				batch_size = size
				break

		if batch_size > 0:
			split_dir = split_sdf_str(self.w_dir, input_file, batch_size)
			return list(split_dir.glob("split_*.sdf"))
		else:
			return [input_file]

	def process_batch(
			self, batch: Path, prepared_receptor: Path,
			pocket_definition: Dict) -> Tuple[Path, Path, Dict[str, Path], Path, Dict[str, Path], Dict[str, Path]]:
		# Process a single batch through the entire pipeline
		batch_dir = self.w_dir / f"batch_{batch.stem.split('_')[-1]}"
		batch_dir.mkdir(exist_ok=True)
		printlog(f"Processing batch: {batch_dir}")

		prepared_batch = self.prepare_ligands(batch, batch_dir)
		all_poses_path, docked_poses = self._run_single_docking(prepared_batch, prepared_receptor, pocket_definition, batch_dir)
		processed_poses = self.post_process(all_poses_path, prepared_receptor, batch_dir)
		selected_poses = self.select_poses(processed_poses, prepared_receptor, pocket_definition, batch_dir)
		rescored_poses = self.rescore_poses(selected_poses, prepared_receptor, pocket_definition, batch_dir)

		return prepared_batch, all_poses_path, docked_poses, processed_poses, selected_poses, rescored_poses

	def run_with_batches(self) -> DockM8Results:
		prepared_receptor = self.prepare_protein()
		pocket_definition = self.determine_pocket(prepared_receptor)

		batches = self.create_batches(self.config['docking_library'])

		all_docked = {program: [] for program in self.config['docking']['docking_programs']}
		all_processed = []
		all_selected, all_rescored = {}, {}

		for batch in batches:
			prepared, all, docked, processed, selected, rescored = self.process_batch(batch, prepared_receptor, pocket_definition)

			for program, path in docked.items():
				all_docked[program].append(path)
			all_processed.append(processed)

			for method, path in selected.items():
				all_selected.setdefault(method, []).append(path)

			for method, path in rescored.items():
				all_rescored.setdefault(method, []).append(path)

		# Combine results
		docked_poses = {
			program: self._combine_sdf_results(paths, Path(program / f"{program}_poses.sdf")) for program,
			paths in all_docked.items()}
		all_poses = self._combine_sdf_results([all], "all_poses.sdf")
		processed_poses = self._combine_sdf_results(all_processed, "all_poses_processed.sdf")
		selected_poses = self._combine_selected_poses(all_selected)
		rescored_poses = self._combine_rescored_poses(all_rescored)

		# Apply consensus
		consensus_results = self._apply_consensus_to_combined(rescored_poses)

		return DockM8Results(docked_poses=docked_poses,
								all_poses=all_poses,
								processed_poses=processed_poses,
								selected_poses=selected_poses,
								rescored_poses=rescored_poses,
								consensus_results=consensus_results,
								docking_programs=self.config['docking']['docking_programs'],
								selection_methods=self.config['pose_selection']['pose_selection_method'],
							)

	def _combine_sdf_results(self, file_list: List[Path], output_filename: str) -> Path:
		# Combine SDF results from multiple files
		(self.w_dir / output_filename).parent.mkdir(exist_ok=True, parents=True)
		with open(self.w_dir / output_filename, 'w') as outfile:
			for file in file_list:
				outfile.write(file.read_text())
		return self.w_dir / output_filename

	def _combine_selected_poses(self, all_selected: Dict[str, List[Path]]) -> Dict[str, Path]:
		# Combine selected poses
		selected_dir = self.w_dir / "selected_poses"
		selected_dir.mkdir(exist_ok=True)
		selected_poses = {}
		for method, paths in all_selected.items():
			poses_from_method = self._combine_sdf_results(paths, selected_dir / f"{method}_selected.sdf")
			selected_poses[method] = poses_from_method
		return selected_poses

	def _combine_rescored_poses(self, all_rescored: Dict[str, List[Path]]) -> Dict[str, Path]:
		# Combine rescored poses
		rescoring_dir = self.w_dir / "rescoring"
		rescoring_dir.mkdir(exist_ok=True)
		rescored_poses = {}
		for method, paths in all_rescored.items():
			combined_df = pd.concat([pd.read_csv(file) for file in paths], ignore_index=True)
			combined_df.to_csv(rescoring_dir / f"{method}_rescored.csv", index=False)
			rescored_poses[method] = rescoring_dir / f"{method}_rescored.csv"
		return rescored_poses

	def run_without_batches(self) -> DockM8Results:
		prepared_receptor = self.prepare_protein()
		pocket_definition = self.determine_pocket(prepared_receptor)
		prepared_ligands = self.prepare_ligands(self.config['docking_library'], self.w_dir)
		all_poses, docked_poses = self._run_single_docking(prepared_ligands, prepared_receptor, pocket_definition, self.w_dir)
		processed_poses = self.post_process(all_poses, prepared_receptor, self.w_dir)
		selected_poses = self.select_poses(processed_poses, prepared_receptor, pocket_definition, self.w_dir)
		rescored_poses = self.rescore_poses(selected_poses, prepared_receptor, pocket_definition, self.w_dir)

		consensus_results = self._apply_consensus_to_combined(rescored_poses)

		return DockM8Results(docked_poses=docked_poses,
								all_poses=all_poses,
								processed_poses=processed_poses,
								selected_poses=selected_poses,
								rescored_poses=rescored_poses,
								consensus_results=consensus_results,
								docking_programs=self.config['docking']['docking_programs'],
								selection_methods=self.config['pose_selection']['pose_selection_method'])

	def _apply_consensus_to_combined(self, rescored_poses: Dict[str, Path]) -> Dict[str, Path]:
		consensus_results = {}
		for method, rescored_file in rescored_poses.items():
			consensus_results[method] = self.apply_consensus(pd.read_csv(rescored_file), method)
		return consensus_results

	@abstractmethod
	def run(self):
		# Abstract method to be implemented by subclasses
		pass


class DockM8Standard(DockM8Base):

	def run(self) -> pd.DataFrame:
		# Run the standard DockM8 pipeline
		if len(self.create_batches(self.config['docking_library'])) > 1:
			return self.run_with_batches()
		else:
			return self.run_without_batches()


class DecoyHandler:

	def __init__(self, config: Dict):
		# Initialize the DecoyHandler
		self.config = config
		self.decoy_library = None

	def generate_decoys(self) -> Path:
		# Generate decoys for the active compounds
		self.decoy_library = generate_decoys(Path(self.config['decoy_generation']['actives']),
												self.config['decoy_generation']['n_decoys'],
												self.config['decoy_generation']['decoy_model'],
												Path(self.config['general']['software']))
		return self.decoy_library

	def determine_optimal_conditions(self, results: pd.DataFrame, w_dir: Path) -> Dict:
		# Determine optimal conditions based on decoy results
		optimal_conditions = determine_optimal_conditions(w_dir, results, self.decoy_library, [10, 5, 2, 1, 0.5])
		printlog(f"Optimal conditions: {optimal_conditions}")
		return optimal_conditions


class DockM8Decoys(DockM8Base):

	def __init__(self, config: Dict):
		# Initialize DockM8Decoys
		super().__init__(config)
		self.decoy_handler = DecoyHandler(config)

	def run(self) -> pd.DataFrame:
		# Run DockM8 with decoy generation and optimization
		self.decoy_handler.generate_decoys()
		initial_results = self.run_with_batches() if len(self.create_batches(
			self.config['docking_library'])) > 1 else self.run_without_batches()
		optimal_conditions = self.decoy_handler.determine_optimal_conditions(initial_results, self.w_dir)
		self.update_config_with_optimal_conditions(optimal_conditions)
		return self.run_with_batches() if len(self.create_batches(
			self.config['docking_library'])) > 1 else self.run_without_batches()

	def update_config_with_optimal_conditions(self, optimal_conditions: Dict):
		# Update configuration with optimal conditions
		self.config['docking']['docking_programs'] = [optimal_conditions['docking_program']]
		self.config['pose_selection']['pose_selection_method'] = [optimal_conditions['selection_method']]
		self.config['rescoring'] = [
			func for func in optimal_conditions['scoring'].split('_') if func in rescoring.RESCORING_FUNCTIONS]
		self.config['consensus'] = optimal_conditions['consensus']


class EnsembleHandler:

	def __init__(self, config: Dict):
		# Initialize EnsembleHandler
		self.config = config
		self.receptor_dict = self._create_receptor_dict()

	def _create_receptor_dict(self) -> Dict[Path, Optional[Path]]:
		# Create a dictionary of receptors and their reference ligands
		return {
			Path(receptor): (Path(self.config['pocket_detection']['reference_ligand(s)'][i])
								if self.config['pocket_detection']['reference_ligand(s)'] else None) for i,
			receptor in enumerate(self.config['receptor(s)'])}


class DockM8Ensemble(DockM8Base):

	def __init__(self, config: Dict):
		# Initialize DockM8Ensemble
		super().__init__(config)
		self.ensemble_handler = EnsembleHandler(config)

	def run(self) -> Dict[str, pd.DataFrame]:
		# Run DockM8 for an ensemble of receptors
		ensemble_results = {}
		for receptor, reference_ligand in self.ensemble_handler.receptor_dict.items():
			printlog(f"Processing receptor: {receptor}")
			self.config['receptor(s)'][0] = receptor
			self.config['pocket_detection']['reference_ligand(s)'] = [reference_ligand] if reference_ligand else None
			self.w_dir = self._setup_working_directory()

			results = self.run_with_batches() if len(self.create_batches(
				self.config['docking_library'])) > 1 else self.run_without_batches()
			ensemble_results[receptor.stem] = results

		return ensemble_results


class DockM8EnsembleDecoys(DockM8Base):

	def __init__(self, config: Dict):
		# Initialize DockM8EnsembleDecoys
		super().__init__(config)
		self.ensemble_handler = EnsembleHandler(config)
		self.decoy_handler = DecoyHandler(config)
		self.optimal_conditions = None

	def run(self) -> Dict[str, pd.DataFrame]:
		# Run DockM8 for an ensemble of receptors with decoy generation and optimization
		self.decoy_handler.generate_decoys()
		ensemble_results = {}

		for i, (receptor, reference_ligand) in enumerate(self.ensemble_handler.receptor_dict.items()):
			printlog(f"Processing receptor: {receptor}")
			self.config['receptor(s)'][0] = receptor
			self.config['pocket_detection']['reference_ligand(s)'] = [reference_ligand] if reference_ligand else None
			self.w_dir = self._setup_working_directory()

			if i == 0:
				initial_results = self.run_with_batches() if len(self.create_batches(
					self.config['docking_library'])) > 1 else self.run_without_batches()
				self.optimal_conditions = self.decoy_handler.determine_optimal_conditions(initial_results, self.w_dir)
				self.update_config_with_optimal_conditions(self.optimal_conditions)
				self._save_optimal_conditions()
			else:
				self._load_and_apply_optimal_conditions()

			results = self.run_with_batches() if len(self.create_batches(
				self.config['docking_library'])) > 1 else self.run_without_batches()
			ensemble_results[receptor.stem] = results

		return ensemble_results

	def update_config_with_optimal_conditions(self, optimal_conditions: Dict):
		# Update configuration with optimal conditions
		self.config['docking']['docking_programs'] = [optimal_conditions['docking_program']]
		self.config['pose_selection']['pose_selection_method'] = [optimal_conditions['selection_method']]
		self.config['rescoring'] = [
			func for func in optimal_conditions['scoring'].split('_') if func in rescoring.RESCORING_FUNCTIONS]
		self.config['consensus'] = optimal_conditions['consensus']

	def _save_optimal_conditions(self):
		# Save optimal conditions to a file
		optimal_conditions_file = self.w_dir.parent / "optimal_conditions.json"
		with open(optimal_conditions_file, 'w') as f:
			json.dump(self.optimal_conditions, f, indent=2)
		printlog(f"Optimal conditions saved to: {optimal_conditions_file}")

	def _load_and_apply_optimal_conditions(self):
		# Load and apply optimal conditions from a file
		optimal_conditions_file = self.w_dir.parent / "optimal_conditions.json"
		if optimal_conditions_file.exists():
			with open(optimal_conditions_file, 'r') as f:
				self.optimal_conditions = json.load(f)
			self.update_config_with_optimal_conditions(self.optimal_conditions)
			printlog(f"Loaded and applied optimal conditions from: {optimal_conditions_file}")
		else:
			printlog("Warning: Optimal conditions file not found. Using current configuration.")


def create_dockm8(config: Dict) -> DockM8Base:
	# Create the appropriate DockM8 instance based on configuration
	mode = config['general']['mode']
	decoys = config['decoy_generation']['gen_decoys']

	if mode == 'ensemble' and decoys:
		return DockM8EnsembleDecoys(config)
	elif mode == 'ensemble':
		return DockM8Ensemble(config)
	elif decoys:
		return DockM8Decoys(config)
	else:
		return DockM8Standard(config)


def run_dockm8(config: Dict):
	# Run the DockM8 pipeline
	dockm8 = create_dockm8(config)
	results = dockm8.run()
	printlog("DockM8 run completed successfully.")
	return results


if __name__ == "__main__":
	# Main execution point
	parser = argparse.ArgumentParser(description="Run DockM8 docking pipeline")
	parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
	args = parser.parse_args()

	results = run_dockm8(Path(args.config))
