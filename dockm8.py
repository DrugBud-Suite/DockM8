# Import required libraries and scripts
import argparse
import sys
import warnings
from pathlib import Path
import os

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

# Import modules for docking, scoring, protein and ligand preparation, etc.
from scripts.utilities.config_parser import check_config
from scripts.pose_selection.pose_selection import select_poses
from scripts.consensus.consensus import apply_consensus_methods
from scripts.docking.docking import dockm8_docking, DOCKING_PROGRAMS
from scripts.library_preparation.library_preparation import prepare_library
from scripts.docking_postprocessing.docking_postprocessing import docking_postprocessing
from scripts.performance_calculation import determine_optimal_conditions
from scripts.pocket_finding.pocket_finder import PocketFinder
from scripts.protein_preparation.protein import Protein
from scripts.rescoring.rescoring import rescore_poses, RESCORING_FUNCTIONS
from scripts.utilities.logging import printlog
from software.DeepCoy.generate_decoys import generate_decoys

# Suppress warnings to clean up output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize argument parser to handle command line arguments
parser = argparse.ArgumentParser(description="Parse required arguments")

parser.add_argument("--config", type=str, help="Path to the configuration file")

config = check_config(Path(parser.parse_args().config))


# Main function to manage docking process
def dockm8(software: Path,
			receptor: Path,
			prepare_proteins: dict,
			ligand_preparation: dict,
			pocket_detection: dict,
			reference_ligand: Path,
			docking_library: Path,
			docking: dict,
			post_docking: dict,
			pose_selection: dict,
			n_cpus: int,
			rescoring: list,
			consensus: str,
			threshold: float):

	# Set working directory based on the receptor file
	w_dir = Path(receptor).parent / Path(receptor).stem
	printlog(f"The working directory has been set to: {w_dir}")
	w_dir.mkdir(exist_ok=True)

	# Create Protein object and prepare it
	protein = Protein(str(receptor), w_dir)
	if not protein.is_prepared:
		protein.prepare_protein(**prepare_proteins)
	prepared_receptor = protein.pdb_file

	# Prepare the ligands for docking
	if not (w_dir / "prepared_library.sdf").exists():
		prepared_library = prepare_library(input_data=docking_library,
											**ligand_preparation,
											software=software,
											n_cpus=n_cpus,
											output_sdf=w_dir / "prepared_library.sdf")
	else:
		prepared_library = w_dir / "prepared_library.sdf"

	# Determine the docking pocket
	pocket_finder = PocketFinder(software_path=software)
	pocket_definition = pocket_finder.find_pocket(mode=pocket_detection["method"],
													receptor=prepared_receptor,
													ligand=reference_ligand if reference_ligand else None,
													radius=pocket_detection.get("radius", 10),
													manual_pocket=pocket_detection.get("manual_pocket", None),
													dogsitescorer_method=pocket_detection.get(
														"dogsitescorer_method", 'Volume'))

	# Perform the docking operation
	all_poses_path = dockm8_docking(library=prepared_library,
									w_dir=w_dir,
									protein_file=prepared_receptor,
									pocket_definition=pocket_definition,
									software=software,
									docking_programs=docking["docking_programs"],
									exhaustiveness=docking["exhaustiveness"],
									n_poses=docking["n_poses"],
									n_cpus=n_cpus)

	# Postprocessing
	if not (w_dir / "allposes_processed.sdf").exists():
		processed_poses = docking_postprocessing(input_data=all_poses_path,
													protein_file=prepared_receptor,
													minimize_poses=post_docking["minimize_poses"],
													bust_poses=post_docking["bust_poses"],
													strain_cutoff=post_docking["strain_cutoff"],
													clash_cutoff=post_docking["clash_cutoff"],
													classy_pose=post_docking["classy_pose"],
													classy_pose_model=post_docking["classy_pose_model"],
													n_cpus=n_cpus,
													output_sdf=w_dir / "allposes_processed.sdf")

	# Create clustering directory
	(w_dir / "clustering").mkdir(exist_ok=True)

	# Select Poses
	if isinstance(pose_selection["pose_selection_method"], str):
		pose_selection_methods = [pose_selection["pose_selection_method"]]
	for method in pose_selection_methods:
		selected_poses = select_poses(poses=processed_poses,
										selection_method=method,
										clustering_method=pose_selection["clustering_method"],
										pocket_definition=pocket_definition,
										protein_file=prepared_receptor,
										software=software,
										n_cpus=n_cpus,
										output_file=w_dir / f"clustering/{method}_clustered.sdf")

		# Create rescoring directory
		(w_dir / "rescoring").mkdir(exist_ok=True)

		# Rescore poses for each selection method
		rescored_poses = rescore_poses(protein_file=prepared_receptor,
										pocket_definition=pocket_definition,
										software=software,
										poses=selected_poses,
										functions=rescoring,
										n_cpus=n_cpus,
										output_file=w_dir / f"rescoring/{method}_rescored.csv")

		# Create consensus directory
		(w_dir / "consensus").mkdir(exist_ok=True)

		# Apply consensus methods to the poses
		consensus_results = apply_consensus_methods(poses_input=rescored_poses,
													consensus_methods=consensus,
													standardization_type="min_max",
													output_path=w_dir / "consensus")

	return consensus_results


def dockm8_decoys(software: Path,
					receptor: Path,
					prepare_proteins: dict,
					ligand_preparation: dict,
					pocket_detection: dict,
					reference_ligand: Path,
					docking_library: Path,
					docking: dict,
					post_docking: dict,
					pose_selection: dict,
					n_cpus: int,
					rescoring: list,
					consensus: str,
					threshold: float):

	# Set working directory based on the receptor file
	w_dir = Path(receptor).parent / (Path(receptor).stem + "_decoys")
	printlog(f"The working directory has been set to: {w_dir}")
	w_dir.mkdir(exist_ok=True)

	# Create Protein object and prepare it
	protein = Protein(str(receptor), w_dir)
	if not protein.is_prepared:
		protein.prepare_protein(**prepare_proteins)
	prepared_receptor = protein.pdb_file

	# Prepare the ligands for docking
	if not (w_dir / "prepared_library.sdf").exists():
		prepared_library = prepare_library(input_data=docking_library,
											**ligand_preparation,
											software=software,
											n_cpus=n_cpus,
											output_sdf=w_dir / "prepared_library.sdf")
	else:
		prepared_library = w_dir / "prepared_library.sdf"

	# Determine the docking pocket
	pocket_finder = PocketFinder(software_path=software)
	pocket_definition = pocket_finder.find_pocket(mode=pocket_detection["method"],
													receptor=prepared_receptor,
													ligand=reference_ligand if reference_ligand else None,
													radius=pocket_detection.get("radius", 10),
													manual_pocket=pocket_detection.get("manual_pocket", None),
													dogsitescorer_method=pocket_detection.get(
														"dogsitescorer_method", 'Volume'))

	results = {}

	for program in docking["docking_programs"]:
		printlog(f"Processing docking program: {program}")
		results[program] = {}

		# 1. Perform docking for each program
		output_sdf = w_dir / program.lower() / f"{program.lower()}_poses.sdf"
		if not output_sdf.exists():
			printlog(f"Running {program} docking...")
			docking_function = DOCKING_PROGRAMS[program](software_path=software)
			(w_dir / program.lower()).mkdir(exist_ok=True, parents=True)
			docking_results = docking_function.dock(prepared_library,
													prepared_receptor,
													pocket_definition,
													docking["exhaustiveness"],
													docking["n_poses"],
													n_cpus,
													"concurrent_process",
													output_sdf=output_sdf)

		if output_sdf.exists() and output_sdf.stat().st_size > 0:
			printlog(f"{program} docking completed. Results saved to {output_sdf}")
		else:
			printlog(f"ERROR: No results obtained from {program} docking")
			continue

		# 2. Apply postprocessing to each docking software separately
		postprocessed_sdf = w_dir / program.lower() / f"{program.lower()}_postprocessed.sdf"
		if not postprocessed_sdf.exists():
			postprocessed_poses = docking_postprocessing(output_sdf,
															prepared_receptor,
															post_docking["minimize_poses"],
															post_docking["bust_poses"],
															post_docking["strain_cutoff"],
															post_docking["clash_cutoff"],
															post_docking["classy_pose"],
															post_docking["classy_pose_model"],
															n_cpus,
															output_sdf=postprocessed_sdf)

		# 3. Apply pose selection to each set of postprocessed poses
		for method in pose_selection["pose_selection_method"]:
			if method.startswith("bestpose_") and method.split("_")[1] != program:
				printlog(
					f"Skipping {method} selection for {program} as it is not the best pose selection method for {program}"
				)
				continue

			selected_poses_sdf = w_dir / program.lower() / f"{program.lower()}_selected_poses_{method}.sdf"
			if not selected_poses_sdf.exists():
				selected_poses = select_poses(postprocessed_sdf,
												method,
												pose_selection["clustering_method"],
												pocket_definition,
												prepared_receptor,
												software,
												n_cpus,
												output_file=selected_poses_sdf)

			# 4. Apply rescoring to each set of selected poses
			rescored_file = w_dir / program.lower() / f"{program.lower()}_rescored_{method}.csv"
			#if not rescored_file.exists():
			rescored_poses = rescore_poses(prepared_receptor,
											pocket_definition,
											software,
											selected_poses_sdf,
											rescoring,
											n_cpus,
											output_file=rescored_file)

			# Store the results in the nested dictionary
			results[program][method] = rescored_file

	return results


def run_dockm8(config):
	printlog("Starting DockM8 run...")
	software = config["general"]["software"]
	decoy_generation = config["decoy_generation"]
	if decoy_generation["gen_decoys"]:
		decoy_library = generate_decoys(Path(decoy_generation.get("actives")),
										decoy_generation.get("n_decoys"),
										decoy_generation.get("decoy_model"),
										software)

		if config["general"]["mode"] == "single":
			# Run DockM8 on the decoy library
			printlog("Running DockM8 on the decoy library...")
			results = dockm8_decoys(software=software,
									receptor=config["receptor(s)"][0],
									prepare_proteins=config["protein_preparation"],
									ligand_preparation=config["ligand_preparation"],
									pocket_detection=config["pocket_detection"],
									reference_ligand=config["pocket_detection"]["reference_ligand(s)"][0]
									if config["pocket_detection"]["reference_ligand(s)"] else None,
									docking_library=decoy_library,
									docking=config["docking"],
									post_docking=config["post_docking"],
									pose_selection=config["pose_selection"],
									n_cpus=config["general"]["n_cpus"],
									rescoring=config["rescoring"],
									consensus=config["consensus"],
									threshold=config["threshold"])
			# Calculate performance metrics
			optimal_conditions = determine_optimal_conditions(
				Path(config["receptor(s)"][0]).parent / (Path(config["receptor(s)"][0]).stem + "_decoys"),
				results,
				decoy_library, [10, 5, 2, 1, 0.5])
			# Determine optimal conditions
			printlog(f"Optimal conditions: {optimal_conditions}")
			optimal_docking_config = {
				"docking_programs": [optimal_conditions['docking_program']
									],                                                               # Wrap in a list as docking_programs expects a list
				"exhaustiveness": config['docking']['exhaustiveness'],
				"n_poses": config['docking']['n_poses']}
			if optimal_conditions["selection_method"] == "bestpose":
				pass
			if "_" in optimal_conditions["selection_method"]:
				optimal_docking_config["docking_programs"] = list(optimal_conditions["selection_method"].split("_")[1])
			else:
				pass
			optimal_rescoring_functions = list(optimal_conditions["scoring"].split("_"))
			optimal_rescoring_functions = [func for func in optimal_rescoring_functions if func in RESCORING_FUNCTIONS]
			config["pose_selection"]["pose_selection_method"] = optimal_conditions["selection_method"]
			                                                                           # Save optimal conditions to a file
			with open(config["receptor(s)"][0].parent / "DeepCoy" / "optimal_conditions.txt", "w") as file:
				file.write(str(optimal_conditions))
			                                                                           # Run DockM8 on the docking library using the optimal conditions
			printlog("Running DockM8 on the docking library using optimal conditions...")
			print(optimal_conditions)
			dockm8(software=software,
					receptor=config["receptor(s)"][0],
					prepare_proteins=config["protein_preparation"],
					ligand_preparation=config["ligand_preparation"],
					pocket_detection=config["pocket_detection"],
					reference_ligand=config["pocket_detection"]["reference_ligand(s)"][0]
					if config["pocket_detection"]["reference_ligand(s)"] else None,
					docking_library=config["docking_library"],
					docking=optimal_docking_config,
					post_docking=config["post_docking"],
					pose_selection=config["pose_selection"],
					n_cpus=config["general"]["n_cpus"],
					rescoring=optimal_rescoring_functions,
					consensus=optimal_conditions["consensus"],
					threshold=config["threshold"])

			printlog("DockM8 has finished running in single mode...")
		if config["general"]["mode"] == "ensemble":
			# Generate target and reference ligand dictionnary
			receptor_dict = {}
			for i, receptor in enumerate(config["receptor(s)"]):
				if config["pocket_detection"]["reference_ligand(s)"] is None:
					receptor_dict[receptor] = None
				else:
					receptor_dict[receptor] = config["pocket_detection"]["reference_ligand(s)"][i]
			# Run DockM8 on the decoy library
			printlog("Running DockM8 on the decoy library...")
			results = dockm8_decoys(software=software,
									receptor=config["receptor(s)"][0],
									prepare_proteins=config["protein_preparation"],
									ligand_preparation=config["ligand_preparation"],
									pocket_detection=config["pocket_detection"],
									reference_ligand=config["pocket_detection"]["reference_ligand(s)"][0]
									if config["pocket_detection"]["reference_ligand(s)"] else None,
									docking_library=decoy_library,
									docking=config["docking"],
									post_docking=config["post_docking"],
									pose_selection=config["pose_selection"],
									n_cpus=config["general"]["n_cpus"],
									rescoring=config["rescoring"],
									consensus=config["consensus"],
									threshold=config["threshold"])
			printlog(f"Performance: {results}")
			# Calculate performance metrics
			performance = determine_optimal_conditions(
				Path(config["receptor(s)"][0]).parent / (Path(config["receptor(s)"][0]).stem + "_decoys"),
				results,
				decoy_library, [10, 5, 2, 1, 0.5])
			# Determine optimal conditions
			optimal_conditions = performance.sort_values(by="BEDROC", ascending=False).iloc[0].to_dict()
			optimal_docking_config = {
				"docking_programs": [optimal_conditions['docking_program']
									],                                                         # Wrap in a list as docking_programs expects a list
				"exhaustiveness": config['docking']['exhaustiveness'],
				"n_poses": config['docking']['n_poses']}
			if optimal_conditions["selection_method"] == "bestpose":
				pass
			if "_" in optimal_conditions["selection_method"]:
				optimal_docking_config["docking_programs"] = list(optimal_conditions["selection_method"].split("_")[1])
			else:
				pass
			optimal_rescoring_functions = list(optimal_conditions["scoring"].split("_"))
			optimal_rescoring_functions = [func for func in optimal_rescoring_functions if func in RESCORING_FUNCTIONS]
			config["pose_selection"]["pose_selection_method"] = optimal_conditions["selection_method"]
			                                                                     # Save optimal conditions to a file
			with open(config["receptor(s)"][0].parent / "DeepCoy" / "optimal_conditions.txt", "w") as file:
				file.write(str(optimal_conditions))
			                                                                     # Run DockM8 on the docking library using the optimal conditions
			printlog("Running DockM8 on the docking library using optimal conditions...")
			for receptor, ligand in receptor_dict.items():
				dockm8(software=software,
						receptor=receptor,
						prepare_proteins=config["protein_preparation"],
						ligand_preparation=config["ligand_preparation"],
						pocket_detection=config["pocket_detection"],
						reference_ligand=ligand,
						docking_library=Path(config["docking_library"]),
						docking=optimal_docking_config,
						post_docking=config["post_docking"],
						pose_selection=config["pose_selection"],
						n_cpus=config["general"]["n_cpus"],
						rescoring=optimal_rescoring_functions,
						consensus=optimal_conditions["consensus"],
						threshold=config["threshold"])

				printlog("DockM8 has finished running in ensemble mode...")
	else:
		if config["general"]["mode"] == "single":
			# Run DockM8 in single mode
			printlog("Running DockM8 in single mode...")
			dockm8(software=software,
				receptor=config["receptor(s)"][0],
				prepare_proteins=config["protein_preparation"],
				ligand_preparation=config["ligand_preparation"],
				pocket_detection=config["pocket_detection"],
				reference_ligand=config["pocket_detection"]["reference_ligand(s)"][0]
				if config["pocket_detection"]["reference_ligand(s)"] else None,
				docking_library=config["docking_library"],
				docking=config["docking"],
				post_docking=config["post_docking"],
				pose_selection=config["pose_selection"],
				n_cpus=config["general"]["n_cpus"],
				rescoring=config["rescoring"],
				consensus=config["consensus"],
				threshold=config["threshold"])

			printlog("DockM8 has finished running in single mode...")
		if config["general"]["mode"] == "ensemble":
			# Generate target and reference ligand dictionnary
			receptor_dict = {}
			for i, receptor in enumerate(config["receptor(s)"]):
				if config["pocket_detection"]["reference_ligand(s)"] is None:
					receptor_dict[receptor] = None
				else:
					receptor_dict[receptor] = config["pocket_detection"]["reference_ligand(s)"][i]
			# Run DockM8 in ensemble mode
			printlog("Running DockM8 in ensemble mode...")
			for receptor, ligand in receptor_dict.items():
				dockm8(software=software,
					receptor=receptor,
					prepare_proteins=config["protein_preparation"],
					ligand_preparation=config["ligand_preparation"],
					pocket_detection=config["pocket_detection"],
					reference_ligand=ligand,
					docking_library=config["docking_library"],
					docking=config["docking"],
					post_docking=config["post_docking"],
					pose_selection=config["pose_selection"],
					n_cpus=config["general"]["n_cpus"],
					rescoring=config["rescoring"],
					consensus=config["consensus"],
					threshold=config["threshold"])

			printlog("DockM8 has finished running in ensemble mode...")
	return


run_dockm8(config)
