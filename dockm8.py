# Import required libraries and scripts
import argparse
import os
import sys
import warnings
from pathlib import Path
import time

from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

# Import modules for docking, scoring, protein and ligand preparation, etc.
from scripts.utilities.config_parser import check_config
from scripts.pose_selection.pose_selection import select_poses
from scripts.consensus.consensus import apply_consensus_methods
from scripts.docking.docking import DOCKING_PROGRAMS, concat_all_poses
from scripts.library_preparation.main import prepare_library
from scripts.docking_postprocessing.docking_postprocessing import docking_postprocessing
from scripts.performance_calculation import *
from scripts.pocket_finding.pocket_finding import pocket_finder
from scripts.postprocessing import *
from scripts.protein_preparation.protein_preparation import prepare_protein
from scripts.rescoring.rescoring import rescore_poses, RESCORING_FUNCTIONS
from scripts.utilities.logging import printlog
from software.DeepCoy.generate_decoys import generate_decoys

# Suppress warnings to clean up output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
	printlog("The working directory has been set to:", w_dir)
	(w_dir).mkdir(exist_ok=True)

	# Prepare the protein for docking
	prepared_receptor = prepare_protein(protein_file_or_code=receptor, output_dir=w_dir, **prepare_proteins)

	# Prepare the ligands for docking
	prepared_library = prepare_library(input_sdf=docking_library,
										id_column="ID",
										**ligand_preparation,
										software=software,
										n_cpus=n_cpus)

	# Determine the docking pocket
	pocket_definition = pocket_finder(mode=pocket_detection["method"],
										software=software,
										receptor=prepared_receptor,
										ligand=reference_ligand,
										**pocket_detection)

	# Perform the docking operation
	(w_dir / "docking").mkdir(exist_ok=True)
	for program in docking["docking_programs"]:
		docking_class = DOCKING_PROGRAMS[program]
		docking_function = docking_class(software)
		
		output_sdf = w_dir / f"docking/{program.lower()}_poses.sdf"
		docking_function.dock(
			library=prepared_library,
			protein_file=prepared_receptor,
			pocket_definition=pocket_definition,
			exhaustiveness=docking["exhaustiveness"],
			n_poses=docking["n_poses"],
			n_cpus=n_cpus,
			output_sdf=output_sdf
		)

	# Combine all docking results into a single file
	concat_all_poses(w_dir / "docking/allposes.sdf", docking["docking_programs"], n_cpus)

	# Combine all poses
	combined_poses = pd.concat(all_poses, ignore_index=True)

	# Save combined poses
	output_file = w_dir / "allposes.sdf"
	PandasTools.WriteSDF(combined_poses,
							str(output_file),
							molColName="Molecule",
							idName="Pose ID",
							properties=list(combined_poses.columns))

	processed_poses = docking_postprocessing(input_sdf=output_file,
												output_path=w_dir / "allposes_processed.sdf",
												protein_file=prepared_receptor,
												**post_docking,
												n_cpus=n_cpus)

	# Load all poses from SDF file and perform clustering
	printlog("Loading all poses SDF file...")
	tic = time.perf_counter()
	all_poses = PandasTools.LoadSDF(str(processed_poses),
									idName="Pose ID",
									molColName="Molecule",
									includeFingerprints=False)
	toc = time.perf_counter()
	printlog(f"Finished loading all poses SDF in {toc-tic:0.4f}!")

	# Select Poses
	pose_selection_methods = pose_selection["pose_selection_method"]
	for method in pose_selection_methods:
		if not os.path.isfile(w_dir / f"clustering/{method}_clustered.sdf"):
			select_poses(selection_method=method,
							clustering_method=pose_selection["clustering_method"],
							w_dir=w_dir,
							protein_file=receptor,
							software=software,
							all_poses=all_poses,
							n_cpus=n_cpus)

	# Rescore poses for each selection method
	for method in pose_selection_methods:
		rescore_poses(w_dir=w_dir,
						protein_file=prepared_receptor,
						pocket_definition=pocket_definition,
						software=software,
						clustered_sdf=w_dir / "clustering" / f"{method}_clustered.sdf",
						functions=rescoring,
						n_cpus=n_cpus)

	# Apply consensus methods to the poses
	for method in pose_selection_methods:
		apply_consensus_methods(w_dir=w_dir,
								selection_method=method,
								consensus_methods=consensus,
								rescoring_functions=rescoring,
								standardization_type="min_max")

	return


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
			dockm8(software=software,
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
			performance = calculate_performance(decoy_library.parent, decoy_library, [10, 5, 2, 1, 0.5])
			# Determine optimal conditions
			optimal_conditions = performance.sort_values(by="EF1", ascending=False).iloc[0].to_dict()
			if optimal_conditions["clustering"] == "bestpose":
				pass
			if "_" in optimal_conditions["clustering"]:
				config["docking"]["docking_programs"] = list(optimal_conditions["clustering"].split("_")[1])
			else:
				pass
			optimal_rescoring_functions = list(optimal_conditions["scoring"].split("_"))
			optimal_rescoring_functions = [func for func in optimal_rescoring_functions if func in RESCORING_FUNCTIONS]
			config["pose_selection"]["pose_selection_method"] = optimal_conditions["clustering"]
			# Save optimal conditions to a file
			with open(config["receptor(s)"][0].parent / "DeepCoy" / "optimal_conditions.txt", "w") as file:
				file.write(str(optimal_conditions))
			# Run DockM8 on the docking library using the optimal conditions
			printlog("Running DockM8 on the docking library using optimal conditions...")
			dockm8(software=software,
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
			dockm8(software=software,
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
			performance = calculate_performance(decoy_library.parent, decoy_library, [10, 5, 2, 1, 0.5])
			# Determine optimal conditions
			optimal_conditions = performance.sort_values(by="EF1", ascending=False).iloc[0].to_dict()
			if optimal_conditions["clustering"] == "bestpose":
				pass
			if "_" in optimal_conditions["clustering"]:
				config["docking"]["docking_programs"] = list(optimal_conditions["clustering"].split("_")[1])
			else:
				pass
			optimal_rescoring_functions = list(optimal_conditions["scoring"].split("_"))
			optimal_rescoring_functions = [func for func in optimal_rescoring_functions if func in RESCORING_FUNCTIONS]
			config["pose_selection"]["pose_selection_method"] = optimal_conditions["clustering"]
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
					docking_library=decoy_library,
					docking=config["docking"],
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
