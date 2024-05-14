# Import required libraries and scripts
import argparse
import os
import sys
import warnings
from pathlib import Path
import time

from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
dockm8_path = next(
    (p / "DockM8" for p in Path(__file__).resolve().parents if (p / "DockM8").is_dir()),
    None,
)
sys.path.append(str(dockm8_path))

# Import modules for docking, scoring, protein and ligand preparation, etc.
from scripts.utilities.config_parser import check_config
from scripts.clustering_functions import *
from scripts.consensus_methods import *
from scripts.docking.docking import dockm8_docking, concat_all_poses
from scripts.library_preparation.main import prepare_library
from scripts.docking_postprocessing.docking_postprocessing import docking_postprocessing
from scripts.performance_calculation import *
from scripts.pocket_finding.pocket_finding import pocket_finder
from scripts.postprocessing import *
from scripts.protein_preparation.protein_preparation import prepare_protein
from scripts.rescoring_functions import *
from scripts.utilities.utilities import printlog
from software.DeepCoy.generate_decoys import generate_decoys

# Suppress warnings to clean up output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize argument parser to handle command line arguments
parser = argparse.ArgumentParser(description="Parse required arguments")

parser.add_argument("--config", type=str, help="Path to the configuration file")

config = check_config(Path(parser.parse_args().config))

# Main function to manage docking process
def dockm8(
    software: Path,
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
    threshold: float,
):
    # Set working directory based on the receptor file
    w_dir = Path(receptor).parent / Path(receptor).stem
    print("The working directory has been set to:", w_dir)
    (w_dir).mkdir(exist_ok=True)

    # Prepare the protein for docking (e.g., adding hydrogens)
    prepared_receptor = prepare_protein(
        protein_file_or_code=receptor,
        output_dir=w_dir,
        select_best_chain=prepare_proteins["select_best_chain"],
        fix_nonstandard_residues=prepare_proteins["fix_nonstandard_residues"],
        fix_missing_residues=prepare_proteins["fix_missing_residues"],
        remove_hetero=prepare_proteins["remove_heteroatoms"],
        remove_water=prepare_proteins["remove_water"],
        add_missing_hydrogens_pH=prepare_proteins["add_hydrogens"],
        protonate=prepare_proteins["protonation"],
    )

    # Prepare the ligands for docking (e.g., adding hydrogens, generating conformers)

    if not (w_dir / "final_library.sdf").exists():
        prepare_library(
            input_sdf=docking_library,
            output_dir=w_dir,
            id_column="ID",
            protonation=ligand_preparation["protonation"],
            conformers=ligand_preparation["conformers"],
            software=software,
            n_cpus=n_cpus,
            n_conformers=ligand_preparation["n_conformers"]
        )

    # Determine the docking pocket
    pocket_definition = pocket_finder(
        mode=pocket_detection["method"],
        software=software,
        receptor=prepared_receptor,
        ligand=reference_ligand,
        radius=pocket_detection["radius"],
        manual_pocket=pocket_detection["manual_pocket"],
    )

    # Perform the docking operation
    if not (w_dir / "allposes.sdf").exists():
        dockm8_docking(
            w_dir=w_dir,
            protein_file=prepared_receptor,
            pocket_definition=pocket_definition,
            software=software,
            docking_programs=docking["docking_programs"],
            n_poses=docking["n_poses"],
            exhaustiveness=docking["exhaustiveness"],
            n_cpus=n_cpus,
        )

        # Concatenate all poses into a single file
        concat_all_poses(
            w_dir=w_dir,
            docking_programs=docking["docking_programs"],
            protein_file=prepared_receptor,
            bust_poses=docking["bust_poses"],
            n_cpus=n_cpus,
        )
    
    processed_poses = docking_postprocessing(
        input_sdf=w_dir / "allposes.sdf",
        output_path=w_dir / "allposes_processed.sdf",
        protein_file=prepared_receptor,
        bust_poses=post_docking['bust_poses'],
        strain_cutoff=post_docking['strain_cutoff'],
        clash_cutoff=post_docking['clash_cutoff'],
        n_cpus=n_cpus
    )

    # Load all poses from SDF file and perform clustering
    print("Loading all poses SDF file...")
    tic = time.perf_counter()
    all_poses = PandasTools.LoadSDF(
        str(processed_poses),
        idName="Pose ID",
        molColName="Molecule",
        includeFingerprints=False,
        
    )
    toc = time.perf_counter()
    print(f"Finished loading all poses SDF in {toc-tic:0.4f}!")

    # Select Poses
    pose_selection_methods = pose_selection["pose_selection_method"]
    for method in pose_selection_methods:
        if not os.path.isfile(w_dir / f"clustering/{method}_clustered.sdf"):
            select_poses(
                selection_method=method,
                clustering_method=pose_selection["clustering_method"],
                w_dir=w_dir,
                protein_file=receptor,
                pocket_definition=pocket_definition,
                software=software,
                all_poses=all_poses,
                n_cpus=n_cpus,
            )

    # Rescore poses for each selection method
    for method in pose_selection_methods:
        rescore_poses(
            w_dir=w_dir,
            protein_file=prepared_receptor,
            pocket_definition=pocket_definition,
            software=software,
            clustered_sdf=w_dir / "clustering" / f"{method}_clustered.sdf",
            functions=rescoring,
            n_cpus=n_cpus,
        )

    # Apply consensus methods to the poses
    for method in pose_selection_methods:
        apply_consensus_methods(
            w_dir=w_dir,
            selection_method=method,
            consensus_methods=consensus,
            rescoring_functions=rescoring,
            standardization_type="min_max",
        )
    return


def run_dockm8(config):
    printlog("Starting DockM8 run...")
    software = config["general"]["software"]
    decoy_generation = config["decoy_generation"]
    if decoy_generation["gen_decoys"]:
        decoy_library = generate_decoys(
            Path(decoy_generation.get("actives")),
            decoy_generation.get("n_decoys"),
            decoy_generation.get("decoy_model"),
            software,
        )
        if config["general"]["mode"] == "single":
            # Run DockM8 on the decoy library
            printlog("Running DockM8 on the decoy library...")
            dockm8(
                software=software,
                receptor=config["receptor(s)"][0],
                prepare_proteins=config["protein_preparation"],
                ligand_preparation=config["ligand_preparation"],
                pocket_detection=config["pocket_detection"],
                reference_ligand=config["pocket_detection"]["reference_ligand(s)"][0] if config["pocket_detection"]["reference_ligand(s)"] else None,
                docking_library=decoy_library,
                docking=config["docking"],
                post_docking=config["post_docking"],
                pose_selection=config["pose_selection"],
                n_cpus=config["general"]["n_cpus"],
                rescoring=config["rescoring"],
                consensus=config["consensus"],
                threshold=config["threshold"],
            )
            # Calculate performance metrics
            performance = calculate_performance(
                decoy_library.parent, decoy_library, [10, 5, 2, 1, 0.5]
            )
            # Determine optimal conditions
            optimal_conditions = (
                performance.sort_values(by="EF1", ascending=False).iloc[0].to_dict()
            )
            if optimal_conditions["clustering"] == "bestpose":
                pass
            if "_" in optimal_conditions["clustering"]:
                config["docking"]["docking_programs"] = list(
                    optimal_conditions["clustering"].split("_")[1]
                )
            else:
                pass
            optimal_rescoring_functions = list(optimal_conditions["scoring"].split("_"))
            config["pose_selection"]["pose_selection_method"] = optimal_conditions[
                "clustering"
            ]
            # Save optimal conditions to a file
            with open(
                config["receptor(s)"][0].parent / "DeepCoy" / "optimal_conditions.txt",
                "w",
            ) as file:
                file.write(str(optimal_conditions))
            # Run DockM8 on the docking library using the optimal conditions
            printlog("Running DockM8 on the docking library using optimal conditions...")
            dockm8(
                software=software,
                receptor=config["receptor(s)"][0],
                prepare_proteins=config["protein_preparation"],
                ligand_preparation=config["ligand_preparation"],
                pocket_detection=config["pocket_detection"],
                reference_ligand=config["pocket_detection"]["reference_ligand(s)"][0] if config["pocket_detection"]["reference_ligand(s)"] else None,
                docking_library=decoy_library,
                docking=config["docking"],
                post_docking=config["post_docking"],
                pose_selection=config["pose_selection"],
                n_cpus=config["general"]["n_cpus"],
                rescoring=optimal_rescoring_functions,
                consensus=optimal_conditions["consensus"],
                threshold=config["threshold"],
            )
            printlog("DockM8 has finished running in single mode...")
        if config["general"]["mode"] == "ensemble":
            # Generate target and reference ligand dictionnary
            receptor_dict = {}
            for i, receptor in enumerate(config["receptor(s)"]):
                if config["pocket_detection"]["reference_ligand(s)"] is None:
                    receptor_dict[receptor] = None
                else:
                    receptor_dict[receptor] = config["pocket_detection"][
                        "reference_ligand(s)"
                    ][i]
            # Run DockM8 on the decoy library
            printlog("Running DockM8 on the decoy library...")
            dockm8(
                software=software,
                receptor=config["receptor(s)"][0],
                prepare_proteins=config["protein_preparation"],
                ligand_preparation=config["ligand_preparation"],
                pocket_detection=config["pocket_detection"],
                reference_ligand=config["pocket_detection"]["reference_ligand(s)"][0] if config["pocket_detection"]["reference_ligand(s)"] else None,
                docking_library=decoy_library,
                docking=config["docking"],
                post_docking=config["post_docking"],
                pose_selection=config["pose_selection"],
                n_cpus=config["general"]["n_cpus"],
                rescoring=config["rescoring"],
                consensus=config["consensus"],
                threshold=config["threshold"],
            )
            # Calculate performance metrics
            performance = calculate_performance(
                decoy_library.parent, decoy_library, [10, 5, 2, 1, 0.5]
            )
            # Determine optimal conditions
            optimal_conditions = (
                performance.sort_values(by="EF1", ascending=False).iloc[0].to_dict()
            )
            if optimal_conditions["clustering"] == "bestpose":
                pass
            if "_" in optimal_conditions["clustering"]:
                config["docking"]["docking_programs"] = list(
                    optimal_conditions["clustering"].split("_")[1]
                )
            else:
                pass
            optimal_rescoring_functions = list(optimal_conditions["scoring"].split("_"))
            config["pose_selection"]["pose_selection_method"] = optimal_conditions[
                "clustering"
            ]
            # Save optimal conditions to a file
            with open(
                config["receptor(s)"][0].parent / "DeepCoy" / "optimal_conditions.txt",
                "w",
            ) as file:
                file.write(str(optimal_conditions))
            # Run DockM8 on the docking library using the optimal conditions
            printlog("Running DockM8 on the docking library using optimal conditions...")
            for receptor, ligand in receptor_dict.items():
                dockm8(
                    software=software,
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
                    threshold=config["threshold"],
                )
                printlog("DockM8 has finished running in ensemble mode...")
    else:
        if config["general"]["mode"] == "single":
            # Run DockM8 in single mode
            printlog("Running DockM8 in single mode...")
            dockm8(
                software=software,
                receptor=config["receptor(s)"][0],
                prepare_proteins=config["protein_preparation"],
                ligand_preparation=config["ligand_preparation"],
                pocket_detection=config["pocket_detection"],
                reference_ligand=config["pocket_detection"]["reference_ligand(s)"][0] if config["pocket_detection"]["reference_ligand(s)"] else None,
                docking_library=config["docking_library"],
                docking=config["docking"],
                post_docking=config["post_docking"],
                pose_selection=config["pose_selection"],
                n_cpus=config["general"]["n_cpus"],
                rescoring=config["rescoring"],
                consensus=config["consensus"],
                threshold=config["threshold"],
            )
            printlog("DockM8 has finished running in single mode...")
        if config["general"]["mode"] == "ensemble":
            # Generate target and reference ligand dictionnary
            receptor_dict = {}
            for i, receptor in enumerate(config["receptor(s)"]):
                if config["pocket_detection"]["reference_ligand(s)"] is None:
                    receptor_dict[receptor] = None
                else:
                    receptor_dict[receptor] = config["pocket_detection"][
                        "reference_ligand(s)"
                    ][i]
            # Run DockM8 in ensemble mode
            printlog("Running DockM8 in ensemble mode...")
            for receptor, ligand in receptor_dict.items():
                dockm8(
                    software=software,
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
                    threshold=config["threshold"],
                )
            printlog("DockM8 has finished running in ensemble mode...")
    return

run_dockm8(config)
