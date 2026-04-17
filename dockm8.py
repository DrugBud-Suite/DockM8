#!/usr/bin/env python3
"""DockM8: An integrated pipeline for molecular docking, scoring, and consensus analysis."""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Import required functions from refactored modules
from scripts.consensus.consensus import (
    _METHODS as CONSENSUS_METHODS,
    apply_consensus_scoring,
)
from scripts.docking.docking import DOCKING_PROGRAMS
from scripts.library_preparation.library_preparation import prepare_library
from scripts.performance.analyzer import run_consensus_analysis
from scripts.pocket_finding.pocket_finder import (
    VALID_METHODS as POCKET_METHODS,
    find_pocket,
)
from scripts.pose_selection.pose_selection import select_poses
from scripts.pose_selection.posebusters import bust_poses
from scripts.protein_preparation.protein_preparation import prepare_protein
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS, rescore_poses
from scripts.utilities.fast_sdf_loader import fast_load_sdf
from scripts.utilities.fast_sdf_writer import fast_write_sdf
from scripts.utilities.logging import printlog
from software.DeepCoy.generate_decoys import generate_decoys

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="DockM8: Molecular docking, scoring, and consensus analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core arguments
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "--software",
        required=True,
        type=str,
        help="Path to the software folder containing all external tools"
    )
    required.add_argument(
        "--receptor",
        required=True,
        type=str,
        nargs="+",
        help="Path to protein file(s). Multiple files trigger ensemble mode."
    )
    required.add_argument(
        "--docking_library",
        required=True,
        type=str,
        help="Path to the docking library .sdf file"
    )
    required.add_argument(
        "--id_column",
        required=True,
        type=str,
        help="Column name for the unique identifier in the library"
    )
    required.add_argument(
        "--docking_program",
        required=True,
        nargs="+",
        type=str,
        choices=list(DOCKING_PROGRAMS.keys()),
        help="Docking program to use"
    )

    required.add_argument(
        "--mode",
        type=str,
        choices=["single", "ensemble"],
        default="single",
        help="Mode of operation: single receptor or ensemble of multiple receptors"
    )
    
    # Global settings
    global_args = parser.add_argument_group("Global settings")
    global_args.add_argument(
        "--n_cpus",
        type=int,
        default=max(1, int(os.cpu_count() * 0.9)),
        help="Number of CPUs to use for all parallel operations"
    )

    # Pocket finding options
    pocket = parser.add_argument_group("Pocket finding options")
    pocket.add_argument(
        "--pocket_method",
        type=str,
        choices=POCKET_METHODS,
        default="Reference",
        help="Method to use for pocket determination"
    )
    pocket.add_argument(
        "--reference_ligand",
        type=str,
        nargs="+",
        help="Path to reference ligand file(s) for pocket definition. "
             "Multiple files should correspond to --receptor order."
    )
    pocket.add_argument(
        "--manual_pocket",
        type=str,
        help="Manual pocket definition in format 'center:x,y,z*size:x,y,z'"
    )
    pocket.add_argument(
        "--dogsitescorer_method",
        type=str,
        default="Volume",
        choices=["Volume", "Druggability_Score", "Surface", "Depth"],
        help="DogSiteScorer metric to select binding site"
    )
    pocket.add_argument(
        "--pocket_radius",
        type=int,
        default=10,
        help="Radius for pocket finding (when applicable)"
    )

    # Protein preparation options
    protein = parser.add_argument_group("Protein preparation options")
    protein.add_argument(
        "--prepare_protein",
        type=str2bool,
        default=True,
        help="Whether to prepare the protein"
    )
    protein.add_argument(
        "--protein_protonation",
        type=str,
        default="protoss",
        choices=["protoss", "pdbfixer", "none"],
        help="Method for protein protonation"
    )
    protein.add_argument(
        "--fix_nonstandard_residues",
        type=str2bool,
        default=True,
        help="Fix nonstandard residues during protein preparation"
    )
    protein.add_argument(
        "--fix_missing_residues",
        type=str2bool,
        default=True,
        help="Fix missing residues during protein preparation"
    )
    protein.add_argument(
        "--remove_hetero",
        type=str2bool,
        default=True,
        help="Remove heteroatoms during protein preparation"
    )
    protein.add_argument(
        "--remove_water",
        type=str2bool,
        default=True,
        help="Remove water molecules during protein preparation"
    )

    # Ligand preparation options
    ligand = parser.add_argument_group("Ligand preparation options")
    ligand.add_argument(
        "--ligand_protonation",
        required=True,
        type=str,
        choices=["GypsumDL", "None"],
        help="Method to use for ligand protonation"
    )
    ligand.add_argument(
        "--conformers",
        default="MMFF",
        type=str,
        choices=["UFF", "MMFF", "GypsumDL"],
        help="Method to use for conformer generation"
    )
    ligand.add_argument(
        "--standardize_ids",
        type=str2bool,
        default=True,
        help="Standardize compound IDs (WARNING: setting to False may cause errors with numeric-only IDs or IDs containing special characters)"
    )
    ligand.add_argument(
        "--remove_salts",
        type=str2bool,
        default=True,
        help="Remove salts during ligand preparation"
    )
    ligand.add_argument(
        "--min_ph",
        type=float,
        default=6.4,
        help="Minimum pH for ligand protonation (GypsumDL)"
    )
    ligand.add_argument(
        "--max_ph",
        type=float,
        default=8.4,
        help="Maximum pH for ligand protonation (GypsumDL)"
    )
    ligand.add_argument(
        "--pka_precision",
        type=float,
        default=1.0,
        help="pKa precision for ligand protonation (GypsumDL)"
    )

    # Docking options
    docking = parser.add_argument_group("Docking options")
    docking.add_argument(
        "--n_poses",
        default=10,
        type=int,
        help="Number of poses to generate per molecule"
    )
    docking.add_argument(
        "--exhaustiveness",
        default=8,
        type=int,
        help="Exhaustiveness parameter for docking programs"
    )

    docking.add_argument(
        "--bust_poses",
        type=str2bool,
        default=False,
        help="Apply PoseBusters to filter out low-quality poses (may significantly increase runtime)"
    )

    # Pose selection and scoring options
    scoring = parser.add_argument_group("Pose selection and scoring options")
    selection_choices = ["bestpose"] + list(RESCORING_FUNCTIONS.keys())
    scoring.add_argument(
        "--selection_method",
        type=str,
        nargs="+",
        default="bestpose",
        choices=selection_choices,
        help="Method to use for pose selection"
    )
    scoring.add_argument(
        "--rescoring_functions",
        type=str,
        nargs="+",
        choices=list(RESCORING_FUNCTIONS.keys()),
        default=["GNINA-Affinity", "CNN-Score"],
        help="Rescoring functions to use"
    )
    scoring.add_argument(
        "--consensus_method",
        type=str,
        choices=list(CONSENSUS_METHODS.keys()),
        default="rbr",
        help="Consensus method to use"
    )
    scoring.add_argument(
        "--ensemble_top_percent",
        type=float,
        default=1.0,
        help="Report compounds in the top X percent across all structures when using ensemble mode"
    )

    # Decoy generation options
    decoys = parser.add_argument_group("Decoy generation options")
    decoys.add_argument(
        "--generate_decoys",
        type=str2bool,
        default=False,
        help="Generate decoys for performance assessment"
    )
    decoys.add_argument(
        "--actives",
        type=str,
        help="Path to active compounds SDF file for decoy generation"
    )
    decoys.add_argument(
        "--decoy_model",
        type=str,
        choices=["DUDE", "DEKOIS", "DUDE_P"],
        default="DUDE",
        help="Model to use for decoy generation"
    )
    decoys.add_argument(
        "--n_decoys",
        type=int,
        default=20,
        help="Number of decoys to generate per active"
    )
    decoys.add_argument(
        "--performance_thresholds",
        type=float,
        nargs="+",
        default=[1, 2, 5],
        help="Thresholds for performance metrics calculation"
    )

    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args, parser)
    
    return args


def validate_arguments(args, parser):
    """Validate command line arguments and their combinations."""
    ensemble_mode = args.mode.lower() == "ensemble"
    if ensemble_mode and len(args.receptor) < 2:
        parser.error("Ensemble mode requires at least two receptor files")

    # Validate pocket finding requirements
    if args.pocket_method in ["Reference", "RoG"] and not args.reference_ligand:
        parser.error(f"Must specify --reference_ligand when --pocket_method is {args.pocket_method}")
    
    if args.pocket_method == "Manual" and not args.manual_pocket:
        parser.error("Must specify --manual_pocket when --pocket_method is Manual")
    
    # Validate decoy generation requirements
    if args.generate_decoys and not args.actives:
        parser.error("Must specify --actives when --generate_decoys is True")
    
    # Validate reference ligands in ensemble mode (multiple receptors)
    if len(args.receptor) > 1 and args.reference_ligand and len(args.receptor) != len(args.reference_ligand):
        parser.error("Number of --reference_ligand files must match number of --receptor files in ensemble mode")
    
    # Decoy mode is not supported with ensemble mode
    if args.generate_decoys and len(args.receptor) > 1:
        parser.error("Decoy generation mode is not supported when using ensemble mode (multiple receptors)")
    
    # Warning about ID standardization
    if not args.standardize_ids:
        printlog("WARNING: ID standardization is disabled. This may cause errors if IDs are numeric-only or contain special characters.")
    
    # Handle mode-specific argument validation
    if not args.generate_decoys:
        # In regular mode, ensure single values for key parameters
        if isinstance(args.docking_program, list) and len(args.docking_program) > 1:
            parser.error("Multiple docking programs can only be specified in decoy generation mode. Use --generate_decoys=True to enable decoy mode.")
        
        if isinstance(args.selection_method, list) and len(args.selection_method) > 1:
            parser.error("Multiple selection methods can only be specified in decoy generation mode. Use --generate_decoys=True to enable decoy mode.")
            
        # Convert to single values if they're lists with one element
        if isinstance(args.docking_program, list):
            args.docking_program = args.docking_program[0]
        if isinstance(args.selection_method, list):
            args.selection_method = args.selection_method[0]
    else:
        # Make sure arguments are lists even if only one value was provided
        if not isinstance(args.docking_program, list):
            args.docking_program = [args.docking_program]
        if not isinstance(args.selection_method, list):
            args.selection_method = [args.selection_method]
        if not isinstance(args.rescoring_functions, list):
            args.rescoring_functions = [args.rescoring_functions]
        
        # Log the search space to inform the user
        printlog("Decoy optimization search space:")
        printlog(f"  Docking programs: {', '.join(args.docking_program)}")
        printlog(f"  Selection methods: {', '.join(args.selection_method)}")
        printlog(f"  Rescoring functions: {', '.join(args.rescoring_functions)}")


def str2bool(value):
    """Convert string representation of boolean to actual boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_single_mode_paths(working_dir):
    """Get paths for single receptor mode (flat structure)."""
    return {
        'receptor_file': working_dir / "prepared_receptor.pdb",
        'pocket_file': working_dir / "pocket_definition.json",
        'library_dir': working_dir / "library",
        'docking_dir': working_dir / "docking",
        'selection_dir': working_dir / "selection",
        'rescoring_dir': working_dir / "rescoring",
        'consensus_dir': working_dir / "consensus",
        'final_results_dir': working_dir / "final_results"
    }


def get_ensemble_mode_paths(working_dir, receptor_name):
    """Get paths for ensemble mode (receptor subdirectories)."""
    receptor_dir = working_dir / f"receptor_{receptor_name}"
    return {
        'receptor_dir': receptor_dir,
        'receptor_file': receptor_dir / "prepared_receptor.pdb",
        'pocket_file': receptor_dir / "pocket_definition.json",
        'library_dir': working_dir / "library",  # Shared
        'docking_dir': receptor_dir / "docking",
        'selection_dir': receptor_dir / "selection",
        'rescoring_dir': receptor_dir / "rescoring",
        'consensus_dir': receptor_dir / "consensus",
        'final_results_dir': working_dir / "final_results"  # Shared
    }


def get_decoy_mode_paths(working_dir):
    """Get paths for decoy optimization mode."""
    paths = get_single_mode_paths(working_dir)
    paths['decoy_optimization_dir'] = working_dir / "decoy_optimization"
    return paths


def determine_workflow_mode(args):
    """Determine the workflow mode based on arguments."""
    if args.generate_decoys:
        return "decoy"
    elif args.mode.lower() == "ensemble" and len(args.receptor) > 1:
        return "ensemble"
    else:
        return "single"


def get_workflow_paths(working_dir, mode, receptor_name=None):
    """Get appropriate paths based on workflow mode."""
    if mode == "ensemble":
        if receptor_name is None:
            raise ValueError("receptor_name required for ensemble mode")
        return get_ensemble_mode_paths(working_dir, receptor_name)
    elif mode == "decoy":
        return get_decoy_mode_paths(working_dir)
    else:  # single mode
        return get_single_mode_paths(working_dir)


def setup_working_directory(library_path):
    """Create and return working directory for the pipeline."""
    library_name = Path(library_path).stem
    base_dir = Path(library_path).parent
    working_dir = base_dir / f"dockm8_{library_name}"
    working_dir.mkdir(parents=True, exist_ok=True)
    printlog(f"Results will be saved to {working_dir}")
    return working_dir


def prepare_compound_library(library_path, args, software_path, library_dir):
    """Prepare compound library for docking."""
    library_dir.mkdir(parents=True, exist_ok=True)
    prepared_library_filename = f"prepared_{Path(library_path).name}"
    prepared_library_path = library_dir / prepared_library_filename
    
    if prepared_library_path.exists() and prepared_library_path.stat().st_size > 0:
        printlog(f"Using existing prepared library: {prepared_library_path}")
        return prepared_library_path
    
    printlog(f"Preparing compound library: {library_path}")
    try:
        prepare_library(
            input_data=Path(library_path),
            protonation=args.ligand_protonation,
            conformers=args.conformers,
            software=software_path,
            n_cpus=args.n_cpus,
            standardize_ids=args.standardize_ids,
            remove_salts=args.remove_salts,
            min_ph=args.min_ph,
            max_ph=args.max_ph,
            pka_precision=args.pka_precision,
            output_sdf=prepared_library_path
        )
        printlog(f"Library preparation complete: {prepared_library_path}")
        return prepared_library_path
    except Exception as e:
        printlog(f"Error preparing library: {e}")
        raise


def prepare_receptor_structure(receptor_path, args, paths):
    """Prepare receptor structure for docking."""
    prepared_receptor_path = paths['receptor_file']
    
    if prepared_receptor_path.exists() and prepared_receptor_path.stat().st_size > 0:
        printlog(f"Using existing prepared receptor: {prepared_receptor_path}")
        return prepared_receptor_path
    
    if not args.prepare_protein:
        printlog(f"Protein preparation skipped, using raw input: {receptor_path}")
        return Path(receptor_path)
    
    printlog(f"Preparing protein structure: {receptor_path}")
    try:
        # Create receptor directory if it doesn't exist (for ensemble mode)
        if 'receptor_dir' in paths:
            paths['receptor_dir'].mkdir(parents=True, exist_ok=True)
        
        return prepare_protein(
            protein_file_or_code=receptor_path,
            output_dir=prepared_receptor_path.parent,
            fix_nonstandard_residues=args.fix_nonstandard_residues,
            fix_missing_residues=args.fix_missing_residues,
            protonation_method=args.protein_protonation,
            remove_hetero=args.remove_hetero,
            remove_water=args.remove_water,
        )
    except Exception as e:
        printlog(f"Error preparing protein: {e}")
        raise


def determine_binding_pocket(prepared_receptor, args, paths, ref_ligand=None):
    """Determine binding pocket for docking."""
    pocket_file = paths['pocket_file']
    
    if pocket_file.exists():
        printlog(f"Using existing pocket definition: {pocket_file}")
        with open(pocket_file) as f:
            import json
            return json.load(f)
    
    printlog("Determining docking pocket")
    try:
        if args.pocket_method == "Manual":
            pocket_definition = find_pocket(
                mode="Manual",
                receptor=prepared_receptor,
                manual_pocket=args.manual_pocket
            )
        elif args.pocket_method == "Dogsitescorer":
            pocket_definition = find_pocket(
                mode="Dogsitescorer",
                receptor=prepared_receptor,
                dogsitescorer_method=args.dogsitescorer_method
            )
        else:  # Reference or RoG
            if not ref_ligand:
                raise ValueError(f"Reference ligand required for {args.pocket_method} pocket finding")
            
            pocket_definition = find_pocket(
                mode=args.pocket_method,
                receptor=prepared_receptor,
                ligand=ref_ligand,
                radius=args.pocket_radius
            )
        
        # Save pocket definition for future runs
        with open(pocket_file, 'w') as f:
            import json
            json.dump(pocket_definition, f)
        
        printlog(f"Pocket definition: {pocket_definition}")
        return pocket_definition
    except Exception as e:
        printlog(f"Error determining pocket: {e}")
        raise


def perform_docking(prepared_library, prepared_receptor, pocket_definition,
                   docking_program, args, paths, software_path):
    """Perform molecular docking."""
    docking_dir = paths['docking_dir']
    docking_dir.mkdir(parents=True, exist_ok=True)
    docking_output_path = docking_dir / f"{docking_program.lower()}_docked.sdf"
    
    if docking_output_path.exists() and docking_output_path.stat().st_size > 0:
        printlog(f"Using existing docking results: {docking_output_path}")
        return docking_output_path
    
    printlog(f"Running docking with {docking_program}")
    try:
        docking_class = DOCKING_PROGRAMS[docking_program]
        docking_instance = docking_class(software_path)
        
        docking_output = docking_instance.dock(
            library=prepared_library,
            protein_file=prepared_receptor,
            pocket_definition=pocket_definition,
            exhaustiveness=args.exhaustiveness,
            n_poses=args.n_poses,
            n_cpus=args.n_cpus,
            output_sdf=docking_output_path
        )
        
        printlog(f"Docking complete: {docking_output}")
        return docking_output
    except Exception as e:
        printlog(f"Error during docking: {e}")
        raise


def apply_posebusters(docking_output, prepared_receptor, args, paths):
    """Apply PoseBusters to filter low-quality poses."""
    if not args.bust_poses:
        return docking_output
    
    docking_program = Path(docking_output).stem.split('_')[0]
    busted_poses_path = paths['docking_dir'] / f"{docking_program}_busted.sdf"
    
    if busted_poses_path.exists() and busted_poses_path.stat().st_size > 0:
        printlog(f"Using existing PoseBusters results: {busted_poses_path}")
        return busted_poses_path
    
    printlog("Applying PoseBusters to filter low-quality poses")
    try:
        dockm8_dir = Path(__file__).resolve().parent
        config_path = dockm8_dir / "scripts" / "pose_selection" / "posebusters_config.yml"
        
        filtered_output = bust_poses(
            poses_path=docking_output,
            protein_file=prepared_receptor,
            config_path=config_path,
            output_path=busted_poses_path,
            n_cpus=args.n_cpus
        )
        
        printlog(f"PoseBusters filtering complete: {filtered_output}")
        return filtered_output
    except Exception as e:
        printlog(f"WARNING: PoseBusters filtering failed: {e}")
        printlog("Continuing with original poses")
        return docking_output


def select_docked_poses(docking_output, prepared_receptor, pocket_definition,
                       selection_method, args, paths, software_path):
    """Select poses using specified method."""
    selection_dir = paths['selection_dir']
    selection_dir.mkdir(parents=True, exist_ok=True)
    selected_poses_path = selection_dir / f"{selection_method}_selected.sdf"
    
    if selected_poses_path.exists() and selected_poses_path.stat().st_size > 0:
        printlog(f"Using existing pose selection: {selected_poses_path}")
        return selected_poses_path
    
    printlog(f"Selecting poses using method: {selection_method}")
    try:
        select_poses(
            selection_method=selection_method,
            poses_input=docking_output,
            protein_file=prepared_receptor,
            pocket_definition=pocket_definition,
            software=software_path,
            output_file=selected_poses_path,
            n_cpus=args.n_cpus
        )
        
        printlog(f"Pose selection complete: {selected_poses_path}")
        return selected_poses_path
    except Exception as e:
        printlog(f"Error selecting poses: {e}")
        raise


def rescore_selected_poses(selected_poses, prepared_receptor, pocket_definition,
                          rescoring_functions, args, paths, software_path):
    """Rescore selected poses with specified functions."""
    rescoring_dir = paths['rescoring_dir']
    rescoring_dir.mkdir(parents=True, exist_ok=True)
    rescored_poses_path = rescoring_dir / "rescored_poses.sdf"
    
    if rescored_poses_path.exists() and rescored_poses_path.stat().st_size > 0:
        printlog(f"Using existing rescoring results: {rescored_poses_path}")
        return rescored_poses_path
    
    printlog(f"Rescoring poses with functions: {', '.join(rescoring_functions)}")
    try:
        rescore_poses(
            protein_file=prepared_receptor,
            pocket_definition=pocket_definition,
            software=software_path,
            poses=selected_poses,
            functions=rescoring_functions,
            n_cpus=args.n_cpus,
            output_file=rescored_poses_path
        )
        
        printlog(f"Rescoring complete: {rescored_poses_path}")
        return rescored_poses_path
    except Exception as e:
        printlog(f"Error rescoring poses: {e}")
        raise


def calculate_consensus(rescored_poses_csv, consensus_method, rescoring_functions,
                           id_column, paths):
    """Apply consensus scoring to rescored poses."""
    # Skip consensus for single scoring function
    if len(rescoring_functions) <= 1 or consensus_method.lower() == "none":
        printlog("Using single scoring function, skipping consensus")
        return rescored_poses_csv
    
    consensus_dir = paths['consensus_dir']
    consensus_dir.mkdir(parents=True, exist_ok=True)
    consensus_output_path = consensus_dir / f"{consensus_method}_consensus.csv"
    
    if consensus_output_path.exists() and consensus_output_path.stat().st_size > 0:
        printlog(f"Using existing consensus results: {consensus_output_path}")
        return consensus_output_path
    
    printlog(f"Applying consensus method: {consensus_method}")
    try:
        consensus_result = apply_consensus_scoring(
            data=rescored_poses_csv,
            method=consensus_method,
            columns=rescoring_functions,
            id_column="ID",
            output=consensus_output_path,
            normalize=True
        )
        
        printlog(f"Consensus scoring complete: {consensus_result}")
        return consensus_result
    except Exception as e:
        printlog(f"Error applying consensus: {e}")
        raise


def generate_decoy_library(actives_path, args, software_path, decoys_dir):
    """Generate decoy compounds for active molecules."""
    decoy_library = decoys_dir / f"{Path(actives_path).stem}_decoys.sdf"
    
    if decoy_library.exists() and decoy_library.stat().st_size > 0:
        printlog(f"Using existing decoy library: {decoy_library}")
        return decoy_library
    
    printlog(f"Generating decoys using {args.decoy_model} model")
    try:
        decoy_result = generate_decoys(
            input_sdf=Path(actives_path),
            n_decoys=args.n_decoys,
            model=args.decoy_model,
            software=software_path
        )
        
        printlog(f"Decoy generation complete: {decoy_result}")
        return decoy_result
    except Exception as e:
        printlog(f"Error generating decoys: {e}")
        raise


def create_activity_data(decoy_library, n_cpus, performance_dir):
    """Create activity data CSV from decoy library."""
    activity_data_path = performance_dir / "activity_data.csv"
    
    if activity_data_path.exists() and activity_data_path.stat().st_size > 0:
        printlog(f"Using existing activity data: {activity_data_path}")
        return activity_data_path
    
    printlog("Creating activity data from decoy library")
    try:
        activity_df = fast_load_sdf(
            decoy_library,
            molColName=None,
            idName="ID",
            batch_size=100,
            n_cpus=n_cpus
        )
        activity_df = activity_df[["ID", "Activity"]]
        activity_df.to_csv(activity_data_path, index=False)
        
        printlog(f"Activity data created: {activity_data_path}")
        return activity_data_path
    except Exception as e:
        printlog(f"Error creating activity data: {e}")
        raise


def analyze_performance(scoring_data_path, activity_data_path, thresholds,
                       n_cpus, docking_program, selection_method, performance_dir):
    """Analyze performance of scoring combinations."""
    analysis_results_path = performance_dir / f"{docking_program}_{selection_method}_performance.csv"
    
    if analysis_results_path.exists() and analysis_results_path.stat().st_size > 0:
        printlog(f"Using existing performance analysis: {analysis_results_path}")
        return pd.read_csv(analysis_results_path)
    
    printlog(f"Analyzing performance of {docking_program}/{selection_method}")
    try:
        analysis_results = run_consensus_analysis(
            scoring_data_path=scoring_data_path,
            activity_data_path=activity_data_path,
            thresholds=thresholds,
            n_jobs=n_cpus,
            include_single=True
        )
        
        # Add metadata about this parameter combination
        analysis_results['docking_program'] = docking_program
        analysis_results['selection_method'] = selection_method
        
        # Save this specific combination's results
        analysis_results.to_csv(analysis_results_path, index=False)
        
        printlog(f"Performance analysis complete: {analysis_results_path}")
        return analysis_results
    except Exception as e:
        printlog(f"Error analyzing performance: {e}")
        raise


def determine_optimal_parameters(combined_results, performance_dir):
    """Determine optimal parameters based on enrichment factors."""
    try:
        # Filter to threshold=1 (top 1%) for primary optimization metric
        if "threshold" in combined_results.columns:
            results_at_1pct = combined_results[combined_results["threshold"] == 1]
        else:
            results_at_1pct = combined_results

        # Sort by enrichment factor (column name from calculate_metrics is "ef")
        ef_col = "ef" if "ef" in results_at_1pct.columns else "EF1"
        best_row = results_at_1pct.sort_values(by=ef_col, ascending=False).iloc[0]

        optimal_params = {
            "docking_program": best_row["docking_program"],
            "selection_method": best_row["selection_method"],
        }

        if "scoring_function" in best_row and not pd.isna(best_row["scoring_function"]):
            optimal_params["rescoring_functions"] = [best_row["scoring_function"]]
            optimal_params["consensus_method"] = "none"
        elif "combination" in best_row and not pd.isna(best_row["combination"]):
            optimal_params["rescoring_functions"] = best_row["combination"].split("+")
            optimal_params["consensus_method"] = best_row["consensus_method"]
        else:
            raise ValueError("Could not determine optimal scoring functions from analysis results")

        # Add performance metrics using actual column names from calculate_metrics
        auc_col = "auc_roc" if "auc_roc" in best_row.index else "AUC"
        optimal_params["metrics"] = {
            "ef": float(best_row.get(ef_col, 0)),
            "auc_roc": float(best_row.get(auc_col, 0)),
        }

        optimal_params_path = performance_dir / "optimal_parameters.json"
        with open(optimal_params_path, "w") as f:
            json.dump(optimal_params, f, indent=2)

        printlog("Optimal parameters determined:")
        printlog(f"  Docking program: {optimal_params['docking_program']}")
        printlog(f"  Selection method: {optimal_params['selection_method']}")
        printlog(f"  Rescoring functions: {', '.join(optimal_params['rescoring_functions'])}")
        printlog(f"  Consensus method: {optimal_params['consensus_method']}")
        printlog("  Performance metrics:")
        for metric, value in optimal_params['metrics'].items():
            printlog(f"    {metric}: {value:.4f}")

        return optimal_params
    except Exception as e:
        printlog(f"Error determining optimal parameters: {e}")
        raise


from scripts.consensus.consensus import normalize_input_data


def prepare_final_results(results_by_receptor, working_dir, ensemble_mode, using_consensus, id_column, ensemble_top_percent):
    """Prepare final output files based on processing mode."""
    if ensemble_mode:
        output_dir = working_dir / "final_results"
    else:
        # For single mode, use the final_results_dir from paths
        paths = get_workflow_paths(working_dir, "single")
        output_dir = paths['final_results_dir']
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Single receptor mode
    if not ensemble_mode:
        if not results_by_receptor:
            printlog("Error: No successful receptor processing results found.")
            return None
            
        # Just use the first (and only) receptor's results in single mode
        receptor_name = next(iter(results_by_receptor))
        receptor_data = results_by_receptor[receptor_name]
        
        if using_consensus:
            # Single mode with consensus: Final SDF with consensus scores
            final_output = output_dir / "final_results_with_consensus.sdf"
            
            if final_output.exists() and final_output.stat().st_size > 0:
                printlog(f"Using existing final results: {final_output}")
                return final_output
                
            try:
                # Load rescored poses and consensus results
                rescored_df = fast_load_sdf(
                    receptor_data['rescored'],
                    molColName="Molecule",
                    idName="Pose ID",
                    batch_size=100
                )
                
                # Load consensus results
                consensus_df = pd.read_csv(receptor_data['consensus'])
                
                # Ensure IDs are strings for reliable merging
                rescored_df["ID"] = rescored_df["ID"].astype(str)
                consensus_df["ID"] = consensus_df["ID"].astype(str)
                
                # Merge consensus scores with poses
                merged_df = pd.merge(
                    rescored_df,
                    consensus_df,
                    left_on="ID",
                    right_on="ID",
                    how="inner"
                )
                
                # Sort by consensus score (highest first) for single mode
                consensus_score_col = [col for col in consensus_df.columns if col != "ID"][0]
                merged_df = merged_df.sort_values(consensus_score_col, ascending=False)
                
                # Save the final output with poses and consensus scores
                fast_write_sdf(
                    df=merged_df,
                    output_path=final_output,
                    molColName="Molecule",
                    idName="ID",
                    properties=list(merged_df.columns)
                )
                
                printlog(f"Final results with consensus scores saved to: {final_output}")
                return final_output
                
            except Exception as e:
                printlog(f"Error creating final output with consensus: {e}")
                import traceback
                printlog(traceback.format_exc())
                return None
        else:
            # Single mode with single function: Final SDF with normalized scores
            final_output = output_dir / "final_results_single_function.sdf"
            
            if final_output.exists() and final_output.stat().st_size > 0:
                printlog(f"Using existing final results: {final_output}")
                return final_output
                
            try:
                # For single function, we can just use the rescored poses directly
                rescored_df = fast_load_sdf(
                    receptor_data['rescored'],
                    molColName="Molecule",
                    idName="Pose ID",
                    batch_size=100
                )
                
                score_columns = [col for col in rescored_df.columns if col in RESCORING_FUNCTIONS]
                
                # Sort by raw score using the correct direction from RESCORING_FUNCTIONS
                if score_columns:
                    score_col = score_columns[0]
                    scoring_info = RESCORING_FUNCTIONS[score_col]
                    # If best value is "min", sort ascending; if "max", sort descending
                    ascending = scoring_info["best_value"] == "min"
                    rescored_df = rescored_df.sort_values(score_col, ascending=ascending)
                
                # Save the final output
                fast_write_sdf(
                    df=rescored_df,
                    output_path=final_output,
                    molColName="Molecule",
                    idName="Pose ID",
                    properties=list(rescored_df.columns)
                )
                
                printlog(f"Final results with normalized scores saved to: {final_output}")
                return final_output
                
            except Exception as e:
                printlog(f"Error creating final output with single function: {e}")
                import traceback
                printlog(traceback.format_exc())
                return None
    
    # Ensemble mode
    else:
        final_output = output_dir / "final_ensemble_results.csv"
        
        if final_output.exists() and final_output.stat().st_size > 0:
            printlog(f"Using existing ensemble results: {final_output}")
            return final_output
            
        try:
            all_receptor_dfs = []
            
            for receptor_name, receptor_data in results_by_receptor.items():
                if using_consensus:
                    # Load consensus results for this receptor
                    df = pd.read_csv(receptor_data['consensus'])
                    id_col_to_use = id_column if id_column in df.columns else 'ID'
                    
                    # Find the score column (consensus method name or first non-ID column)
                    score_columns = [col for col in df.columns if col != id_col_to_use]
                    if score_columns:
                        score_col = score_columns[0]
                        # Calculate which compounds are in the top X%
                        threshold = np.percentile(df[score_col].values, 100 - ensemble_top_percent)
                        top_mask = df[score_col] >= threshold
                        
                        # Select only top compounds and rename score column for clarity
                        top_df = df.loc[top_mask, [id_col_to_use, score_col]]
                        top_df = top_df.rename(columns={
                            id_col_to_use: "ID",
                            score_col: f"consensus_score_{receptor_name}"
                        })
                        
                        all_receptor_dfs.append(top_df)
                else:
                    # Load rescored results for this receptor
                    df = pd.read_csv(receptor_data['rescored'].with_suffix('.csv'))
                    id_col_to_use = id_column if id_column in df.columns else 'ID'
                    
                    # Find score columns (non-structural columns that aren't ID)
                    basic_cols = {"ID", "Pose ID", "Pose_Number", "Docking_program", "SMILES"}
                    score_columns = [col for col in df.columns
                                    if col not in basic_cols and not col.startswith("RDKit")]
                    
                    if score_columns:
                        # Use the first scoring function
                        score_col = score_columns[0]
                        
                        # Use normalize_input_data for proper normalization
                        scores_df = df[[id_col_to_use, score_col]].copy()
                        normalized_df, _ = normalize_input_data(scores_df, [score_col], id_col_to_use)
                        
                        # Rename normalized column for clarity
                        norm_col = score_col  # In normalized_df, the column keeps its original name but is normalized
                        
                        # Calculate which compounds are in the top X%
                        threshold = np.percentile(normalized_df[norm_col].values, 100 - ensemble_top_percent)
                        top_mask = normalized_df[norm_col] >= threshold
                        
                        # Get unique compounds
                        df_unique = normalized_df.loc[top_mask].sort_values(norm_col, ascending=False)
                        df_unique = df_unique.drop_duplicates(subset=[id_col_to_use])
                        
                        # Select only top compounds and rename for clarity
                        top_df = df_unique[[id_col_to_use, norm_col]]
                        top_df = top_df.rename(columns={
                            id_col_to_use: "ID",
                            norm_col: f"score_{receptor_name}"
                        })
                        
                        all_receptor_dfs.append(top_df)
            
            # Find compounds common to all receptors
            if not all_receptor_dfs:
                printlog("No valid data from any receptor. Cannot create ensemble results.")
                pd.DataFrame().to_csv(final_output, index=False)
                return final_output
                
            merged_df = all_receptor_dfs[0]
            for i in range(1, len(all_receptor_dfs)):
                merged_df = pd.merge(merged_df, all_receptor_dfs[i], on="ID", how="inner")
            
            if merged_df.empty:
                printlog(f"No compounds found in top {ensemble_top_percent}% across all receptors.")
            else:
                printlog(f"Found {len(merged_df)} compounds in top {ensemble_top_percent}% across all {len(all_receptor_dfs)} receptors.")
            
            # Save the final ensemble output
            merged_df.to_csv(final_output, index=False)
            printlog(f"Final ensemble results saved to: {final_output}")
            return final_output
            
        except Exception as e:
            printlog(f"Error creating ensemble output: {e}")
            import traceback
            printlog(traceback.format_exc())
            return None


def process_single_receptor_workflow(receptor_path, ref_ligand, args, working_dir,
                                    software_path, prepared_library,
                                    docking_program, selection_method, rescoring_functions,
                                    consensus_method, workflow_mode):
    """Process a single receptor through the workflow pipeline."""
    receptor_name = Path(receptor_path).stem
    
    # Get appropriate paths based on workflow mode
    if workflow_mode == "ensemble":
        paths = get_workflow_paths(working_dir, workflow_mode, receptor_name)
    else:
        paths = get_workflow_paths(working_dir, workflow_mode)
    
    printlog(f"Processing receptor: {receptor_name}")
    
    try:
        # 1. Prepare protein
        prepared_receptor = prepare_receptor_structure(receptor_path, args, paths)
        
        # 2. Determine docking pocket
        pocket_definition = determine_binding_pocket(prepared_receptor, args, paths, ref_ligand)
        
        # 3. Perform docking
        docking_output = perform_docking(
            prepared_library, prepared_receptor, pocket_definition,
            docking_program, args, paths, software_path
        )
        
        # 4. Optional: Apply PoseBusters
        docking_output = apply_posebusters(docking_output, prepared_receptor, args, paths)
        
        # 5. Pose selection
        selected_poses = select_docked_poses(
            docking_output, prepared_receptor, pocket_definition,
            selection_method, args, paths, software_path
        )
        
        # 6. Rescoring
        rescored_poses = rescore_selected_poses(
            selected_poses, prepared_receptor, pocket_definition,
            rescoring_functions, args, paths, software_path
        )
        
        # 7. Consensus scoring (if applicable)
        rescored_poses_csv = rescored_poses.with_suffix('.csv')
        consensus_result = calculate_consensus(
            rescored_poses_csv, consensus_method, rescoring_functions,
            args.id_column, paths
        )
        
        # Return results
        return {
            'dir': paths.get('receptor_dir', working_dir),
            'receptor': prepared_receptor,
            'poses': docking_output,
            'selected': selected_poses,
            'rescored': rescored_poses,
            'consensus': consensus_result
        }
    
    except Exception as e:
        printlog(f"Error processing receptor {receptor_name}: {e}")
        import traceback
        printlog(traceback.format_exc())
        return None


def run_decoy_optimization_workflow(args):
    """Run the decoy-based parameter optimization workflow."""
    software_path = Path(args.software)
    
    # Set up directory structure
    working_dir = setup_working_directory(args.docking_library)
    
    # Check for existing optimal parameters
    optimal_params_path = working_dir / "performance" / "optimal_parameters.json"
    if optimal_params_path.exists():
        try:
            import json
            with open(optimal_params_path) as f:
                optimal_params = json.load(f)
            
            printlog("Found existing optimal parameters:")
            printlog(f"  Docking program: {optimal_params['docking_program']}")
            printlog(f"  Selection method: {optimal_params['selection_method']}")
            printlog(f"  Rescoring functions: {', '.join(optimal_params['rescoring_functions'])}")
            printlog(f"  Consensus method: {optimal_params['consensus_method']}")
            
            # Get the decoy library path
            decoy_library = working_dir / "decoys" / f"{Path(args.actives).stem}_decoys.sdf"
            if decoy_library.exists():
                return decoy_library, optimal_params
        except Exception as e:
            printlog(f"Error loading existing optimal parameters: {e}")
    
    # 1. Generate decoys
    decoys_dir = working_dir / "decoys"
    decoys_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        decoy_library = generate_decoy_library(args.actives, args, software_path, decoys_dir)
    except Exception as e:
        printlog(f"Critical error in decoy generation: {e}")
        sys.exit(1)
    
    # 2. Prepare the protein structure
    try:
        # Get paths for decoy mode (single receptor with decoy optimization)
        paths = get_workflow_paths(working_dir, "decoy")
        prepared_receptor = prepare_receptor_structure(args.receptor[0], args, paths)
    except Exception as e:
        printlog(f"Critical error in protein preparation: {e}")
        sys.exit(1)
    
    # 3. Determine binding pocket
    try:
        ref_ligand = Path(args.reference_ligand[0]) if args.reference_ligand else None
        pocket_definition = determine_binding_pocket(prepared_receptor, args, paths, ref_ligand)
    except Exception as e:
        printlog(f"Critical error in pocket determination: {e}")
        sys.exit(1)
    
    # 4. Prepare decoy library
    try:
        library_dir = working_dir / "library"
        library_dir.mkdir(parents=True, exist_ok=True)
        prepared_library = prepare_compound_library(decoy_library, args, software_path, working_dir)
    except Exception as e:
        printlog(f"Critical error in library preparation: {e}")
        sys.exit(1)
    
    # 5. Create activity data CSV
    performance_dir = working_dir / "performance"
    performance_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        activity_data_path = create_activity_data(decoy_library, args.n_cpus, performance_dir)
    except Exception as e:
        printlog(f"Error creating activity data: {e}")
        sys.exit(1)
    
    # Create combined results file path
    combined_results_path = performance_dir / "all_performance_results.csv"
    
    # Check if we already have combined results
    if combined_results_path.exists() and combined_results_path.stat().st_size > 0:
        printlog(f"Using existing performance results: {combined_results_path}")
        combined_results = pd.read_csv(combined_results_path)
    else:
        # 6. Process each docking program and selection method combination
        all_performance_results = []
        
        for docking_program in args.docking_program:
            # Perform docking
            docking_dir = working_dir / "docking" / docking_program.lower()
            docking_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                docking_output = perform_docking(
                    prepared_library, prepared_receptor, pocket_definition,
                    docking_program, args, {'docking_dir': docking_dir}, software_path
                )

                # Apply PoseBusters if requested
                docking_output = apply_posebusters(docking_output, prepared_receptor, args, {'docking_dir': docking_dir})
                
                # If docking fails, continue with next program
                if not docking_output or not docking_output.exists():
                    continue
                
                # For each docking output, process all selection methods
                for selection_method in args.selection_method:
                    # Set up directories
                    selection_dir = working_dir / "pose_selection" / docking_program.lower() / selection_method
                    selection_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Select poses
                    try:
                        selected_poses = select_docked_poses(
                            docking_output, prepared_receptor, pocket_definition,
                            selection_method, args, {'selection_dir': selection_dir}, software_path
                        )
                        
                        # Rescore poses
                        rescoring_dir = working_dir / "rescoring" / docking_program.lower() / selection_method
                        rescoring_dir.mkdir(parents=True, exist_ok=True)
                        
                        rescored_poses = rescore_selected_poses(
                            selected_poses, prepared_receptor, pocket_definition,
                            args.rescoring_functions, args, {'rescoring_dir': rescoring_dir}, software_path
                        )
                        
                        # Analyze performance
                        analysis_results = analyze_performance(
                            rescored_poses.with_suffix('.csv'), activity_data_path,
                            args.performance_thresholds, args.n_cpus,
                            docking_program, selection_method, performance_dir
                        )
                        
                        all_performance_results.append(analysis_results)
                        
                    except Exception as e:
                        printlog(f"Error processing {docking_program}/{selection_method}: {e}")
                        continue
                    
            except Exception as e:
                printlog(f"Error with docking program {docking_program}: {e}")
                continue
        
        # Combine all results
        if all_performance_results:
            combined_results = pd.concat(all_performance_results, ignore_index=True)
            combined_results.to_csv(combined_results_path, index=False)
        else:
            printlog("Error: No valid performance results were obtained from any parameter combination")
            sys.exit(1)
    
    # 7. Find best parameters
    try:
        optimal_params = determine_optimal_parameters(combined_results, performance_dir)
    except Exception as e:
        printlog(f"Error determining optimal parameters: {e}")
        sys.exit(1)
    
    return decoy_library, optimal_params


def run_virtual_screening_workflow(args, optimal_params=None):
    """Run the virtual screening workflow with either user-specified or optimized parameters."""
    software_path = Path(args.software)
    
    # Set up directory structure
    working_dir = setup_working_directory(args.docking_library)
    
    # Use optimal parameters if provided, otherwise use user-specified ones
    docking_program = args.docking_program
    selection_method = args.selection_method
    rescoring_functions = args.rescoring_functions
    consensus_method = args.consensus_method

    if optimal_params:
        printlog("Using optimized parameters from decoy analysis:")
        if 'docking_program' in optimal_params:
             docking_program = optimal_params["docking_program"]
        selection_method = optimal_params["selection_method"]
        rescoring_functions = optimal_params["rescoring_functions"]
        consensus_method = optimal_params["consensus_method"]
        
        printlog(f"  Docking program: {docking_program}")
        printlog(f"  Selection method: {selection_method}")
        printlog(f"  Rescoring functions: {', '.join(rescoring_functions)}")
        printlog(f"  Consensus method: {consensus_method}")
    else:
        if isinstance(docking_program, list):
            docking_program = docking_program[0]
        if isinstance(selection_method, list):
            selection_method = selection_method[0]

    # Determine workflow mode
    workflow_mode = determine_workflow_mode(args)
    ensemble_mode = workflow_mode == "ensemble"
    
    if ensemble_mode:
        printlog("Running in ensemble mode with multiple receptors")
        if not isinstance(args.receptor, list) or len(args.receptor) < 2:
            printlog("Warning: Ensemble mode requires at least two receptors. Switching to single receptor mode.")
            ensemble_mode = False
            workflow_mode = "single"
            if not isinstance(args.receptor, list):
                args.receptor = [args.receptor]
    elif not isinstance(args.receptor, list):
        args.receptor = [args.receptor]
    
    # Prepare compound library (once for all receptors)
    try:
        # Get library directory based on workflow mode
        if workflow_mode == "ensemble":
            library_dir = working_dir / "library"
        else:
            paths = get_workflow_paths(working_dir, workflow_mode)
            library_dir = paths['library_dir']
        
        prepared_library = prepare_compound_library(args.docking_library, args, software_path, library_dir)
    except Exception as e:
        printlog(f"Critical error in library preparation: {e}")
        sys.exit(1)
    
    # Process each receptor
    results_by_receptor = {}
    receptor_list = [str(r) for r in args.receptor]  # Ensure paths are strings

    for i, receptor_path_str in enumerate(receptor_list):
        receptor_path = Path(receptor_path_str)
        
        # Get reference ligand for this receptor if available
        ref_ligand = None
        if args.reference_ligand:
            ref_ligand_list = args.reference_ligand if isinstance(args.reference_ligand, list) else [args.reference_ligand]
            if len(ref_ligand_list) > i:
                ref_ligand = Path(ref_ligand_list[i])
            elif ref_ligand_list:
                ref_ligand = Path(ref_ligand_list[0])
        
        # Process this receptor
        result = process_single_receptor_workflow(
            receptor_path, ref_ligand, args, working_dir, software_path, prepared_library,
            docking_program, selection_method, rescoring_functions, consensus_method, workflow_mode
        )
        
        if result:
            results_by_receptor[receptor_path.stem] = result
    
    # Prepare final output
    printlog("Preparing final results")
    using_consensus = (consensus_method.lower() != "none" and len(rescoring_functions) > 1)
    final_output = prepare_final_results(
        results_by_receptor, working_dir, ensemble_mode,
        using_consensus, args.id_column, args.ensemble_top_percent
    )
    
    if final_output:
        printlog(f"Final results available at: {final_output}")
    
    return working_dir


def main():
    """Main entry point for the DockM8 pipeline."""
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Unified workflow that handles both decoy optimization and virtual screening
        if args.generate_decoys:
            printlog("Running decoy-based optimization workflow")
            decoy_library, optimal_params = run_decoy_optimization_workflow(args)
            
            printlog("Running virtual screening with optimized parameters")
            working_dir = run_virtual_screening_workflow(args, optimal_params)
        else:
            printlog("Running virtual screening with user-provided parameters")
            working_dir = run_virtual_screening_workflow(args)
        
        elapsed_time = time.time() - start_time
        printlog(f"DockM8 completed in {elapsed_time:.2f} seconds")
        printlog(f"Results saved to {working_dir}")
        
    except Exception as e:
        printlog(f"Error during DockM8 execution: {e}")
        import traceback
        printlog(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
