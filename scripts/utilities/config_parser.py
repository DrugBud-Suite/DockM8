import math
import os
import re
import sys
from pathlib import Path

import yaml


def none_constructor(loader, node):
    return None


yaml.add_constructor("tag:yaml.org,2002:null", none_constructor)

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.pose_selection.clustering.clustering_metrics.clustering_metrics import CLUSTERING_METRICS
from scripts.consensus.consensus import CONSENSUS_METHODS
from scripts.docking.docking import DOCKING_PROGRAMS
from scripts.library_preparation.library_preparation import CONFORMER_OPTIONS, PROTONATION_OPTIONS
from scripts.pocket_finding.pocket_finder import POCKET_DETECTION_OPTIONS
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS
from scripts.utilities.logging import printlog


class DockM8Error(Exception):
    """Custom Error for DockM8 specific issues."""

    def __init__(self, message):
        printlog(message)  # Log the message when the exception is created
        super().__init__(message)  # Call the superclass constructor with the message


class DockM8Warning(Warning):
    """Custom warning for DockM8 specific issues."""

    def __init__(self, message):
        printlog(message)  # Log the message when the warning is created
        super().__init__(message)  # Call the superclass constructor with the message


def resolve_path(path, config_dir):
    if isinstance(path, str):
        # Convert to Path object for easier manipulation
        path_obj = Path(path)

        # If it's an absolute path, use it as is
        if path_obj.is_absolute():
            if path_obj.exists():
                return str(path_obj)
            else:
                raise FileNotFoundError(f"File not found: {path}")

        # For relative paths, always resolve relative to config_dir
        resolved_path = config_dir / path_obj
        if resolved_path.exists():
            return str(resolved_path.resolve())
        else:
            raise FileNotFoundError(f"File not found: {resolved_path}")

    return path


def process_config_paths(config, config_file_path):
    config_dir = Path(config_file_path).parent

    # Process receptor paths
    if "receptor(s)" in config:
        config["receptor(s)"] = [resolve_path(r, config_dir) for r in config["receptor(s)"]]

    # Process docking library path
    if "docking_library" in config:
        config["docking_library"] = resolve_path(config["docking_library"], config_dir)

    # Process reference ligand paths
    if "pocket_detection" in config and "reference_ligand(s)" in config["pocket_detection"]:
        config["pocket_detection"]["reference_ligand(s)"] = [
            resolve_path(r, config_dir) for r in config["pocket_detection"]["reference_ligand(s)"]
        ]

    return config


def check_config(config_input):
    """Check the configuration file or dictionary for any errors or warnings."""
    printlog("Validating DockM8 configuration...")

    if isinstance(config_input, (str, Path)):
        config_file_path = Path(config_input).resolve()
        if not config_file_path.is_file():
            raise DockM8Error(f"Configuration file not found: {config_file_path}")

        try:
            with open(config_file_path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise DockM8Error(f"Error loading configuration file: {e}")

        # Process paths after loading the config
        try:
            config = process_config_paths(config, config_file_path)
        except FileNotFoundError as e:
            raise DockM8Error(f"Error processing paths in configuration: {str(e)}")
    elif isinstance(config_input, dict):
        config = config_input
    else:
        raise DockM8Error("Invalid configuration input. Expected a dictionary or a file path.")

    general_config = config.get("general", {})

    # Handle software path
    software_path = general_config.get("software")
    if software_path in [None, "None", "none", ""]:
        software_path = dockm8_path / "software"
        DockM8Warning(
            f"DockM8 configuration warning: Software path not specified or set to None in the configuration file. Defaulting to {software_path}"
        )
    # Check if the software path is a valid directory
    if not Path(software_path).is_dir():
        raise DockM8Error(
            f"DockM8 configuration error: Invalid software path ({software_path}) specified in the configuration file."
        )
    else:
        config["general"]["software"] = Path(software_path)

    # Retrieve the mode from the configuration, defaulting to 'single' if not found
    mode = general_config.get("mode", "single")
    # Normalize and validate the mode to handle different case inputs and ensure it's one of the expected values
    if mode.lower() not in ["single", "ensemble", "active_learning"]:
        raise DockM8Error(
            "DockM8 configuration error: Invalid mode ({}) specified in the configuration file.".format(mode)
        )

    # Retrieve the number of CPUs to use from the configuration, defaulting to 0 if not found or improperly specified
    n_cpus = general_config.get("n_cpus", 0)
    try:
        # Attempt to convert n_cpus to an integer and use a fallback if it's set to 0
        config["general"]["n_cpus"] = int(n_cpus) if int(n_cpus) != 0 else int(os.cpu_count() * 0.9)
    except ValueError:
        raise DockM8Error(
            f"DockM8 configuration error: Invalid n_cpus value ({n_cpus}) specified in the configuration file. It must be a valid integer."
        )

    decoy_config = config.get("decoy_generation", {})
    # Check if decoy generation is enabled and validate conditions
    if decoy_config.get("gen_decoys", False):  # Default to False if gen_decoys is not specified
        # Validate the active compounds path
        active_path = decoy_config.get("actives")
        if not Path(active_path).is_file():
            raise DockM8Error(
                f"DockM8 configuration error: Invalid actives path ({active_path}) specified in the configuration file."
            )
        else:
            config["decoy_generation"]["actives"] = active_path
            # Validate the number of decoys to generate
        try:
            int(decoy_config.get("n_decoys"))
        except ValueError:
            raise DockM8Error(
                f"DockM8 configuration error: Invalid number of decoys ({decoy_config.get('n_decoys')}) specified; it must be an integer."
            )

        # Validate the decoy model
        decoy_model = decoy_config.get("decoy_model")
        if decoy_model not in ["DEKOIS", "DUDE", "DUDE_P"]:
            raise DockM8Error(
                f"DockM8 configuration error: Invalid decoy model ({decoy_model}) specified in the configuration file."
            )

        # Check for rescoring configuration, which must also be correctly formatted and within limits
        rescoring_config = config.get("rescoring", [])
        if len(rescoring_config) > 8:
            # Calculating the number of possible combinations for warnings
            try:
                pose_selection_methods = len(config.get("pose_selection", {}).get("methods", []))
                docking_methods = len(config.get("docking", []))
                possibilities = math.factorial(len(rescoring_config)) * pose_selection_methods * docking_methods * 7
                DockM8Warning(
                    f"At least {possibilities} possible combinations will be tried for optimization, this may take a while."
                )
            except Exception as e:
                raise DockM8Error(f"Error calculating possibilities: {str(e)}")

    # Check the protein input files
    receptors = config.get("receptor(s)", [])

    if mode == "single":
        if len(receptors) > 1:
            DockM8Warning(
                "DockM8 configuration warning: Multiple receptor files detected in single mode, only the first file will be used."
            )
        config["receptor(s)"] = receptors[:1]  # Slice to keep only the first element as a list
    else:
        pass
    for receptor in receptors:
        if len(receptor) == 4 and receptor.isalnum() and not receptor.isdigit():
            printlog(f"PDB ID detected: {receptor}, structure will be downloaded from the PDB.")
        elif len(receptor) == 6 and receptor.isalnum() and not receptor.isdigit():
            printlog(f"Uniprot ID detected: {receptor}, AlphaFold structure will be downloaded from the database.")
        else:
            if not receptor.endswith(".pdb"):
                raise DockM8Error(
                    f"DockM8 configuration error: Invalid receptor file format ({receptor}) specified in the configuration file. Please use .pdb files."
                )
            if not Path(receptor).is_file():
                raise DockM8Error(
                    f"DockM8 configuration error: Invalid receptor path ({receptor}) specified in the configuration file."
                )
    config["receptor(s)"] = [Path(receptor) for receptor in receptors]

    # Check docking library configuration
    docking_library = config.get("docking_library")
    if not Path(docking_library).is_file():
        raise DockM8Error(
            f"DockM8 configuration error: Invalid docking library path ({docking_library}) specified in the configuration file."
        )
    elif not docking_library.endswith(".sdf"):
        raise DockM8Error(
            f"DockM8 configuration error: Invalid docking library file format ({docking_library}) specified in the configuration file. Please use .sdf files."
        )
    else:
        config["docking_library"] = Path(docking_library)
    # Check protein preparation configuration
    protein_preparation = config.get("protein_preparation", {})

    # Define the conditions expected to be boolean values
    conditions = [
        "select_best_chain",
        "minimize",
        "fix_nonstandard_residues",
        "fix_missing_residues",
        "remove_hetero",
        "remove_water",
        "protonate",
    ]

    # Check each condition for boolean type
    for condition in conditions:
        # Use `.get()` to safely access the configuration, defaulting to None if not found
        if not isinstance(protein_preparation.get(condition), bool):
            raise DockM8Error(
                f"DockM8 configuration error: '{condition}' in 'protein_preparation' section must be a boolean (true/false) value."
            )

    # Check the 'add_hydrogens' setting under specific conditions
    add_hydrogens = protein_preparation.get("add_hydrogens")
    protonation = protein_preparation.get("protonation", False)

    # Check if 'add_hydrogens' should be ignored based on 'protonation' being True
    if add_hydrogens is not None and protonation:
        DockM8Warning("DockM8 configuration warning: 'add_hydrogens' will be ignored as 'protonation' is set to True.")

    # Logically, it's also important to confirm 'add_hydrogens' is a valid numeric type if set
    if add_hydrogens is not None and not isinstance(add_hydrogens, (int, float)):
        raise DockM8Error(
            f"DockM8 configuration error: 'add_hydrogens' value ({add_hydrogens}) must be a numeric type."
        )

    # Check ligand preparation configuration
    ligand_preparation = config.get("ligand_preparation", {})

    # Validate 'protonation' setting
    protonation = ligand_preparation.get("protonation")
    if protonation not in PROTONATION_OPTIONS:
        raise DockM8Error(
            f"DockM8 configuration error: 'protonation' in 'ligand_preparation' section must be either {', '.join(PROTONATION_OPTIONS)}."
        )

    # Validate 'conformers' setting
    conformers = ligand_preparation.get("conformers")
    if conformers not in CONFORMER_OPTIONS:
        raise DockM8Error(
            f"DockM8 configuration error: 'conformers' in 'ligand_preparation' section must be either {', '.join(CONFORMER_OPTIONS)}."
        )

    # Validate 'n_conformers' is an integer
    n_conformers = ligand_preparation.get("n_conformers")
    if not isinstance(n_conformers, int):
        raise DockM8Error(
            "DockM8 configuration error: 'n_conformers' in 'ligand_preparation' section must be an integer value."
        )

    # Validate 'min_ph' is a float
    min_ph = ligand_preparation.get("min_ph")
    if not isinstance(min_ph, (float, int)):
        raise DockM8Error("DockM8 configuration error: 'min_ph' in 'ligand_preparation' section must be a float value.")

    # Validate 'max_ph' is a float
    max_ph = ligand_preparation.get("max_ph")
    if not isinstance(max_ph, (float, int)):
        raise DockM8Error("DockM8 configuration error: 'max_ph' in 'ligand_preparation' section must be a float value.")

    # Validate 'pka_precision' is a float
    pka_precision = ligand_preparation.get("pka_precision")
    if not isinstance(pka_precision, (float, int)):
        raise DockM8Error(
            "DockM8 configuration error: 'pka_precision' in 'ligand_preparation' section must be a float value."
        )

    # Validate standardize_ids, standardize_tautomers, remove_salts are booleans
    standardize_ids = ligand_preparation.get("standardize_ids")
    if not isinstance(standardize_ids, bool):
        raise DockM8Error(
            "DockM8 configuration error: 'standardize_ids' in 'ligand_preparation' section must be a boolean (true/false) value."
        )
    standardize_tautomers = ligand_preparation.get("standardize_tautomers")
    if not isinstance(standardize_tautomers, bool):
        raise DockM8Error(
            "DockM8 configuration error: 'standardize_tautomers' in 'ligand_preparation' section must be a boolean (true/false) value."
        )
    remove_salts = ligand_preparation.get("remove_salts")
    if not isinstance(remove_salts, bool):
        raise DockM8Error(
            "DockM8 configuration error: 'remove_salts' in 'ligand_preparation' section must be a boolean (true/false) value."
        )

    # Check pocket detection configuration
    pocket_detection = config.get("pocket_detection", {})

    # Valid pocket detection methods
    method = pocket_detection.get("method")

    # Validate the method
    if method not in POCKET_DETECTION_OPTIONS:
        raise DockM8Error(
            f"DockM8 configuration error: Invalid pocket detection method ({method}) specified in the configuration file. Must be either {', '.join(POCKET_DETECTION_OPTIONS)}."
        )

    # Validate reference ligands for specific methods
    if method in ["Reference", "RoG"]:
        reference_ligands = pocket_detection.get("reference_ligand(s)", [])
        if not reference_ligands:
            raise DockM8Error(
                "DockM8 configuration error: Reference ligand(s) file(s) path(s) is/are required for 'Reference' or 'RoG' pocket detection methods."
            )
        for reference_ligand in reference_ligands:
            if not reference_ligand.endswith(".sdf"):
                raise DockM8Error(
                    f"DockM8 configuration error: Invalid reference ligand file format ({reference_ligand}) specified in the configuration file. Please use .sdf files."
                )
            if not Path(reference_ligand).is_file():
                raise DockM8Error(
                    f"DockM8 configuration error: Invalid reference ligand file path ({reference_ligand}) specified in the configuration file."
                )
        if mode == "single" and len(reference_ligands) > 1:
            DockM8Warning(
                "DockM8 configuration warning: Multiple reference ligand files detected in single mode, only the first file will be used."
            )
            reference_ligands = reference_ligands[:1]
        config["pocket_detection"]["reference_ligand(s)"] = [
            Path(reference_ligand) for reference_ligand in reference_ligands
        ]
    # Validate radius for the "Reference" method
    if method == "Reference":
        radius = pocket_detection.get("radius")
        if not isinstance(radius, (float, int)):
            raise DockM8Error("DockM8 configuration error: Pocket detection radius must be a number.")

    # Validate manual pocket definitions for the "Manual" method
    if method == "Manual":
        manual_pocket = pocket_detection.get("manual_pocket")
        if not manual_pocket:
            raise DockM8Error(
                "DockM8 configuration error: Manual pocket definition is required for 'Manual' pocket detection method."
            )
        if not re.match(
            r"center:-?\d+(\.\d+)?,-?\d+(\.\d+)?,-?\d+(\.\d+)?\*size:-?\d+(\.\d+)?,-?\d+(\.\d+)?,-?\d+(\.\d+)?",
            manual_pocket,
        ):
            raise DockM8Error(
                "DockM8 configuration error: Invalid manual pocket definition format. Format should be 'center:x,y,z*size:x,y,z' where x, y, and z are numbers."
            )

    # Check docking configuration
    docking = config.get("docking", {})

    # Validate each docking program specified in the configuration
    docking_programs = docking.get("docking_programs", [])
    for program in docking_programs:
        if program not in DOCKING_PROGRAMS:
            raise DockM8Error(
                f"DockM8 configuration error: Invalid docking program ({program}) specified in the configuration file. Must be one of {', '.join(DOCKING_PROGRAMS.keys())}."
            )

    # Validate 'n_poses' to ensure it is an integer
    n_poses = docking.get("n_poses")
    if not isinstance(n_poses, int):
        raise DockM8Error("DockM8 configuration error: 'n_poses' in 'docking' section must be an integer value.")

    # Validate 'exhaustiveness' to ensure it is an integer
    exhaustiveness = docking.get("exhaustiveness")
    if not isinstance(exhaustiveness, int):
        raise DockM8Error("DockM8 configuration error: 'exhaustiveness' in 'docking' section must be an integer value.")

    # Check if exhaustiveness is not set and SMINA, GNINA, QVINA2, or QVINAW are in docking programs
    if "exhaustiveness" not in docking and any(
        program in docking_programs for program in ["SMINA", "GNINA", "QVINA2", "QVINAW"]
    ):
        DockM8Warning(
            "DockM8 configuration warning: exhaustiveness is not set and SMINA, GNINA, QVINA2, or QVINAW are in docking programs. Setting exhaustiveness to 8."
        )
        config["docking"]["exhaustiveness"] = 8

    # Check Docking postprocessing settings
    post_docking = config.get("post_docking", {})
    minimize_poses = post_docking.get("minimize_poses")
    if not isinstance(minimize_poses, bool):
        raise DockM8Error(
            "DockM8 configuration error: 'minimize_poses' in 'post_docking' section must be a boolean (true/false) value."
        )
    clash_cutoff = post_docking.get("clash_cutoff")
    if clash_cutoff not in [None, "None", "none"] and not isinstance(clash_cutoff, int):
        raise DockM8Error(
            "DockM8 configuration error: 'clash_cutoff' in 'post_docking' section must be an integer value or None."
        )

    strain_cutoff = post_docking.get("strain_cutoff")
    if strain_cutoff not in [None, "None", "none"] and not isinstance(strain_cutoff, int):
        raise DockM8Error(
            "DockM8 configuration error: 'strain_cutoff' in 'post_docking' section must be an integer value or None."
        )

    # Convert "None" and "none" to None for consistency
    if clash_cutoff in ["None", "none"]:
        config["post_docking"]["clash_cutoff"] = None
    if strain_cutoff in ["None", "none"]:
        config["post_docking"]["strain_cutoff"] = None
    bust_poses = post_docking.get("bust_poses")
    if not isinstance(bust_poses, bool):
        raise DockM8Error(
            "DockM8 configuration error: 'bust_poses' in 'post_docking' section must be a boolean (true/false) value."
        )
    classy_pose = post_docking.get("classy_pose")
    if not isinstance(classy_pose, bool):
        raise DockM8Error(
            "DockM8 configuration error: 'classy_pose' in 'post_docking' section must be a boolean (true/false) value."
        )
    classy_pose_model = post_docking.get("classy_pose_model")
    if classy_pose_model in ["SVM", "LGBM"] and not classy_pose:
        classy_pose = True
        DockM8Warning(
            "DockM8 warning: 'classy_pose_model' in 'post_docking' section is set but 'classy_pose' is not enabled. Setting 'classy_pose' to True."
        )
    if classy_pose_model not in ["SVM", "LGBM", "None", None]:
        raise DockM8Error(
            "DockM8 configuration error: 'classy_pose_model' in 'post_docking' section must be either 'SVM' or 'LGBM' or 'None'."
        )

    # Check pose selection configuration
    pose_selection = config.get("pose_selection", {})
    methods = pose_selection.get("method", [])
    docking_programs = config.get("docking", {}).get("docking_programs", [])

    valid_methods = (
        list(CLUSTERING_METRICS.keys())
        + ["bestpose", "bestpose_GNINA", "bestpose_SMINA", "bestpose_PLANTS", "bestpose_QVINA2", "bestpose_QVINAW"]
        + list(RESCORING_FUNCTIONS.keys())
    )

    # Validate each method in pose selection
    for method in methods:
        if method not in valid_methods:
            raise DockM8Error(
                f"DockM8 configuration error: Invalid pose selection method ({method}) specified in the configuration file."
            )

    # Check for program-specific bestpose selections
    for program in DOCKING_PROGRAMS:
        bestpose_key = f"bestpose_{program}"
        if bestpose_key in methods and program not in docking_programs:
            DockM8Warning(
                f"DockM8 configuration warning: {bestpose_key} was selected as a pose selection method but {program} is not in the list of docking programs. Ignoring {bestpose_key}."
            )
            config["pose_selection"]["method"] = methods.remove(bestpose_key)
            # Ensure there's always at least one method in the list
            if not methods:
                config["pose_selection"]["method"] = methods.append(bestpose_key)
                config["docking"]["docking_programs"] = docking_programs.append(program)
                DockM8Warning(
                    f"DockM8 configuration warning: Restored {bestpose_key} as a pose selection method and added {program} to the list of docking programs."
                )

    # Check for clustering method validity
    clustering_method = pose_selection.get("clustering_method")
    if clustering_method not in ["KMedoids", "Aff_Prop", None]:
        raise DockM8Error(
            "DockM8 configuration error: 'clustering_method' in 'pose_selection' section must be either 'KMedoids' or 'Aff_Prop'."
        )

    # Check if clustering metrics are used without a specified clustering method
    if any(method in CLUSTERING_METRICS for method in methods) and not clustering_method:
        DockM8Warning(
            "DockM8 configuration warning: 'clustering_method' is not set for clustering metrics, defaulting to 'KMedoids'."
        )
        config["pose_selection"]["clustering_method"] = "KMedoids"

    # Validate rescoring methods
    rescoring_methods = config.get("rescoring", [])
    valid_rescoring_methods = list(RESCORING_FUNCTIONS.keys())
    for method in rescoring_methods:
        if method not in valid_rescoring_methods:
            raise DockM8Error(
                f"DockM8 configuration error: Invalid rescoring method ({method}) specified in the configuration file."
            )

    # Validate consensus method
    consensus_method = config.get("consensus")
    valid_consensus_methods = list(CONSENSUS_METHODS.keys())
    if consensus_method not in valid_consensus_methods:
        raise DockM8Error(
            f"DockM8 configuration error: Invalid consensus method ({consensus_method}) specified in the configuration file. Must be one of {valid_consensus_methods}."
        )
    # Check threshold for ensemble and active learning modes
    threshold = config.get("threshold", None)

    if mode in ["ensemble", "active_learning"]:
        if threshold is None:
            DockM8Warning(
                f"DockM8 configuration warning: {mode} mode requires a threshold to be set. Setting to default (1%)"
            )
            config["threshold"] = 0.01  # Setting default threshold

    printlog("DockM8 configuration was successfully validated.")

    return config


# config_file = dockm8_path / "config.yml"

# config = check_config(config_file)

# print(config)
