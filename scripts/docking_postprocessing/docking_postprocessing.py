import sys
from pathlib import Path

from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

import warnings

from scripts.docking_postprocessing.posebusters.posebusters import pose_buster
from scripts.docking_postprocessing.posecheck.posecheck import pose_checker
from scripts.docking_postprocessing.minimisation.minimize import minimize_all_ligands
from scripts.docking_postprocessing.classy_pose.classy_pose import classy_pose_filter
from scripts.utilities.utilities import parallel_SDF_loader
from scripts.utilities.logging import printlog

warnings.filterwarnings("ignore")


def log_dataframe_length_change(previous_length, current_length, process_name):
	if previous_length != current_length:
		printlog(f"Removed {previous_length - current_length} poses during {process_name} postprocessing.")


def docking_postprocessing(input_sdf: Path,
							output_path: Path,
							protein_file: Path,
							minimize_poses: bool,
							bust_poses: bool,
							strain_cutoff: int,
							clash_cutoff: int,
							classy_pose: bool,
							classy_pose_model: str,
							n_cpus: int) -> Path:
	"""
    Perform postprocessing on docking results.

    Args:
        input_sdf (Path): Path to the input SDF file containing docking results.
        output_path (Path): Path to save the postprocessed SDF file.
        protein_file (Path): Path to the protein file used for docking.
        minimize_poses (bool): Flag indicating whether to perform ligand minimization.
        bust_poses (bool): Flag indicating whether to apply pose busting.
        strain_cutoff (int): Cutoff value for strain energy. Poses with strain energy above this cutoff will be removed.
        clash_cutoff (int): Cutoff value for clash score. Poses with clash score above this cutoff will be removed.
        classy_pose (bool): Flag indicating whether to perform classy pose analysis.
        classy_pose_model (str): Path to the classy pose model file.
        n_cpus (int): Number of CPUs to use for parallel processing.

    Returns:
        Path: Path to the postprocessed SDF file.
    """
	printlog("Postprocessing docking results...")
	sdf_dataframe = parallel_SDF_loader(input_sdf, molColName="Molecule", idName="Pose ID", n_cpus=n_cpus)
	initial_length = len(sdf_dataframe)

	if minimize_poses:
		sdf_dataframe = minimize_all_ligands(protein_file, str(input_sdf), n_cpus)
	log_dataframe_length_change(initial_length, len(sdf_dataframe), "Ligand Minimization")
	initial_length = len(sdf_dataframe)

	if (strain_cutoff is not None) and (clash_cutoff is not None):
		sdf_dataframe = pose_checker(sdf_dataframe, protein_file, clash_cutoff, strain_cutoff, n_cpus)
	log_dataframe_length_change(initial_length, len(sdf_dataframe), "PoseChecker")
	initial_length = len(sdf_dataframe)

	if bust_poses:
		sdf_dataframe = pose_buster(sdf_dataframe, protein_file, n_cpus)
	log_dataframe_length_change(initial_length, len(sdf_dataframe), "PoseBusters")
	initial_length = len(sdf_dataframe)

	if classy_pose:
		sdf_dataframe = classy_pose_filter(sdf_dataframe, protein_file, classy_pose_model, n_cpus)
	log_dataframe_length_change(initial_length, len(sdf_dataframe), "ClassyPose")

	PandasTools.WriteSDF(sdf_dataframe,
							str(output_path),
							molColName="Molecule",
							idName="Pose ID",
							properties=list(sdf_dataframe.columns))
	return output_path
