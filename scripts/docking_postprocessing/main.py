import os
import sys
from pathlib import Path

from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
dockm8_path = next(
    (p / "DockM8" for p in Path(__file__).resolve().parents if (p / "DockM8").is_dir()),
    None,
)
sys.path.append(str(dockm8_path))

import warnings

from scripts.docking_postprocessing.posebusters.posebusters import pose_buster
from scripts.docking_postprocessing.posecheck.posecheck import pose_checker
from scripts.utilities.utilities import parallel_SDF_loader, printlog

warnings.filterwarnings("ignore")


def docking_postprocessing(
    input_sdf: Path,
    output_path: Path,
    protein_file: Path,
    bust_poses: bool,
    strain_cutoff: int,
    clash_cutoff: int,
    n_cpus: int,
):
    """
    Perform postprocessing on docking results.

    Args:
        input_sdf (Path): Path to the input SDF file containing docking results.
        output_path (Path): Path to save the postprocessed SDF file.
        protein_file (Path): Path to the protein file used for docking.
        bust_poses (bool): Flag indicating whether to apply pose busting.
        strain_cutoff (int): Cutoff value for strain energy. Poses with strain energy above this cutoff will be removed.
        clash_cutoff (int): Cutoff value for clash score. Poses with clash score above this cutoff will be removed.
        n_cpus (int): Number of CPUs to use for parallel processing.

    Returns:
        Path: Path to the postprocessed SDF file.
    """
    printlog("Postprocessing docking results...")
    sdf_dataframe = parallel_SDF_loader(
        input_sdf, molColName="Molecule", idName="Pose ID", n_cpus=n_cpus
    )
    starting_length = len(sdf_dataframe)
    if (strain_cutoff is not None) and (clash_cutoff is not None):
        checked_poses = pose_checker(
            sdf_dataframe, protein_file, clash_cutoff, strain_cutoff, n_cpus
        )
        sdf_dataframe = checked_poses
        step1_length = len(sdf_dataframe)
        printlog(f"Removed {starting_length - step1_length} poses during PoseChecker postprocessing.")
    else:
        step1_length = starting_length
    if bust_poses:
        sdf_dataframe = pose_buster(sdf_dataframe, protein_file, n_cpus)
        step2_length = len(sdf_dataframe)
        if step1_length:
            printlog(f"Removed {step1_length - step2_length} poses during PoseBusters postprocessing.")
        else:
            printlog(f"Removed {starting_length - step2_length} poses during PoseBusters postprocessing.")
    PandasTools.WriteSDF(
        sdf_dataframe,
        str(output_path),
        molColName="Molecule",
        idName="Pose ID",
        properties=list(sdf_dataframe.columns),
    )
    return output_path
