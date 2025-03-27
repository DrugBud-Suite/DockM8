import sys
import warnings
from pathlib import Path

import pandas as pd

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring import RESCORING_FUNCTIONS, rescore_poses
from scripts.utilities.fast_sdf_loader import fast_load_sdf
from scripts.utilities.fast_sdf_writer import fast_write_sdf

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def select_poses(
    selection_method: str,
    poses_input: Path | pd.DataFrame,
    protein_file: Path,
    pocket_definition: dict,
    software: Path,
    output_file: Path | None = None,
    n_cpus: int = 1,
) -> pd.DataFrame:
    """Select poses based on specified method and return results as DataFrame.
    
    Args:
        selection_method (str): Method to use for pose selection (e.g. "bestpose", "GNINA", etc)
        poses_input (Path | pd.DataFrame): Input poses as SDF file path or DataFrame
        protein_file (Path): Path to protein structure file
        pocket_definition (dict): Dictionary containing pocket definition
        software (Path): Path to software directory
        output_file (Path, optional): Path to save output poses. If None, only returns DataFrame
        n_cpus (int): Number of CPU cores to use
        
    Returns:
        pd.DataFrame: Selected poses with scoring data
        
    Raises:
        ValueError: If invalid selection method or missing required data
        RuntimeError: If pose selection fails
    """
    try:
        # Load poses if needed using fast_sdf_loader
        if isinstance(poses_input, Path):
            all_poses = fast_load_sdf(
                poses_input,
                molColName="Molecule",
                idName="Pose ID",
                batch_size=100,
                n_cpus = n_cpus
            )
        else:
            all_poses = poses_input.copy()

        # Verify Pose ID exists
        if "Pose ID" not in all_poses.columns:
            raise ValueError("Error during pose selection: Input data must contain 'Pose ID' column")

        all_poses["ID"] = all_poses["Pose ID"].str.split("_").str[0].astype(str)

        # Select poses based on method
        if selection_method == "bestpose":
            # Add pose numbering columns
            all_poses["Pose_Number"] = all_poses["Pose ID"].str.split("_").str[2].astype(int)
            all_poses["Docking_program"] = all_poses["Pose ID"].str.split("_").str[1].astype(str)
            min_pose_indices = all_poses.groupby(["ID", "Docking_program"])["Pose_Number"].idxmin()
            selected_poses = all_poses.loc[min_pose_indices]
        elif selection_method in RESCORING_FUNCTIONS:
            score_column = RESCORING_FUNCTIONS[selection_method]["column_name"]
            best_value = RESCORING_FUNCTIONS[selection_method]["best_value"]
            
            if score_column in all_poses.columns and not all_poses[score_column].isna().any():
                # All scores present and valid, no need to rescore
                poses_with_scores = all_poses
            else:
                # Some scores missing or invalid, need to rescore
                poses_with_scores = rescore_poses(
                    protein_file=protein_file,
                    pocket_definition=pocket_definition,
                    software=software,
                    poses=all_poses,
                    functions=[selection_method],
                    n_cpus=n_cpus,
                    output_file=None
                )
            # Select best poses based on scoring function
            if best_value == "min":
                best_pose_indices = poses_with_scores.groupby("ID")[score_column].idxmin()
            elif best_value == "max":
                best_pose_indices = poses_with_scores.groupby("ID")[score_column].idxmax()
            else:
                raise ValueError(f"Invalid best_value '{best_value}' for scoring function {selection_method}")
                
            selected_poses = poses_with_scores.loc[best_pose_indices]

        # Write output if path provided
        if output_file is not None:
            fast_write_sdf(
                df=selected_poses,
                output_path=output_file,
                molColName="Molecule",
                idName="Pose ID",
                properties=list(selected_poses.columns),
                n_cpus = n_cpus
            )

        return selected_poses

    except Exception as e:
        print(e)
        raise RuntimeError(f"Pose selection failed: {str(e)}") from e
