import os
import shutil
import sys
import tempfile
import warnings
from functools import partial
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools
import traceback

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring_functions.AAScore import AAScore
from scripts.rescoring.rescoring_functions.AD4 import AD4
from scripts.rescoring.rescoring_functions.CHEMPLP import CHEMPLP
from scripts.rescoring.rescoring_functions.ConvexPLR import ConvexPLR
from scripts.rescoring.rescoring_functions.GenScore import GenScore
from scripts.rescoring.rescoring_functions.gnina import Gnina
from scripts.rescoring.rescoring_functions.KORP_PL import KORPL
from scripts.rescoring.rescoring_functions.LinF9 import LinF9
from scripts.rescoring.rescoring_functions.NNScore import NNScore
from scripts.rescoring.rescoring_functions.PLECScore import PLECScore
from scripts.rescoring.rescoring_functions.PLP import PLP
from scripts.rescoring.rescoring_functions.RFScoreVS import RFScoreVS
from scripts.rescoring.rescoring_functions.RTMScore import RTMScore
from scripts.rescoring.rescoring_functions.SCORCH import SCORCH
from scripts.rescoring.rescoring_functions.vinardo import Vinardo
from scripts.utilities.logging import printlog
from scripts.utilities.utilities import parallel_SDF_loader

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# yapf: disable
RESCORING_FUNCTIONS = {
 "AAScore": {"class": AAScore, "column_name": "AAScore", "best_value": "min", "score_range": (100, -100)},
 "AD4": {"class": AD4, "column_name": "AD4", "best_value": "min", "score_range": (1000, -100)},
 "CHEMPLP": {"class": CHEMPLP, "column_name": "CHEMPLP", "best_value": "min", "score_range": (200, -200)},
 "ConvexPLR": {"class": ConvexPLR, "column_name": "ConvexPLR", "best_value": "max", "score_range": (-10, 10)},
 "GNINA-Affinity": {"class": partial(Gnina, score_type="affinity"), "column_name": "GNINA-Affinity", "best_value": "min", "score_range": (100, -100)},
 "CNN-Score": {"class": partial(Gnina, score_type="cnn_score"), "column_name": "CNN-Score", "best_value": "max", "score_range": (0, 1)},
 "CNN-Affinity": {"class": partial(Gnina, score_type="cnn_affinity"), "column_name": "CNN-Affinity", "best_value": "max", "score_range": (0, 20)},
 "GenScore-scoring": {"class": partial(GenScore, score_type="scoring"), "column_name": "GenScore-scoring", "best_value": "max", "score_range": (0, 100)},
 "GenScore-docking": {"class": partial(GenScore, score_type="docking"), "column_name": "GenScore-docking", "best_value": "max", "score_range": (0, 100)},
 "GenScore-balanced": {"class": partial(GenScore, score_type="balanced"), "column_name": "GenScore-balanced", "best_value": "max", "score_range": (0, 100)},
 "KORP-PL": {"class": KORPL, "column_name": "KORP-PL", "best_value": "min", "score_range": (200, -500)},
 "LinF9": {"class": LinF9, "column_name": "LinF9", "best_value": "min", "score_range": (50, -50)},
 "NNScore": {"class": NNScore, "column_name": "NNScore", "best_value": "max", "score_range": (0, 20)},
 "PLECScore": {"class": PLECScore, "column_name": "PLECScore", "best_value": "max", "score_range": (0, 20)},
 "PLP": {"class": PLP, "column_name": "PLP", "best_value": "min", "score_range": (200, -200)},
 "RFScoreVS": {"class": RFScoreVS, "column_name": "RFScoreVS", "best_value": "max", "score_range": (5, 10)},
 "RTMScore": {"class": RTMScore, "column_name": "RTMScore", "best_value": "max", "score_range": (0, 100)},
 "SCORCH": {"class": SCORCH, "column_name": "SCORCH", "best_value": "max", "score_range": (0, 1)},
 "Vinardo": {"class": Vinardo, "column_name": "Vinardo", "best_value": "min", "score_range": (200, -20)}
}
# yapf: enable

def create_temp_dir(name: str) -> Path:
    """
    Creates a temporary directory for the scoring function.

    Args:
    name (str): The name of the scoring function.

    Returns:
    Path: The path to the temporary directory.
    """
    os.makedirs(Path.home() / "dockm8_temp_files", exist_ok=True)
    return Path(tempfile.mkdtemp(dir=Path.home() / "dockm8_temp_files", prefix=f"dockm8_{name}_"))

from pathlib import Path

def load_poses(poses_input: Path | pd.DataFrame, requested_functions: list[str], output_file: Path | None = None) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, pd.Series]]:
    """Load poses and identify which poses need scoring for each function.

    Args:
        poses_input: Path to SDF file or DataFrame containing poses
        requested_functions: List of scoring functions to process
        output_file: Path to the existing output file (optional)

    Returns:
        tuple containing:
        - DataFrame with poses
        - Dictionary mapping functions to list of Pose IDs that need scoring
        - Dictionary of existing valid scores
    """
    # Load poses from input
    if isinstance(poses_input, Path):
        printlog(f"Loading poses from {poses_input}")
        if poses_input.suffix == ".sdf":
            poses_df = parallel_SDF_loader(poses_input, molColName="Molecule", idName="Pose ID", SMILES="SMILES")
        else:
            raise ValueError(f"Input poses must be in .sdf format. Path supplied was: {poses_input}")
    elif isinstance(poses_input, pd.DataFrame):
        poses_df = poses_input.copy()
    else:
        raise ValueError("Input poses must be in .sdf format or supplied as a dataframe.")

    all_pose_ids = poses_df["Pose ID"].tolist()
    poses_to_score = {func: all_pose_ids.copy() for func in requested_functions}
    existing_scores = {}

    # Check for existing scores in input file
    for function in requested_functions:
        column_name = RESCORING_FUNCTIONS[function]["column_name"]
        if column_name in poses_df.columns:
            # Convert score column to numeric if it exists
            poses_df[column_name] = pd.to_numeric(poses_df[column_name], errors='coerce')
            
            # Find poses that don't need scoring (have valid scores)
            valid_scores = poses_df[~poses_df[column_name].isna()].set_index("Pose ID")[column_name]
            if not valid_scores.empty:
                existing_scores[column_name] = valid_scores
                # Remove poses with valid scores from the to-score list
                poses_to_score[function] = [pid for pid in poses_to_score[function]
                                          if pid not in valid_scores.index]

    # If output file exists, check for additional existing scores
    if output_file and output_file.exists():
        try:
            if output_file.suffix == ".sdf":
                existing_results = PandasTools.LoadSDF(output_file, molColName=None, idName="Pose ID")
            else:
                existing_results = pd.read_csv(output_file)
            
            # Process each scoring function
            for function in requested_functions:
                column_name = RESCORING_FUNCTIONS[function]["column_name"]
                if column_name in existing_results.columns:
                    # Convert score column to numeric
                    existing_results[column_name] = pd.to_numeric(existing_results[column_name], errors='coerce')
                    
                    # Find additional valid scores from output file
                    valid_scores = existing_results[~existing_results[column_name].isna()].set_index("Pose ID")[column_name]
                    if not valid_scores.empty:
                        # Update existing scores, preferring input file scores if they exist
                        if column_name in existing_scores:
                            valid_scores = valid_scores[~valid_scores.index.isin(existing_scores[column_name].index)]
                        if not valid_scores.empty:
                            if column_name in existing_scores:
                                existing_scores[column_name] = pd.concat([existing_scores[column_name], valid_scores])
                            else:
                                existing_scores[column_name] = valid_scores
                            
                            # Remove poses with valid scores from the to-score list
                            poses_to_score[function] = [pid for pid in poses_to_score[function]
                                                      if pid not in valid_scores.index]
                    
        except Exception as e:
            printlog(f"Error loading existing results from output file: {e}. Will use scores from input file only.")

    return poses_df, poses_to_score, existing_scores

def create_working_files(poses_df: pd.DataFrame, temp_dir: Path) -> Path:
    """Create temporary SDF file from poses DataFrame"""
    sdf_path = temp_dir / "temp_poses.sdf"
    PandasTools.WriteSDF(poses_df, sdf_path, molColName="Molecule", idName="Pose ID", properties=list(poses_df.columns))
    return sdf_path

def apply_scoring_function(
    sdf_path: Path, function_name: str, protein_file: Path, pocket_definition: dict, software: Path, n_cpus: int, pose_ids_to_score: list[str] | None = None
) -> pd.DataFrame:
    """Apply a single scoring function and return results"""
    scoring_info = RESCORING_FUNCTIONS.get(function_name)
    if not scoring_info:
        printlog(f"Unknown scoring function: {function_name}")
        return pd.DataFrame()

    if pose_ids_to_score:
        temp_sdf_path = Path(tempfile.mkdtemp(prefix="dockm8_temp_sdf_")) / "filtered_poses.sdf"
        mols = [mol for mol in Chem.SDMolSupplier(str(sdf_path), sanitize=False) if mol is not None and mol.GetProp("_Name") in pose_ids_to_score]
        writer = Chem.SDWriter(str(temp_sdf_path))
        for mol in mols:
            writer.write(mol)
        writer.close()
        sdf_path_to_use = temp_sdf_path
    else:
        sdf_path_to_use = sdf_path

    scoring_function = scoring_info["class"](software_path=software)
    try:
        return scoring_function.rescore(
            sdf_path_to_use, n_cpus, protein_file=protein_file, pocket_definition=pocket_definition
        )
    except Exception as e:
        printlog(f"Failed to apply {function_name}: {e}")
        return pd.DataFrame()
    finally:
        scoring_function.cleanup()
        if pose_ids_to_score:
            shutil.rmtree(sdf_path_to_use.parent, ignore_errors=True)

def prepare_results(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare final results DataFrame with required columns and formatting"""
    if df.empty:
        return df

    # Add required columns
    if "ID" not in df.columns:
        df["ID"] = df["Pose ID"].str.split("_").str[0]
    if "SMILES" not in df.columns and "Molecule" in df.columns:
        df["SMILES"] = df["Molecule"].apply(lambda x: Chem.MolToSmiles(x) if x is not None else None)

    # Select and order columns
    result_columns = ["Pose ID", "ID", "SMILES", "Molecule"] + [
        col for col in df.columns if col not in ["Pose ID", "ID", "SMILES", "Molecule"]
    ]
    result_df = df[result_columns].sort_values("Pose ID")

    # Define non-numeric columns
    non_numeric_columns = ["Pose ID", "ID", "SMILES", "Molecule"]

    # Convert only numeric columns
    for col in result_df.columns:
        if col not in non_numeric_columns:
            try:
                result_df[col] = pd.to_numeric(result_df[col].values, errors="coerce")
            except TypeError as e:
                print(f"Could not convert column {col}: {e}")

    return result_df

def save_results(df: pd.DataFrame, output_file: Path) -> None:
    """Save results to both CSV and SDF formats"""
    if output_file:
        csv_output = df.drop(columns=["Molecule"], errors="ignore")
        csv_output.to_csv(output_file.with_suffix(".csv"), index=False)
        if "Molecule" in df.columns:
            sdf_path = output_file.with_suffix(".sdf")
            PandasTools.WriteSDF(
                df, str(sdf_path), molColName="Molecule", idName="Pose ID", properties=list(df.columns)
            )

def rescore_poses(
    protein_file: Path,
    pocket_definition: dict,
    software: Path,
    poses: Path | pd.DataFrame,
    functions: list[str],
    n_cpus: int,
    output_file: Path | None = None,
) -> pd.DataFrame:
    """Main function to orchestrate pose rescoring process"""
    RDLogger.DisableLog("rdApp.*")
    temp_dir = Path(tempfile.mkdtemp(prefix="dockm8_rescore_"))

    try:
        # Load initial data and check for existing scores
        poses_df, poses_to_score, existing_scores = load_poses(poses, functions, output_file)

        # Create working files
        temp_dir = Path(tempfile.mkdtemp(prefix="dockm8_rescore_"))
        sdf_path = create_working_files(poses_df, temp_dir)

        try:
            # Score only the poses that need it for each function
            for function in functions:
                pose_ids_to_score = poses_to_score[function]
                
                if pose_ids_to_score:
                    result = apply_scoring_function(
                        sdf_path, function, protein_file, pocket_definition, software, n_cpus, pose_ids_to_score
                    )
                    if not result.empty:
                        poses_df = pd.merge(poses_df, result, on="Pose ID", how="left")

            # Merge existing scores back
            if existing_scores:
                existing_scores_df = pd.DataFrame(existing_scores)
                existing_scores_df.index.name = "Pose ID"
                poses_df = poses_df.set_index("Pose ID").combine_first(existing_scores_df).reset_index()

            # Prepare and save final results
            final_results = prepare_results(poses_df)
            if output_file:
                save_results(final_results, output_file)

        except Exception as e:
            error_traceback = traceback.format_exc()
            printlog(f"Error rescoring poses:\n{error_traceback}")
            raise Exception(f"Rescoring error: {str(e)}\n\nTraceback:\n{error_traceback}") from e
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def extract_pose_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extract metadata from Pose IDs into separate columns"""
    df = df.copy()
    df["Pose_Number"] = df["Pose ID"].str.split("_").str[2].astype(int)
    df["Docking_program"] = df["Pose ID"].str.split("_").str[1].astype(str)
    df["ID"] = df["Pose ID"].str.split("_").str[0].astype(str)
    return df

def find_best_poses(df: pd.DataFrame, score_column: str, best_value: str) -> pd.DataFrame:
    """Find best poses based on scoring criteria"""
    if best_value == "min":
        best_scores = df.groupby("ID").agg({score_column: "min"}).reset_index()
    else:
        best_scores = df.groupby("ID").agg({score_column: "max"}).reset_index()

    best_poses = pd.merge(df, best_scores, on=["ID", score_column], how="inner")
    best_poses = best_poses.groupby("ID").first().reset_index()
    return best_poses[["Pose ID", "ID", score_column]]

def rescore_docking(
    poses: Path | pd.DataFrame, protein_file: Path, pocket_definition: dict, software: Path, function: str, n_cpus: int
) -> pd.DataFrame:
    """Rescore docking poses and find best pose for each molecule"""
    RDLogger.DisableLog("rdApp.*")
    temp_dir = Path(tempfile.mkdtemp(prefix="dockm8_rescore_docking_"))

    try:
        # Reuse poses loading function
        poses_df = load_poses(poses, [function])  # Only need to load for the specific function
        sdf_path = create_working_files(poses_df, temp_dir)

        # Get scoring function configuration
        scoring_info = RESCORING_FUNCTIONS.get(function)
        if scoring_info is None:
            raise ValueError(f"Unknown scoring function: {function}")

        # Apply scoring function
        score_df = apply_scoring_function(sdf_path, function, protein_file, pocket_definition, software, n_cpus)

        if score_df.empty:
            return pd.DataFrame()

        # Process results to find best poses
        score_df = extract_pose_metadata(score_df)
        best_poses = find_best_poses(score_df, scoring_info["column_name"], scoring_info["best_value"])

        return best_poses

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
