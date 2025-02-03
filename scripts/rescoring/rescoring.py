import os
import shutil
import traceback
import uuid
from functools import partial
from pathlib import Path
import pandas as pd
from rdkit import RDLogger

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
from scripts.utilities.fast_sdf_loader import fast_load_sdf
from scripts.utilities.fast_sdf_writer import fast_write_sdf
from scripts.utilities.logging import printlog

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

def setup_temp_directory(base_name: str) -> Path:
    """Create organized temporary directory structure"""
    run_id = str(uuid.uuid4())[:8]
    base_temp = Path.home() / "dockm8_temp_files"
    temp_dir = base_temp / f"dockm8_{base_name.lower()}_{run_id}"
    
    # Create directory structure
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(temp_dir / "input", exist_ok=True)
    os.makedirs(temp_dir / "working", exist_ok=True)
    os.makedirs(temp_dir / "output", exist_ok=True)
    
    return temp_dir

def load_poses(
    poses_input: Path | pd.DataFrame,
    requested_functions: list[str],
    output_file: Path | None = None
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, pd.Series]]:
    """Efficiently load poses and identify which need scoring"""
    # Load poses using fast_sdf_loader if input is a file
    if isinstance(poses_input, Path):
        printlog(f"Loading poses from {poses_input}")
        if poses_input.suffix == ".sdf":
            poses_df = fast_load_sdf(
                poses_input,
                molColName="Molecule",
                idName="Pose ID",
                batch_size=100,
                required_props={"SMILES"}
            )
        else:
            raise ValueError(f"Input poses must be in .sdf format. Path supplied was: {poses_input}")
    elif isinstance(poses_input, pd.DataFrame):
        poses_df = poses_input.copy()
    else:
        raise ValueError("Input poses must be in .sdf format or supplied as a dataframe.")

    # Initialize tracking dictionaries
    all_pose_ids = poses_df["Pose ID"].tolist()
    poses_to_score = {func: all_pose_ids.copy() for func in requested_functions}
    existing_scores = {}

    # Check for existing scores in input
    for function in requested_functions:
        column_name = RESCORING_FUNCTIONS[function]["column_name"]
        if column_name in poses_df.columns:
            # Efficient type conversion
            poses_df[column_name] = pd.to_numeric(poses_df[column_name], errors='coerce')
            
            # Find valid scores using vectorized operations
            mask = ~poses_df[column_name].isna()
            if mask.any():
                valid_scores = poses_df.loc[mask, ["Pose ID", column_name]].set_index("Pose ID")[column_name]
                existing_scores[column_name] = valid_scores
                # Update poses to score using set operations
                poses_to_score[function] = list(set(poses_to_score[function]) - set(valid_scores.index))

    # Check output file if it exists
    if output_file and output_file.exists():
        try:
            if output_file.suffix == ".sdf":
                existing_results = fast_load_sdf(
                    output_file,
                    molColName=None,
                    idName="Pose ID"
                )
            else:
                existing_results = pd.read_csv(output_file)
            
            for function in requested_functions:
                column_name = RESCORING_FUNCTIONS[function]["column_name"]
                if column_name in existing_results.columns:
                    existing_results[column_name] = pd.to_numeric(existing_results[column_name], errors='coerce')
                    mask = ~existing_results[column_name].isna()
                    if mask.any():
                        valid_scores = existing_results.loc[mask, ["Pose ID", column_name]].set_index("Pose ID")[column_name]
                        if column_name in existing_scores:
                            # Use set operations for efficient updates
                            new_scores = valid_scores[~valid_scores.index.isin(existing_scores[column_name].index)]
                            if not new_scores.empty:
                                existing_scores[column_name] = pd.concat([existing_scores[column_name], new_scores])
                        else:
                            existing_scores[column_name] = valid_scores
                        
                        poses_to_score[function] = list(set(poses_to_score[function]) - set(valid_scores.index))
                    
        except Exception as e:
            printlog(f"Error loading existing results from output file: {e}")

    return poses_df, poses_to_score, existing_scores

def apply_scoring_function(
    sdf_path: Path,
    function_name: str,
    protein_file: Path,
    pocket_definition: dict,
    software: Path,
    n_cpus: int,
    pose_ids_to_score: list[str] | None = None
) -> pd.DataFrame:
    """Apply a scoring function to the molecules"""
    scoring_info = RESCORING_FUNCTIONS.get(function_name)
    if not scoring_info:
        printlog(f"Unknown scoring function: {function_name}")
        return pd.DataFrame()

    temp_dir = setup_temp_directory(f"score_{function_name.lower()}")
    
    try:
        # Create a filtered SDF if specific poses are requested
        if pose_ids_to_score:
            working_sdf = temp_dir / "working" / "filtered_poses.sdf"
            df = fast_load_sdf(sdf_path, "Molecule", "Pose ID")
            filtered_df = df[df["Pose ID"].isin(pose_ids_to_score)]
            fast_write_sdf(
                df=filtered_df,
                output_path=working_sdf,
                molColName="Molecule",
                idName="Pose ID",
                properties=list(filtered_df.columns)
            )
            sdf_to_score = working_sdf
        else:
            sdf_to_score = sdf_path
        # Initialize and run scoring function
        scoring_function = scoring_info["class"](software_path=software)
        result = scoring_function.rescore(
            str(sdf_to_score),
            n_cpus=n_cpus,
            protein_file=str(protein_file),
            pocket_definition=pocket_definition
        )

        if isinstance(result, pd.DataFrame) and not result.empty:
            return result[["Pose ID", scoring_info["column_name"]]]
        
        return pd.DataFrame()

    except Exception as e:
        printlog(f"Error in scoring function {function_name}: {str(e)}\n{traceback.format_exc()}")
        return pd.DataFrame()
    
    finally:
        pass
        #shutil.rmtree(temp_dir, ignore_errors=True)

def save_results(df: pd.DataFrame, output_file: Path) -> None:
    """Save results using optimized writers"""
    if output_file:
        # Save CSV without molecule column
        csv_output = df.drop(columns=["Molecule"], errors="ignore")
        csv_output.to_csv(output_file.with_suffix(".csv"), index=False)
        
        # Use fast_sdf_writer for SDF output
        if "Molecule" in df.columns:
            fast_write_sdf(
                df=df,
                output_path=output_file.with_suffix(".sdf"),
                molColName="Molecule",
                idName="Pose ID",
                properties=list(df.columns),
                batch_size=100
            )

def prepare_results(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare final results with optimized operations"""
    if df.empty:
        return df

    # Add required columns efficiently
    if "ID" not in df.columns:
        df["ID"] = df["Pose ID"].str.split("_").str[0]
    
    # Only generate SMILES if needed
    if "SMILES" not in df.columns and "Molecule" in df.columns:
        from rdkit import Chem
        df["SMILES"] = df["Molecule"].apply(lambda x: Chem.MolToSmiles(x) if x is not None else None)

    # Optimize column order
    result_columns = ["Pose ID", "ID", "SMILES", "Molecule"]
    additional_cols = [col for col in df.columns if col not in result_columns]
    result_columns.extend(additional_cols)
    
    # Efficient column selection and sorting
    result_df = df[result_columns].sort_values("Pose ID")

    # Convert numeric columns efficiently
    non_numeric = {"Pose ID", "ID", "SMILES", "Molecule"}
    numeric_cols = [col for col in result_df.columns if col not in non_numeric]
    
    if numeric_cols:
        result_df[numeric_cols] = result_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    return result_df

def rescore_poses(
    protein_file: Path,
    pocket_definition: dict,
    software: Path,
    poses: Path | pd.DataFrame,
    functions: list[str],
    n_cpus: int,
    output_file: Path | None = None,
) -> pd.DataFrame:
    """Optimized main function for pose rescoring"""
    RDLogger.DisableLog("rdApp.*")
    
    # Setup organized temporary directory
    temp_dir = setup_temp_directory("rescore")
    
    try:
        # Load initial data efficiently
        poses_df, poses_to_score, existing_scores = load_poses(poses, functions, output_file)
        
        # Create working files using fast writer
        working_sdf = temp_dir / "working" / "poses.sdf"
        fast_write_sdf(
            df=poses_df,
            output_path=working_sdf,
            molColName="Molecule",
            idName="Pose ID",
            properties=list(poses_df.columns),
            batch_size=100
        )

        # Score poses that need it
        for function in functions:
            pose_ids_to_score = poses_to_score[function]
            if pose_ids_to_score:
                result = apply_scoring_function(
                    working_sdf,
                    function,
                    protein_file,
                    pocket_definition,
                    software,
                    n_cpus,
                    pose_ids_to_score
                )
                if not result.empty:
                    # Efficient merge using pose IDs
                    poses_df = pd.merge(
                        poses_df,
                        result[["Pose ID", RESCORING_FUNCTIONS[function]["column_name"]]],
                        on="Pose ID",
                        how="left"
                    )

        # Merge existing scores efficiently
        if existing_scores:
            existing_scores_df = pd.DataFrame(existing_scores)
            existing_scores_df.index.name = "Pose ID"
            poses_df = poses_df.set_index("Pose ID").combine_first(existing_scores_df).reset_index()

        # Prepare and save results
        final_results = prepare_results(poses_df)
        if output_file:
            save_results(final_results, output_file)

        return final_results

    except Exception as e:
        printlog(f"Error during rescoring: {str(e)}")
        raise
    
    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
