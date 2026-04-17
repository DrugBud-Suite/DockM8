import shutil
import sys
import traceback
import uuid
from pathlib import Path
from functools import partial
import pandas as pd
from rdkit import RDLogger

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

# from scripts.rescoring.rescoring_functions.AAScore import AAScore  # Removed from publication
from scripts.rescoring.rescoring_functions.AD4 import AD4
from scripts.rescoring.rescoring_functions.CHEMPLP import CHEMPLP
from scripts.rescoring.rescoring_functions.ConvexPLR import ConvexPLR
from scripts.rescoring.rescoring_functions.GenScore import GenScore
from scripts.rescoring.rescoring_functions.cnn_affinity import CnnAffinity
from scripts.rescoring.rescoring_functions.cnn_score import CnnScore
from scripts.rescoring.rescoring_functions.gnina_affinity import GninaAffinity
from scripts.rescoring.rescoring_functions.gnina_all import GninaAll
from scripts.rescoring.rescoring_functions.KORP_PL import KORPL
from scripts.rescoring.rescoring_functions.LinF9 import LinF9
from scripts.rescoring.rescoring_functions.NNScore import NNScore
# from scripts.rescoring.rescoring_functions.PLECScore import PLECScore  # Removed from publication
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
 # "AAScore": {"class": AAScore, "column_name": "AAScore", "best_value": "min", "score_range": (100, -100)},  # Removed from publication
 "AD4": {"class": AD4, "column_name": "AD4", "best_value": "min", "score_range": (1000, -100)},
 "CHEMPLP": {"class": CHEMPLP, "column_name": "CHEMPLP", "best_value": "min", "score_range": (200, -200)},
 "ConvexPLR": {"class": ConvexPLR, "column_name": "ConvexPLR", "best_value": "max", "score_range": (-10, 10)},
 "GNINA-Affinity": {"class": GninaAffinity, "column_name": "GNINA-Affinity", "best_value": "min", "score_range": (100, -100)},
 "CNN-Score": {"class": CnnScore, "column_name": "CNN-Score", "best_value": "max", "score_range": (0, 1)},
 "CNN-Affinity": {"class": CnnAffinity, "column_name": "CNN-Affinity", "best_value": "max", "score_range": (0, 20)},
 "GNINA-All": {"class": GninaAll, "column_name": "GNINA-All", "best_value": "min", "score_range": (100, -100), "returns_columns": ["GNINA-Affinity", "CNN-Score", "CNN-Affinity"]},
 "GenScore-scoring": {"class": partial(GenScore, score_type="scoring"), "column_name": "GenScore-scoring", "best_value": "max", "score_range": (0, 100)},
 "GenScore-docking": {"class": partial(GenScore, score_type="docking"), "column_name": "GenScore-docking", "best_value": "max", "score_range": (0, 100)},
 "GenScore-balanced": {"class": partial(GenScore, score_type="balanced"), "column_name": "GenScore-balanced", "best_value": "max", "score_range": (0, 100)},
 "KORP-PL": {"class": KORPL, "column_name": "KORP-PL", "best_value": "min", "score_range": (200, -500)},
 "LinF9": {"class": LinF9, "column_name": "LinF9", "best_value": "min", "score_range": (50, -50)},
 "NNScore": {"class": NNScore, "column_name": "NNScore", "best_value": "max", "score_range": (0, 20)},
 # "PLECScore": {"class": PLECScore, "column_name": "PLECScore", "best_value": "max", "score_range": (0, 20)},  # Removed from publication
 "PLP": {"class": PLP, "column_name": "PLP", "best_value": "min", "score_range": (200, -200)},
 "RFScoreVS": {"class": RFScoreVS, "column_name": "RFScoreVS", "best_value": "max", "score_range": (5, 10)},
 "RTMScore": {"class": RTMScore, "column_name": "RTMScore", "best_value": "max", "score_range": (0, 100)},
 "SCORCH": {"class": SCORCH, "column_name": "SCORCH", "best_value": "max", "score_range": (0, 1)},
 "Vinardo": {"class": Vinardo, "column_name": "Vinardo", "best_value": "min", "score_range": (200, -20)}
}
# yapf: enable

def setup_temp_directory(base_name: str) -> Path:
    """Create organized temporary directory structure."""
    run_id = str(uuid.uuid4())[:8]
    base_temp = Path.home() / "dockm8_temp_files"
    temp_dir = base_temp / f"dockm8_{base_name.lower()}_{run_id}"
    
    # Create directory structure
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "input").mkdir(parents=True, exist_ok=True)
    (temp_dir / "working").mkdir(parents=True, exist_ok=True)
    (temp_dir / "output").mkdir(parents=True, exist_ok=True)
    
    return temp_dir

def load_poses(
    poses_input: Path | pd.DataFrame,
    requested_functions: list[str],
    output_file: Path | None = None
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, pd.Series]]:
    """Efficiently load poses and identify which need scoring - optimized version."""
    # Load poses using fast_sdf_loader if input is a file
    if isinstance(poses_input, Path):
        printlog(f"Loading poses from {poses_input}")
        if poses_input.suffix == ".sdf":
            poses_df = fast_load_sdf(
                poses_input,
                molColName="Molecule",
                idName="Pose ID",
                batch_size=100,
            )
        else:
            raise ValueError(f"Input poses must be in .sdf format. Path supplied was: {poses_input}")
    elif isinstance(poses_input, pd.DataFrame):
        poses_df = poses_input.copy()
    else:
        raise ValueError("Input poses must be in .sdf format or supplied as a dataframe.")

    # Get all pose IDs once and convert to set for O(1) lookups
    all_pose_ids = set(poses_df["Pose ID"].unique())
    
    # Initialize tracking dictionaries with sets for faster operations
    poses_to_score = {func: all_pose_ids.copy() for func in requested_functions}
    existing_scores = {}
    
    # Pre-compute column name mapping for faster lookups
    column_mapping = {func: RESCORING_FUNCTIONS[func]["column_name"]
                     for func in requested_functions}
    
    # Convert all potential scoring columns to numeric in a single operation
    potential_score_columns = set()
    for func, details in RESCORING_FUNCTIONS.items():
        if "returns_columns" in details:
            potential_score_columns.update(details["returns_columns"])
        else:
            potential_score_columns.add(details["column_name"])
    
    numeric_cols = [col for col in potential_score_columns if col in poses_df.columns]
    if numeric_cols:
        poses_df[numeric_cols] = poses_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Process input file first with vectorized operations
    print("Checking for scores in input file")
    
    # Process all functions efficiently
    for func in requested_functions:
        scoring_info = RESCORING_FUNCTIONS[func]
        
        # Handle functions that return multiple columns
        if "returns_columns" in scoring_info:
            for col in scoring_info["returns_columns"]:
                if col in poses_df.columns:
                    valid_mask = ~poses_df[col].isna()
                    if valid_mask.any():
                        valid_poses = set(poses_df.loc[valid_mask, "Pose ID"])
                        existing_scores[col] = poses_df.loc[valid_mask, ["Pose ID", col]].set_index("Pose ID")[col]
                        poses_to_score[func] -= valid_poses
        else:
            # Handle single column functions
            col_name = column_mapping.get(func, scoring_info["column_name"])
            if col_name in poses_df.columns:
                valid_mask = ~poses_df[col_name].isna()
                if valid_mask.any():
                    valid_poses = set(poses_df.loc[valid_mask, "Pose ID"])
                    existing_scores[col_name] = poses_df.loc[valid_mask, ["Pose ID", col_name]].set_index("Pose ID")[col_name]
                    poses_to_score[func] -= valid_poses
    
    # Check output file if it exists and we still have poses to score
    any_poses_to_score = any(poses_to_score.values())
    if any_poses_to_score and output_file and output_file.exists():
        try:
            # Determine needed columns to avoid loading unnecessary data
            needed_columns = {"Pose ID"}
            for func in requested_functions:
                scoring_info = RESCORING_FUNCTIONS[func]
                if "returns_columns" in scoring_info:
                    needed_columns.update(scoring_info["returns_columns"])
                else:
                    needed_columns.add(scoring_info["column_name"])
            
            # Efficient file loading
            if output_file.suffix == ".sdf":
                existing_results = fast_load_sdf(
                    output_file,
                    molColName=None,
                    idName="Pose ID",
                    batch_size=1000,
                )
            else:
                try:
                    # Try to load only necessary columns
                    existing_results = pd.read_csv(
                        output_file,
                        usecols=lambda x: x in needed_columns,
                        dtype={"Pose ID": str}
                    )
                except ValueError:
                    # Fallback if column selection fails
                    existing_results = pd.read_csv(output_file)
            
            # Convert numeric columns efficiently in a single pass
            numeric_cols = [col for col in existing_results.columns
                           if col in needed_columns and col != "Pose ID"]
            if numeric_cols:
                existing_results[numeric_cols] = existing_results[numeric_cols].apply(
                    pd.to_numeric, errors='coerce')
            
            # Process all functions with vectorized operations
            for func in requested_functions:
                scoring_info = RESCORING_FUNCTIONS[func]
                
                # Handle functions that return multiple columns
                if "returns_columns" in scoring_info:
                    for col in scoring_info["returns_columns"]:
                        if col in existing_results.columns:
                            valid_mask = ~existing_results[col].isna()
                            if valid_mask.any():
                                valid_poses = set(existing_results.loc[valid_mask, "Pose ID"])
                                valid_scores = existing_results.loc[valid_mask, ["Pose ID", col]].set_index("Pose ID")[col]
                                
                                if col in existing_scores:
                                    # Efficiently add new scores using index operations
                                    new_indices = valid_scores.index.difference(existing_scores[col].index)
                                    if not new_indices.empty:
                                        new_scores = valid_scores[new_indices]
                                        existing_scores[col] = pd.concat([existing_scores[col], new_scores])
                                else:
                                    existing_scores[col] = valid_scores
                                
                                poses_to_score[func] -= valid_poses
                else:
                    # Handle single column functions
                    col_name = column_mapping.get(func, scoring_info["column_name"])
                    if col_name in existing_results.columns:
                        valid_mask = ~existing_results[col_name].isna()
                        if valid_mask.any():
                            valid_poses = set(existing_results.loc[valid_mask, "Pose ID"])
                            valid_scores = existing_results.loc[valid_mask, ["Pose ID", col_name]].set_index("Pose ID")[col_name]
                            
                            if col_name in existing_scores:
                                # Efficiently merge scores using index operations
                                new_indices = valid_scores.index.difference(existing_scores[col_name].index)
                                if not new_indices.empty:
                                    new_scores = valid_scores[new_indices]
                                    existing_scores[col_name] = pd.concat([existing_scores[col_name], new_scores])
                            else:
                                existing_scores[col_name] = valid_scores
                            
                            poses_to_score[func] -= valid_poses
                
        except Exception as e:
            printlog(f"Error loading existing results from output file: {e}")
    
    # Convert sets back to lists for return
    poses_to_score = {func: list(poses) for func, poses in poses_to_score.items()}
    
    print(f"Total number of molecules to be processed: {len(poses_df)}")
    # Print the number of compounds to be scored for each function
    print("\nScoring summary:")
    for function in requested_functions:
        print(f"  {function}: {len(poses_to_score[function])} molecules need to be scored")

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
    """Apply a scoring function to the molecules - optimized version."""
    scoring_info = RESCORING_FUNCTIONS.get(function_name)
    if not scoring_info:
        printlog(f"Unknown scoring function: {function_name}")
        return pd.DataFrame()

    temp_dir = setup_temp_directory(f"score_{function_name.lower()}")
    
    try:
        # Create a filtered SDF if specific poses are requested
        if pose_ids_to_score:
            working_sdf = temp_dir / "working" / "filtered_poses.sdf"
            
            # Use a set for O(1) membership testing
            pose_ids_set = set(pose_ids_to_score)
            
            # Load SDF efficiently and filter with vectorized operations
            df = fast_load_sdf(sdf_path, "Molecule", "Pose ID", n_cpus=n_cpus)
            filtered_df = df[df["Pose ID"].isin(pose_ids_set)]
            
            fast_write_sdf(
                df=filtered_df[['Pose ID', 'Molecule']],
                output_path=working_sdf,
                molColName="Molecule",
                idName="Pose ID",
                batch_size=min(1000, max(100, len(filtered_df) // 20))
            )
            sdf_to_score = working_sdf
        else:
            sdf_to_score = sdf_path
        
        # Initialize and run scoring function
        scoring_function = scoring_info["class"](software_path=software)
        result = scoring_function.rescore(
            sdf_to_score,
            n_cpus=n_cpus,
            protein_file=protein_file,
            pocket_definition=pocket_definition
        )

        if isinstance(result, pd.DataFrame) and not result.empty:
            # For functions that return multiple columns, return all columns
            if "returns_columns" in scoring_info:
                return_columns = ["Pose ID"] + scoring_info["returns_columns"]
                return result[return_columns]
            else:
                # Only return necessary columns
                return result[["Pose ID", scoring_info["column_name"]]]
        
        return pd.DataFrame()

    except Exception as e:
        printlog(f"Error in scoring function {function_name}: {str(e)}\n{traceback.format_exc()}")
        return pd.DataFrame()
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def save_results(df: pd.DataFrame, output_file: Path) -> None:
    """Save results using optimized writers."""
    if output_file:
        # Save CSV without molecule column
        csv_output = df.drop(columns=["Molecule"], errors="ignore")
        
        # Use more efficient CSV writing
        csv_output.to_csv(
            output_file.with_suffix(".csv"),
            index=False,
            float_format="%.6g"  # More efficient number formatting
        )
        
        # Use fast_sdf_writer for SDF output
        if "Molecule" in df.columns:
            
            fast_write_sdf(
                df=df,
                output_path=output_file.with_suffix(".sdf"),
                molColName="Molecule",
                idName="Pose ID",
                properties=list(df.columns),
                batch_size=min(1000, max(100, len(df) // 20))
            )

def prepare_results(df: pd.DataFrame, n_cpus:int) -> pd.DataFrame:
    """Prepare final results with optimized operations."""
    if df.empty:
        return df

    # Add required columns efficiently
    if "ID" not in df.columns:
        df["ID"] = df["Pose ID"].str.split("_").str[0]
    
    # Only generate SMILES if needed
    if "SMILES" not in df.columns and "Molecule" in df.columns:
        from concurrent.futures import ThreadPoolExecutor

        from rdkit import Chem
        
        global generate_smiles
        # Use ThreadPoolExecutor for SMILES generation (I/O bound)
        def generate_smiles(mol):
            return Chem.MolToSmiles(mol) if mol is not None else None
        
        molecules = df["Molecule"].tolist()
        with ThreadPoolExecutor(max_workers=n_cpus) as executor:
            smiles_list = list(executor.map(generate_smiles, molecules))
        
        df["SMILES"] = smiles_list

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
    """Optimized main function for pose rescoring."""
    RDLogger.DisableLog("rdApp.*")
    
    # Setup organized temporary directory
    temp_dir = setup_temp_directory("rescore")
    
    try:
        optimized_functions = functions.copy()
            
        # Load initial data efficiently with optimized detection of already scored molecules
        poses_df, poses_to_score, existing_scores = load_poses(poses, functions, output_file)
        
        # Only process if there are poses to score
        any_poses_to_score = any(len(poses) > 0 for poses in poses_to_score.values())
        
        if any_poses_to_score:
            # Create working SDF with only needed columns
            working_sdf = temp_dir / "working" / "poses.sdf"
            
            # Select only essential columns for the working file
            essential_cols = ["Molecule", "Pose ID"]
            for col in poses_df.columns:
                if col not in essential_cols and not col.startswith("RDKit"):
                    essential_cols.append(col)
            # Optimize batch size based on dataframe size
            batch_size = min(1000, max(100, len(poses_df) // 20))
            
            fast_write_sdf(
                df=poses_df[essential_cols],
                output_path=working_sdf,
                molColName="Molecule",
                idName="Pose ID",
                properties=essential_cols,
                batch_size=batch_size
            )

            # Track results efficiently with indexed dataframe
            results_df = poses_df[["Pose ID"]].copy().set_index("Pose ID")
            
            # Score poses in optimized order
            for function in optimized_functions:
                pose_ids = poses_to_score[function]
                if not pose_ids:
                    continue  # Skip functions with no poses to score
                    
                result = apply_scoring_function(
                    working_sdf,
                    function,
                    protein_file,
                    pocket_definition,
                    software,
                    n_cpus,
                    pose_ids
                )
                
                if not result.empty:
                    scoring_info = RESCORING_FUNCTIONS[function]
                    # Handle functions that return multiple columns
                    if "returns_columns" in scoring_info:
                        result_indexed = result.set_index("Pose ID")
                        for col in scoring_info["returns_columns"]:
                            if col in result.columns:
                                # Explicitly extract the column as a Series before assignment
                                column_series = result_indexed[col].squeeze()
                                # Add the Series to results_df
                                results_df[col] = column_series
                    else:
                        # Add standard column to results_df
                        col_name = scoring_info["column_name"]
                        result_indexed = result.set_index("Pose ID")
                        results_df[col_name] = result_indexed[col_name]
            
            # Add existing scores efficiently
            for col, values in existing_scores.items():
                if col not in results_df.columns:
                    # Add entire column if it doesn't exist
                    results_df[col] = values
                else:
                    # Only update missing values
                    missing_mask = results_df[col].isna()
                    valid_indices = missing_mask.index.intersection(values.index)
                    if len(valid_indices) > 0:
                        results_df.loc[valid_indices, col] = values.loc[valid_indices]
            
            # Merge results back with original data
            results_df = results_df.reset_index()
            
            # Only include necessary columns from original dataframe
            needed_cols = {"Pose ID", "Molecule"}
            for col in poses_df.columns:
                if col not in results_df.columns or col in needed_cols:
                    needed_cols.add(col)
            
            # Merge efficiently using only necessary columns
            final_df = pd.merge(
                poses_df[list(needed_cols)],
                results_df,
                on="Pose ID",
                how="left"
            )
        else:
            # Create a results dataframe even when no scoring is needed
            final_df = poses_df.copy()
            
            # For each column in existing_scores
            for col, scores in existing_scores.items():
                # Create a temporary dataframe with just this column's scores
                temp_df = scores.to_frame().reset_index()
                temp_df.columns = ["Pose ID", f"{col}_temp"]
                
                # Merge with final_df
                final_df = pd.merge(
                    final_df,
                    temp_df,
                    on="Pose ID",
                    how="left"
                )
                
                # If column doesn't exist in final_df, add it
                if col not in final_df.columns:
                    final_df[col] = pd.NA
                
                # Fill in missing values from the temporary column
                mask = final_df[col].isna()
                if mask.any():
                    final_df.loc[mask, col] = final_df.loc[mask, f"{col}_temp"]
                
                # Drop the temporary column
                final_df = final_df.drop(columns=[f"{col}_temp"])
            
        # Prepare and save results
        final_results = prepare_results(final_df, n_cpus)
        if output_file:
            save_results(final_results, output_file)

        return final_results

    except Exception as e:
        printlog(f"Error during rescoring: {str(e)}")
        raise
    
    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
