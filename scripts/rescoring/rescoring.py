import os
import sys
import time
import warnings
from pathlib import Path
from typing import List

import pandas as pd
from rdkit import RDLogger
from tqdm import tqdm

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts"
                     for p in Path(__file__).resolve().parents
                     if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring_functions.AAScore import AAScore_rescoring
from scripts.rescoring.rescoring_functions.AD4 import AD4_rescoring
from scripts.rescoring.rescoring_functions.CHEMPLP import chemplp_rescoring
from scripts.rescoring.rescoring_functions.ConvexPLR import ConvexPLR_rescoring
from scripts.rescoring.rescoring_functions.gnina import gnina_rescoring
from scripts.rescoring.rescoring_functions.KORP_PL import KORPL_rescoring
from scripts.rescoring.rescoring_functions.LinF9 import LinF9_rescoring
from scripts.rescoring.rescoring_functions.NNScore import oddt_nnscore_rescoring
from scripts.rescoring.rescoring_functions.PLECScore import oddt_plecscore_rescoring
from scripts.rescoring.rescoring_functions.PLP import plp_rescoring
from scripts.rescoring.rescoring_functions.RFScoreVS import rfscorevs_rescoring
from scripts.rescoring.rescoring_functions.RTMScore import RTMScore_rescoring
from scripts.rescoring.rescoring_functions.SCORCH import SCORCH_rescoring
from scripts.rescoring.rescoring_functions.vinardo import vinardo_rescoring
from scripts.utilities.utilities import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# add new scoring functions here!
# Dict key: (function, column_name, min or max ordering, min value for scaled standardisation, max value for scaled standardisation)
RESCORING_FUNCTIONS = {
    "GNINA-Affinity": {
        "function": gnina_rescoring,
        "column_name": "GNINA-Affinity",
        "best_value": "min",
        "range": (100, -100),},
    "CNN-Score": {
        "function": gnina_rescoring,
        "column_name": "CNN-Score",
        "best_value": "max",
        "range": (0, 1)},
    "CNN-Affinity": {
        "function": gnina_rescoring,
        "column_name": "CNN-Affinity",
        "best_value": "max",
        "range": (0, 20)},
    "Vinardo": {
        "function": vinardo_rescoring,
        "column_name": "Vinardo",
        "best_value": "min",
        "range": (200, 20)},
    "AD4": {
        "function": AD4_rescoring,
        "column_name": "AD4",
        "best_value": "min",
        "range": (100, -100)},
    "RFScoreVS": {
        "function": rfscorevs_rescoring,
        "column_name": "RFScoreVS",
        "best_value": "max",
        "range": (5, 10)},
    #'RFScoreVS2':       {'function': rfscorevs_rescoring2,    'column_name': 'RFScoreVS',      'best_value': 'max', 'range': (5, 10)},
    "PLP": {
        "function": plp_rescoring,
        "column_name": "PLP",
        "best_value": "min",
        "range": (200, -200)},
    "CHEMPLP": {
        "function": chemplp_rescoring,
        "column_name": "CHEMPLP",
        "best_value": "min",
        "range": (200, -200)},
    "NNScore": {
        "function": oddt_nnscore_rescoring,
        "column_name": "NNScore",
        "best_value": "max",
        "range": (0, 20)},
    "PLECScore": {
        "function": oddt_plecscore_rescoring,
        "column_name": "PLECScore",
        "best_value": "max",
        "range": (0, 20),},
    "LinF9": {
        "function": LinF9_rescoring,
        "column_name": "LinF9",
        "best_value": "min",
        "range": (100, -100)},
    "AAScore": {
        "function": AAScore_rescoring,
        "column_name": "AAScore",
        "best_value": "max",
        "range": (100, -100)},
    "SCORCH": {
        "function": SCORCH_rescoring,
        "column_name": "SCORCH",
        "best_value": "max",
        "range": (0, 1)},
    "RTMScore": {
        "function": RTMScore_rescoring,
        "column_name": "RTMScore",
        "best_value": "max",
        "range": (0, 100)},
    "KORP-PL": {
        "function": KORPL_rescoring,
        "column_name": "KORP-PL",
        "best_value": "min",
        "range": (200, -1000)},
    "ConvexPLR": {
        "function": ConvexPLR_rescoring,
        "column_name": "ConvexPLR",
        "best_value": "max",
        "range": (-10, 10)},}


def rescore_poses(w_dir: Path, protein_file: Path, software: Path,
                  clustered_sdf: Path, functions: List[str],
                  n_cpus: int) -> None:
    """
    Rescores ligand poses using the specified software and scoring functions. The function splits the input SDF file into
    smaller files, and then runs the specified software on each of these files in parallel. The results are then combined into a single
    Pandas dataframe and saved to a CSV file.

    Args:
        w_dir (str): The working directory.
        protein_file (str): The path to the protein file.
        pocket_definition (dict): A dictionary containing the pocket center and size.
        software (str): The path to the software to be used for rescoring.
        clustered_sdf (str): The path to the input SDF file containing the clustered poses.
        functions (List[str]): A list of the scoring functions to be used.
        n_cpus (int): The number of CPUs to use for parallel execution.

    Returns:
        None
    """
    RDLogger.DisableLog("rdApp.*")
    tic = time.perf_counter()
    rescoring_folder_name = Path(clustered_sdf).stem
    rescoring_folder = w_dir / f"rescoring_{rescoring_folder_name}"
    (rescoring_folder).mkdir(parents=True, exist_ok=True)

    skipped_functions = []
    for function in functions:
        function_info = RESCORING_FUNCTIONS.get(function)
        if not (rescoring_folder / f"{function}_rescoring" /
                f"{function}_scores.csv").is_file():
            try:
                function_info["function"](
                    clustered_sdf,
                    n_cpus,
                    function_info["column_name"],
                    protein_file=protein_file,
                    software=software,
                    rescoring_folder=rescoring_folder,
                )
            except Exception as e:
                printlog(e)
                printlog(f"Failed for {function}")
        else:
            skipped_functions.append(function)
    if skipped_functions:
        printlog(f'Skipping functions: {", ".join(skipped_functions)}')

    score_files = [f"{function}_scores.csv" for function in functions]
    csv_files = [
        file for file in (rescoring_folder.rglob("*.csv"))
        if file.name in score_files]
    csv_dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        csv_dfs.append(df)
    if len(csv_dfs) == 1:
        combined_dfs = csv_dfs[0]
    if len(csv_dfs) > 1:
        combined_dfs = csv_dfs[0]
        for df in tqdm(csv_dfs[1:], desc="Combining scores", unit="files"):
            combined_dfs = pd.merge(combined_dfs, df, on="Pose ID", how="inner")
    first_column = combined_dfs.pop("Pose ID")
    combined_dfs.insert(0, "Pose ID", first_column)
    columns = combined_dfs.columns
    col = columns[1:]
    for c in col.tolist():
        if c == "Pose ID":
            pass
        if combined_dfs[c].dtypes is not float:
            combined_dfs[c] = combined_dfs[c].apply(pd.to_numeric,
                                                    errors="coerce")
        else:
            pass
    combined_dfs.to_csv(rescoring_folder / "allposes_rescored.csv", index=False)
    # delete_files(rescoring_folder, 'allposes_rescored.csv')
    toc = time.perf_counter()
    printlog(f"Rescoring complete in {toc - tic:0.4f}!")
    return


def rescore_docking(w_dir: Path, protein_file: Path, software: Path,
                    function: str, n_cpus: int) -> None:
    """
    Rescores ligand poses using the specified software and scoring functions. The function splits the input SDF file into
    smaller files, and then runs the specified software on each of these files in parallel. The results are then combined into a single
    Pandas dataframe and saved to a CSV file.

    Args:
        w_dir (str): The working directory.
        protein_file (str): The path to the protein file.
        pocket_definition (dict): A dictionary containing the pocket center and size.
        software (str): The path to the software to be used for rescoring.
        all_poses (str): The path to the input SDF file containing the clustered poses.
        functions (List[str]): A list of the scoring functions to be used.
        n_cpus (int): The number of CPUs to use for parallel execution.

    Returns:
        None
    """
    RDLogger.DisableLog("rdApp.*")
    tic = time.perf_counter()

    all_poses = Path(f"{w_dir}/allposes.sdf")

    function_info = RESCORING_FUNCTIONS.get(function)

    function_info["function"](
        all_poses,
        n_cpus,
        function_info["column_name"],
        protein_file=protein_file,
        software=software,
        rescoring_folder=w_dir,
    )

    score_file = f"{w_dir}/{function}_rescoring/{function}_scores.csv"

    score_df = pd.read_csv(score_file)
    if "Unnamed: 0" in score_df.columns:
        score_df = score_df.drop(columns=["Unnamed: 0"])

    score_df["Pose_Number"] = score_df["Pose ID"].str.split("_").str[2].astype(
        int)
    score_df["Docking_program"] = score_df["Pose ID"].str.split(
        "_").str[1].astype(str)
    score_df["ID"] = score_df["Pose ID"].str.split("_").str[0].astype(str)

    if function_info["best_value"] == "min":
        best_pose_indices = score_df.groupby("ID")[
            function_info["column_name"]].idxmin()
    else:
        best_pose_indices = score_df.groupby("ID")[
            function_info["column_name"]].idxmax()

    if os.path.exists(score_file):
        os.remove(score_file)
    if os.path.exists(f"{w_dir}/{function}_rescoring"):
        os.rmdir(f"{w_dir}/{function}_rescoring")

    best_poses = pd.DataFrame(score_df.loc[best_pose_indices, "Pose ID"])
    toc = time.perf_counter()
    printlog(f"Rescoring complete in {toc - tic:0.4f}!")
    return best_poses
