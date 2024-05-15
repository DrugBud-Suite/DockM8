import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from subprocess import DEVNULL, STDOUT

import pandas as pd

# Search for 'DockM8' in parent directories
scripts_path = next((p / 'scripts' for p in Path(__file__).resolve().parents if (p / 'scripts').is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import (
    delete_files,
    parallel_executor,
    printlog,
    split_sdf_str,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def rfscorevs_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs):
    """
    Rescores poses in an SDF file using RFScoreVS and returns the results as a pandas DataFrame.

    Args:
        sdf (str): Path to the SDF file containing the poses to be rescored.
        n_cpus (int): Number of CPUs to use for the RFScoreVS calculation.
        column_name (str): Name of the column to be used for the RFScoreVS scores in the output DataFrame.
        kwargs: Additional keyword arguments.

    Keyword Args:
        rescoring_folder (str): Path to the folder for storing the RFScoreVS rescoring results.
        software (str): Path to the RFScoreVS software.
        protein_file (str): Path to the receptor protein file.
        pocket_de (dict): Dictionary containing pocket definitions.

    Returns:
        pandas.DataFrame: DataFrame containing the RFScoreVS scores for each pose in the input SDF file.
    """
    rescoring_folder = kwargs.get("rescoring_folder")
    software = kwargs.get("software")
    protein_file = kwargs.get("protein_file")

    tic = time.perf_counter()

    rfscorevs_rescoring_folder = rescoring_folder / f"{column_name}_rescoring"
    rfscorevs_rescoring_folder.mkdir(parents=True, exist_ok=True)

    split_files_folder = split_sdf_str(
        rescoring_folder / f"{column_name}_rescoring", sdf, n_cpus
    )
    split_files_sdfs = [
        split_files_folder / f
        for f in os.listdir(split_files_folder)
        if f.endswith(".sdf")
    ]
    global rf_score_vs_splitted

    def rf_score_vs_splitted(split_file, protein_file):
        rfscorevs_cmd = f"{software}/rf-score-vs --receptor {protein_file} {split_file} -O {rfscorevs_rescoring_folder / Path(split_file).stem}_RFScoreVS_scores.csv -n 1"
        subprocess.call(rfscorevs_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        return

    parallel_executor(
        rf_score_vs_splitted, split_files_sdfs, n_cpus, protein_file=protein_file
    )

    try:
        rfscorevs_dataframes = [
            pd.read_csv(rfscorevs_rescoring_folder / file, delimiter=",", header=0)
            for file in os.listdir(rfscorevs_rescoring_folder)
            if file.startswith("split") and file.endswith(".csv")
        ]
        rfscorevs_results = pd.concat(rfscorevs_dataframes)
        rfscorevs_results.rename(
            columns={"name": "Pose ID", "RFScoreVS_v2": column_name}, inplace=True
        )
    except Exception as e:
        printlog("ERROR: Failed to process RFScoreVS results!")
        printlog(e)
    rfscorevs_results.to_csv(
        rescoring_folder / f"{column_name}_rescoring" / f"{column_name}_scores.csv",
        index=False,
    )
    delete_files(
        rescoring_folder / f"{column_name}_rescoring", f"{column_name}_scores.csv"
    )
    toc = time.perf_counter()
    printlog(f"Rescoring with RFScoreVS complete in {toc-tic:0.4f}!")
    return rfscorevs_results

    # def rfscorevs_rescoring(sdf : str, n_cpus : int, column_name : str, **kwargs):
    """
    Rescores poses in an SDF file using RFScoreVS and returns the results as a pandas DataFrame.

    Args:
        sdf (str): Path to the SDF file containing the poses to be rescored.
        n_cpus (int): Number of CPUs to use for the RFScoreVS calculation.
        column_name (str): Name of the column to be used for the RFScoreVS scores in the output DataFrame.
        kwargs: Additional keyword arguments.

    Keyword Args:
        rescoring_folder (str): Path to the folder for storing the RFScoreVS rescoring results.
        software (str): Path to the RFScoreVS software.
        protein_file (str): Path to the receptor protein file.
        pocket_de (dict): Dictionary containing pocket definitions.

    Returns:
        pandas.DataFrame: DataFrame containing the RFScoreVS scores for each pose in the input SDF file.
    """
    rescoring_folder = kwargs.get("rescoring_folder")
    software = kwargs.get("software")
    protein_file = kwargs.get("protein_file")

    tic = time.perf_counter()

    rfscorevs_rescoring_folder = rescoring_folder / f"{column_name}_rescoring"
    rfscorevs_rescoring_folder.mkdir(parents=True, exist_ok=True)

    rfscorevs_cmd = f'{software}/rf-score-vs --receptor {protein_file} {sdf} -O {rfscorevs_rescoring_folder / f"{column_name}_scores.csv -n {n_cpus}"}'
    subprocess.call(rfscorevs_cmd, shell=True)

    try:
        rfscorevs_results = pd.read_csv(
            rfscorevs_rescoring_folder / f"{column_name}_scores.csv",
            delimiter=",",
            header=0,
        )
        rfscorevs_results.rename(
            columns={"name": "Pose ID", "RFScoreVS_v2": column_name}, inplace=True
        )
    except Exception as e:
        printlog("ERROR: Failed to process RFScoreVS results!")
        printlog(e)
    rfscorevs_results.to_csv(
        rescoring_folder / f"{column_name}_rescoring" / f"{column_name}_scores.csv",
        index=False,
    )
    delete_files(
        rescoring_folder / f"{column_name}_rescoring", f"{column_name}_scores.csv"
    )
    toc = time.perf_counter()
    printlog(f"Rescoring with RFScoreVS complete in {toc-tic:0.4f}!")
    return rfscorevs_results
