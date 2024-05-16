import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from subprocess import DEVNULL, STDOUT

import pandas as pd
from tqdm import tqdm

# Search for 'DockM8' in parent directories
scripts_path = next((p / 'scripts'
                     for p in Path(__file__).resolve().parents
                     if (p / 'scripts').is_dir()), None)
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


def RTMScore_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs):
    """
    Rescores poses in an SDF file using RTMScore.

    Args:
    - sdf (str): Path to the SDF file containing the poses to be rescored.
    - n_cpus (int): Number of CPUs to use for parallel execution.
    - column_name (str): Name of the column in the output CSV file that will contain the RTMScore scores.
    - **kwargs: Additional keyword arguments.

    Keyword Args:
    - rescoring_folder (str): Path to the folder where the rescoring results will be saved.
    - software (str): Path to the RTMScore software.
    - protein_file (str): Path to the protein file.
    - pocket_definition (str): Path to the pocket definition file.

    Returns:
    - None
    """
    rescoring_folder = kwargs.get("rescoring_folder")
    software = kwargs.get("software")
    protein_file = kwargs.get("protein_file")

    tic = time.perf_counter()
    (rescoring_folder / f"{column_name}_rescoring").mkdir(parents=True,
                                                          exist_ok=True)
    RTMScore_rescoring_results = str(rescoring_folder /
                                     f"{column_name}_rescoring" /
                                     f"{column_name}_scores.csv")
    try:
        RTMScore_command = (
            f'cd {rescoring_folder / "RTMScore_rescoring"} && python {software}/RTMScore-main/example/rtmscore.py'
            + f' -p {str(protein_file).replace(".pdb", "_pocket.pdb")}' +
            f" -l {sdf}" + " -o RTMScore_scores" + " -pl"
            f" -m {software}/RTMScore-main/trained_models/rtmscore_model1.pth")
        subprocess.call(RTMScore_command,
                        shell=True,
                        stdout=DEVNULL,
                        stderr=STDOUT)
    except Exception as e:
        if not os.path.exists(
                os.path.join(software, "RTMScore-main", "example",
                             "rtmscore.py")):
            printlog(
                "ERROR: Failed to run RTMScore! The software folder does not contain rtmscore.py, please reinstall RTMScore."
            )
        else:
            printlog(
                f"ERROR: Failed to run RTMScore! This was likely caused by a failure in generating the pocket graph : {e}."
            )
    df = pd.read_csv(RTMScore_rescoring_results)
    df = df.rename(columns={"id": "Pose ID", "score": f"{column_name}"})
    df["Pose ID"] = df["Pose ID"].str.rsplit("-", n=1).str[0]
    df.to_csv(RTMScore_rescoring_results, index=False)
    delete_files(rescoring_folder / f"{column_name}_rescoring",
                 f"{column_name}_scores.csv")
    toc = time.perf_counter()
    printlog(f"Rescoring with RTMScore complete in {toc-tic:0.4f}!")
    return RTMScore_rescoring_results

    # def RTMScore_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs):
    """
    Rescores poses in an SDF file using RTMScore.

    Args:
    - sdf (str): Path to the SDF file containing the poses to be rescored.
    - n_cpus (int): Number of CPUs to use for parallel execution.
    - column_name (str): Name of the column in the output CSV file that will contain the RTMScore scores.
    - **kwargs: Additional keyword arguments.

    Keyword Args:
    - rescoring_folder (str): Path to the folder where the rescoring results will be saved.
    - software (str): Path to the RTMScore software.
    - protein_file (str): Path to the protein file.
    - pocket_definition (str): Path to the pocket definition file.

    Returns:
    - None
    """
    rescoring_folder = kwargs.get("rescoring_folder")
    software = kwargs.get("software")
    protein_file = kwargs.get("protein_file")

    tic = time.perf_counter()

    split_files_folder = split_sdf_str(
        rescoring_folder / f"{column_name}_rescoring", sdf, n_cpus * 2)
    split_files_sdfs = [
        split_files_folder / f
        for f in os.listdir(split_files_folder)
        if f.endswith(".sdf")
    ]

    (rescoring_folder / f"{column_name}_rescoring").mkdir(parents=True,
                                                          exist_ok=True)

    global RTMScore_rescoring_splitted

    def RTMScore_rescoring_splitted(split_file, protein_file):
        RTMScore_command = (
            f'cd {rescoring_folder / "RTMScore_rescoring"} && python {software}/RTMScore-main/example/rtmscore.py'
            + f' -p {str(protein_file).replace(".pdb", "_pocket.pdb")}' +
            f" -l {split_file}" + f" -o RTMScore_scores_{split_file.stem}" +
            f" -m {software}/RTMScore-main/trained_models/rtmscore_model1.pth")
        subprocess.call(RTMScore_command,
                        shell=True,
                        stdout=DEVNULL,
                        stderr=STDOUT)
        return

    parallel_executor(RTMScore_rescoring_splitted,
                      split_files_sdfs,
                      3,
                      protein_file=protein_file)

    try:
        score_dfs = []
        for file in os.listdir(rescoring_folder / "RTMScore_rescoring"):
            if file.startswith("RTMScore_scores_") and file.endswith(".csv"):
                df = pd.read_csv(rescoring_folder / "RTMScore_rescoring" / file)
                df = df.rename(columns={
                    "id": "Pose ID",
                    "score": f"{column_name}"
                })
                df["Pose ID"] = df["Pose ID"].str.rsplit("-", n=1).str[0]
                score_dfs.append(df)
        RTMScore_rescoring_results = pd.concat(score_dfs)
    except Exception as e:
        printlog(f"Failed to combine {column_name} score files!")
        printlog(e)

    try:
        output_file = str(rescoring_folder / f"{column_name}_rescoring" /
                          f"{column_name}_scores.csv")
        RTMScore_rescoring_results.to_csv(output_file, index=False)
    except Exception as e:
        printlog(f"ERROR: Could not write {column_name} combined scores")
        printlog(e)

    delete_files(rescoring_folder / f"{column_name}_rescoring",
                 f"{column_name}_scores.csv")
    toc = time.perf_counter()
    printlog(f"Rescoring with RTMScore complete in {toc-tic:0.4f}!")
    return

    # def RTMScore_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs):
    """
    Rescores poses in an SDF file using RTMScore.

    Args:
    - sdf (str): Path to the SDF file containing the poses to be rescored.
    - n_cpus (int): Number of CPUs to use for parallel execution.
    - column_name (str): Name of the column in the output CSV file that will contain the RTMScore scores.
    - **kwargs: Additional keyword arguments.

    Keyword Args:
    - rescoring_folder (str): Path to the folder where the rescoring results will be saved.
    - software (str): Path to the RTMScore software.
    - protein_file (str): Path to the protein file.
    - pocket_definition (str): Path to the pocket definition file.

    Returns:
    - None
    """
    rescoring_folder = kwargs.get("rescoring_folder")
    software = kwargs.get("software")
    protein_file = kwargs.get("protein_file")

    tic = time.perf_counter()

    split_files_folder = split_sdf_str(
        rescoring_folder / f"{column_name}_rescoring", sdf, n_cpus * 2)
    split_files_sdfs = [
        split_files_folder / f
        for f in os.listdir(split_files_folder)
        if f.endswith(".sdf")
    ]

    (rescoring_folder / f"{column_name}_rescoring").mkdir(parents=True,
                                                          exist_ok=True)
    for file in tqdm(split_files_sdfs):
        try:
            RTMScore_command = (
                f'cd {rescoring_folder / "RTMScore_rescoring"} && python {software}/RTMScore-main/example/rtmscore.py'
                + f' -p {str(protein_file).replace(".pdb", "_pocket.pdb")}' +
                f" -l {file}" + f" -o RTMScore_scores_{file.stem}" + " -pl"
                f" -m {software}/RTMScore-main/trained_models/rtmscore_model1.pth"
            )
            subprocess.call(RTMScore_command,
                            shell=True,
                            stdout=DEVNULL,
                            stderr=STDOUT)
        except Exception as e:
            printlog(f"Failed to run RTMScore on {file}!")
            printlog(e)

    try:
        score_dfs = []
        for file in os.listdir(rescoring_folder / "RTMScore_rescoring"):
            if file.startswith("RTMScore_scores_") and file.endswith(".csv"):
                df = pd.read_csv(rescoring_folder / "RTMScore_rescoring" / file)
                df = df.rename(columns={
                    "id": "Pose ID",
                    "score": f"{column_name}"
                })
                df["Pose ID"] = df["Pose ID"].str.rsplit("-", n=1).str[0]
                score_dfs.append(df)
        RTMScore_rescoring_results = pd.concat(score_dfs)
    except Exception as e:
        printlog(f"Failed to combine {column_name} score files!")
        printlog(e)

    try:
        output_file = str(rescoring_folder / f"{column_name}_rescoring" /
                          f"{column_name}_scores.csv")
        RTMScore_rescoring_results.to_csv(output_file, index=False)
    except Exception as e:
        printlog(f"ERROR: Could not write {column_name} combined scores")
        printlog(e)

    delete_files(rescoring_folder / f"{column_name}_rescoring",
                 f"{column_name}_scores.csv")
    toc = time.perf_counter()
    printlog(f"Rescoring with RTMScore complete in {toc-tic:0.4f}!")
    return
