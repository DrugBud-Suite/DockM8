import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from subprocess import DEVNULL, STDOUT

import pandas as pd
from pandas import DataFrame

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


def AAScore_rescoring(sdf: str, n_cpus: int, column_name: str,
                      **kwargs) -> DataFrame:
    """
    Rescores poses in an SDF file using the AA-Score tool.

    Args:
    - sdf (str): The path to the SDF file containing the poses to be rescored.
    - n_cpus (int): The number of CPUs to use for parallel processing.
    - column_name (str): The name of the column to be used for the rescored scores.
    - kwargs: Additional keyword arguments.

    Keyword Args:
    - rescoring_folder (str): The path to the folder where the rescored results will be saved.
    - software (str): The path to the AA-Score software.
    - protein_file (str): The path to the protein file.
    - pocket_de (str): The path to the pocket definitions file.

    Returns:
    - A pandas DataFrame containing the rescored poses and their scores.
    """
    tic = time.perf_counter()
    rescoring_folder = kwargs.get("rescoring_folder")
    software = kwargs.get("software")
    protein_file = kwargs.get("protein_file")

    (rescoring_folder / f"{column_name}_rescoring").mkdir(parents=True,
                                                          exist_ok=True)
    pocket = str(protein_file).replace(".pdb", "_pocket.pdb")

    if n_cpus == 1:
        results = rescoring_folder / f"{column_name}_rescoring" / "rescored_AAScore.csv"
        AAscore_cmd = f"python {software}/AA-Score-Tool-main/AA_Score.py --Rec {pocket} --Lig {sdf} --Out {results}"
        subprocess.call(AAscore_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        AAScore_rescoring_results = pd.read_csv(results,
                                                delimiter="\t",
                                                header=None,
                                                names=["Pose ID", column_name])
    else:
        split_files_folder = split_sdf_str(
            rescoring_folder / f"{column_name}_rescoring", sdf, n_cpus)
        split_files_sdfs = [
            Path(split_files_folder) / f
            for f in os.listdir(split_files_folder)
            if f.endswith(".sdf")
        ]
        global AAScore_rescoring_splitted

        def AAScore_rescoring_splitted(split_file):
            AAScore_folder = rescoring_folder / "AAScore_rescoring"
            results = AAScore_folder / f"{split_file.stem}_AAScore.csv"
            AAScore_cmd = f"python {software}/AA-Score-Tool-main/AA_Score.py --Rec {pocket} --Lig {split_file} --Out {results}"
            try:
                subprocess.call(AAScore_cmd,
                                shell=True,
                                stdout=DEVNULL,
                                stderr=STDOUT)
            except Exception as e:
                printlog("AAScore rescoring failed: " + str(e))

        parallel_executor(AAScore_rescoring_splitted, split_files_sdfs, n_cpus)

        try:
            AAScore_dataframes = [
                pd.read_csv(
                    rescoring_folder / "AAScore_rescoring" / file,
                    delimiter="\t",
                    header=None,
                    names=["Pose ID", column_name],
                )
                for file in os.listdir(rescoring_folder / "AAScore_rescoring")
                if file.startswith("split") and file.endswith(".csv")
            ]
        except Exception as e:
            printlog("ERROR: Failed to Load AAScore rescoring SDF file!")
            printlog(e)
        else:
            try:
                AAScore_rescoring_results = pd.concat(AAScore_dataframes)
            except Exception as e:
                printlog("ERROR: Could not combine AAScore rescored poses")
                printlog(e)
            else:
                delete_files(rescoring_folder / "AAScore_rescoring",
                             "AAScore_scores.csv")
        AAScore_rescoring_results.to_csv(
            rescoring_folder / "AAScore_rescoring" / "AAScore_scores.csv",
            index=False)
        toc = time.perf_counter()
        printlog(f"Rescoring with AAScore complete in {toc - tic:0.4f}!")
        return AAScore_rescoring_results
