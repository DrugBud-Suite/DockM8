import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from subprocess import DEVNULL, STDOUT

import pandas as pd
from pandas import DataFrame
from rdkit.Chem import PandasTools

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


def vinardo_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs) -> DataFrame:
    """
    Performs rescoring of poses using the Vinardo scoring function.

    Args:
        sdf (str): The path to the input SDF file containing the poses to be rescored.
        n_cpus (int): The number of CPUs to be used for the rescoring process.
        column_name (str): The name of the column in the output dataframe to store the Vinardo scores.
        **kwargs: Additional keyword arguments for rescoring.

    Keyword Args:
        rescoring_folder (str): The path to the folder for storing the Vinardo rescoring results.
        software (str): The path to the gnina software.
        protein_file (str): The path to the protein file.
        pocket_definition (dict): The pocket definition.

    Returns:
        DataFrame: A dataframe containing the 'Pose ID' and Vinardo score columns for the rescored poses.
    """
    tic = time.perf_counter()

    rescoring_folder = kwargs.get("rescoring_folder")
    software = kwargs.get("software")
    protein_file = kwargs.get("protein_file")
    pocket_definition = kwargs.get("pocket_definition")

    split_files_folder = split_sdf_str(
        rescoring_folder / f"{column_name}_rescoring", sdf, n_cpus
    )
    split_files_sdfs = [
        split_files_folder / f
        for f in os.listdir(split_files_folder)
        if f.endswith(".sdf")
    ]

    vinardo_rescoring_folder = rescoring_folder / f"{column_name}_rescoring"
    vinardo_rescoring_folder.mkdir(parents=True, exist_ok=True)

    global vinardo_rescoring_splitted

    def vinardo_rescoring_splitted(split_file, protein_file):
        vinardo_rescoring_folder = rescoring_folder / f"{column_name}_rescoring--"
        results = (
            vinardo_rescoring_folder / f"{Path(split_file).stem}_{column_name}.sdf"
        )
        vinardo_cmd = (
            f"{software}/gnina"
            f" --receptor {protein_file}"
            f" --ligand {split_file}"
            f" --out {results}"
            " --score_only"
            " --scoring vinardo"
            " --cnn_scoring none"
        )
        try:
            subprocess.call(vinardo_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            printlog(f"{column_name} rescoring failed: " + e)
        return

    parallel_executor(
        vinardo_rescoring_splitted,
        split_files_sdfs,
        n_cpus,
        protein_file=protein_file,
        pocket_definition=pocket_definition,
    )

    try:
        vinardo_dataframes = [
            PandasTools.LoadSDF(
                str(rescoring_folder / f"{column_name}_rescoring" / file),
                idName="Pose ID",
                molColName=None,
                includeFingerprints=False,
                embedProps=False,
            )
            for file in os.listdir(rescoring_folder / f"{column_name}_rescoring")
            if file.startswith("split") and file.endswith(".sdf")
        ]
    except Exception as e:
        printlog(f"ERROR: Failed to Load {column_name} rescoring SDF file!")
        printlog(e)
    try:
        vinardo_rescoring_results = pd.concat(vinardo_dataframes)
    except Exception as e:
        printlog(f"ERROR: Could not combine {column_name} rescored poses")
        printlog(e)
    vinardo_rescoring_results.rename(
        columns={"minimizedAffinity": column_name}, inplace=True
    )
    vinardo_rescoring_results = vinardo_rescoring_results[["Pose ID", column_name]]
    vinardo_scores_path = vinardo_rescoring_folder / f"{column_name}_scores.csv"
    vinardo_rescoring_results.to_csv(vinardo_scores_path, index=False)
    delete_files(vinardo_rescoring_folder, f"{column_name}_scores.csv")
    toc = time.perf_counter()
    printlog(f"Rescoring with Vinardo complete in {toc - tic:0.4f}!")
    return vinardo_rescoring_results
