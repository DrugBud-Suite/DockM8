import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from subprocess import DEVNULL, STDOUT

import pandas as pd
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts"
                     for p in Path(__file__).resolve().parents
                     if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import delete_files, parallel_executor, printlog, split_sdf_str

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def LinF9_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs):
    """
    Performs rescoring of poses in an SDF file using the LinF9 scoring function.

    Args:
    sdf (str): The path to the SDF file containing the poses to be rescored.
    n_cpus (int): The number of CPUs to use for parallel execution.
    column_name (str): The name of the column to store the rescoring results.
    **kwargs: Additional keyword arguments.

    Keyword Args:
    rescoring_folder (str): Path to the folder where the rescoring results will be saved.
    software (str): Path to the software.
    protein_file (str): Path to the protein file.
    pocket_definition (str): Path to the pocket definition file.

    Returns:
    pandas.DataFrame: A DataFrame containing the rescoring results, with columns 'Pose ID' and the specified column name.
    """
    rescoring_folder = kwargs.get("rescoring_folder")
    software = kwargs.get("software")
    protein_file = kwargs.get("protein_file")

    tic = time.perf_counter()
    (rescoring_folder / f"{column_name}_rescoring").mkdir(parents=True,
                                                          exist_ok=True)
    split_files_folder = split_sdf_str(
        rescoring_folder / f"{column_name}_rescoring", sdf, n_cpus)
    split_files_sdfs = [
        Path(split_files_folder) / f
        for f in os.listdir(split_files_folder)
        if f.endswith(".sdf")]

    global LinF9_rescoring_splitted

    def LinF9_rescoring_splitted(split_file, protein_file):
        LinF9_folder = rescoring_folder / "LinF9_rescoring"
        results = LinF9_folder / f"{split_file.stem}_LinF9.sdf"
        LinF9_cmd = (f"{software}/smina.static" +
                     f" --receptor {protein_file}" + f" --ligand {split_file}" +
                     f" --out {results}" + " --cpu 1" +
                     " --scoring Lin_F9 --score_only")
        try:
            subprocess.call(LinF9_cmd,
                            shell=True,
                            stdout=DEVNULL,
                            stderr=STDOUT)
        except Exception as e:
            printlog(f"LinF9 rescoring failed: {e}")
        return

    parallel_executor(LinF9_rescoring_splitted,
                      split_files_sdfs,
                      n_cpus,
                      protein_file=protein_file)

    try:
        LinF9_dataframes = [
            PandasTools.LoadSDF(
                str(rescoring_folder / "LinF9_rescoring" / file),
                idName="Pose ID",
                molColName=None,
                includeFingerprints=False,
                embedProps=False,
            )
            for file in os.listdir(rescoring_folder / "LinF9_rescoring")
            if file.startswith("split") and file.endswith("_LinF9.sdf")]
    except Exception as e:
        printlog("ERROR: Failed to Load LinF9 rescoring SDF file!")
        printlog(e)

    try:
        LinF9_rescoring_results = pd.concat(LinF9_dataframes)
    except Exception as e:
        printlog("ERROR: Could not combine LinF9 rescored poses")
        printlog(e)

    LinF9_rescoring_results.rename(columns={"minimizedAffinity": column_name},
                                   inplace=True)
    LinF9_rescoring_results = LinF9_rescoring_results[["Pose ID", column_name]]
    LinF9_rescoring_results.to_csv(rescoring_folder / "LinF9_rescoring" /
                                   "LinF9_scores.csv",
                                   index=False)
    delete_files(rescoring_folder / "LinF9_rescoring", "LinF9_scores.csv")
    toc = time.perf_counter()
    printlog(f"Rescoring with LinF9 complete in {toc-tic:0.4f}!")
    return LinF9_rescoring_results
