import subprocess
import sys
import time
import warnings
from pathlib import Path
from subprocess import DEVNULL, STDOUT

import pandas as pd

# Search for 'DockM8' in parent directories
dockm8_path = next(
    (p / "DockM8" for p in Path(__file__).resolve().parents if (p / "DockM8").is_dir()),
    None,
)
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import (
    convert_molecules,
    delete_files,
    printlog,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def SCORCH_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs):
    """
    Rescores ligands in an SDF file using SCORCH and saves the results in a CSV file.

    Args:
        sdf (str): Path to the SDF file containing the ligands to be rescored.
        n_cpus (int): Number of CPUs to use for parallel processing.
        column_name (str): Name of the column to store the SCORCH scores in the output CSV file.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
    rescoring_folder = kwargs.get("rescoring_folder")
    software = kwargs.get("software")
    protein_file = kwargs.get("protein_file")

    tic = time.perf_counter()
    SCORCH_rescoring_folder = rescoring_folder / f"{column_name}_rescoring"
    SCORCH_rescoring_folder.mkdir(parents=True, exist_ok=True)
    SCORCH_protein = SCORCH_rescoring_folder / "protein.pdbqt"
    convert_molecules(
        str(protein_file).replace(".pdb", "_pocket.pdb"), SCORCH_protein, "pdb", "pdbqt"
    )
    # Convert ligands to pdbqt
    split_files_folder = SCORCH_rescoring_folder / f"split_{sdf.stem}"
    split_files_folder.mkdir(exist_ok=True)
    convert_molecules(sdf, split_files_folder, "sdf", "pdbqt")
    # Run SCORCH

    SCORCH_command = f"python {software}/SCORCH-1.0.0/scorch.py --receptor {SCORCH_protein} --ligand {split_files_folder} --out {SCORCH_rescoring_folder}/scoring_results.csv --threads {n_cpus} --return_pose_scores"
    subprocess.call(SCORCH_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    # Clean data
    SCORCH_scores = pd.read_csv(SCORCH_rescoring_folder / "scoring_results.csv")
    SCORCH_scores = SCORCH_scores.rename(
        columns={"Ligand_ID": "Pose ID", "SCORCH_pose_score": column_name}
    )
    SCORCH_scores = SCORCH_scores[[column_name, "Pose ID"]]
    SCORCH_rescoring_results = SCORCH_rescoring_folder / f"{column_name}_scores.csv"
    SCORCH_scores.to_csv(SCORCH_rescoring_results, index=False)
    delete_files(SCORCH_rescoring_folder, f"{column_name}_scores.csv")
    toc = time.perf_counter()
    printlog(f"Rescoring with SCORCH complete in {toc-tic:0.4f}!")
    return SCORCH_rescoring_results
