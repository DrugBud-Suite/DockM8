import math
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from subprocess import DEVNULL, STDOUT

import pandas as pd
from rdkit.Chem import PandasTools

cwd = Path.cwd()
dockm8_path = cwd.parents[0] / "DockM8"
sys.path.append(str(dockm8_path))

from scripts.utilities import parallel_executor, printlog, split_sdf_str

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def generate_conformers_GypsumDL(
    input_sdf: Path, output_dir: Path, software: Path, ncpus: int, n_conformers: int = 1
):
    """
    Generates 3D conformers using GypsumDL.

    Args:
        input_sdf (str): Path to the input SDF file.
        output_dir (str): Path to the output directory.
        software (str): Path to the GypsumDL software.
        ncpus (int): Number of CPUs to use for the calculation.

    Raises:
        Exception: If failed to generate conformers.

    """
    printlog("Generating 3D conformers using GypsumDL...")

    # Splitting input SDF file into smaller files for parallel processing
    printlog("Splitting input SDF file into smaller files for parallel processing...")
    split_files_folder = split_sdf_str(output_dir / "GypsumDL_split", input_sdf, 10)
    split_files_sdfs = [
        split_files_folder / f
        for f in os.listdir(split_files_folder)
        if f.endswith(".sdf")
    ]

    global gypsum_dl_run

    def gypsum_dl_run(split_file, output_dir, cpus):
        results_dir = output_dir / "GypsumDL_results"
        try:
            # Running GypsumDL command for each split file
            gypsum_dl_command = (
                f"python {software}/gypsum_dl-1.2.1/run_gypsum_dl.py "
                f"-s {split_file} "
                f"-o {results_dir} "
                f"--job_manager multiprocessing "
                f"-p {cpus} "
                f"-m 1 "
                f"-t 10 "
                f"--skip_adding_hydrogen "
                f"--skip_alternate_ring_conformations "
                f"--skip_making_tautomers "
                f"--skip_enumerate_chiral_mol "
                f"--skip_enumerate_double_bonds "
                f"--max_variants_per_compound {n_conformers} "
                f"--separate_output_files"
            )
            subprocess.call(
                gypsum_dl_command, shell=True, stdout=DEVNULL, stderr=STDOUT
            )
        except Exception as e:
            printlog("ERROR: Failed to generate conformers!")
            printlog(e)
        return

    # Running GypsumDL in parallel
    printlog("Running GypsumDL in parallel...")
    parallel_executor(
        gypsum_dl_run,
        split_files_sdfs,
        3,
        output_dir=output_dir,
        cpus=math.ceil(ncpus // 3),
    )

    results_dfs = []

    # Loading generated conformers from output directory
    for file in os.listdir(output_dir / "GypsumDL_results"):
        if file.endswith(".sdf"):
            sdf_df = PandasTools.LoadSDF(
                str(output_dir / "GypsumDL_results" / file),
                molColName="Molecule",
                idName="ID",
                removeHs=False,
            )
            results_dfs.append(sdf_df)

    combined_df = pd.concat(results_dfs)

    # Remove the row containing GypsumDL parameters from the DataFrame
    final_df = combined_df[
        combined_df["ID"] != "EMPTY MOLECULE DESCRIBING GYPSUM-DL PARAMETERS"
    ]

    # Select only the 'Molecule' and 'ID' columns from the DataFrame
    final_df = final_df[["Molecule", "ID"]]

    output_file = output_dir / "generated_conformers.sdf"

    # Writing final conformers to output file
    PandasTools.WriteSDF(
        final_df,
        str(output_file),
        molColName="Molecule",
        idName="ID",
    )

    # Cleaning up temporary directories and files
    shutil.rmtree(output_dir / "GypsumDL_results")
    shutil.rmtree(output_dir / "GypsumDL_split")
    (output_dir / "gypsum_dl_success.sdf").unlink(missing_ok=True)
    (output_dir / "gypsum_dl_failed.smi").unlink(missing_ok=True)

    return output_file
