import subprocess
import sys
import time
import warnings
from pathlib import Path

from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
dockm8_path = next(
    (p / "DockM8" for p in Path(__file__).resolve().parents if (p / "DockM8").is_dir()),
    None,
)
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import (
    delete_files,
    printlog,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def oddt_plecscore_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs):
    """
    Rescores the input SDF file using the PLECscore rescoring method.

    Args:
    - sdf (str): the path to the input SDF file
    - n_cpus (int): the number of CPUs to use for the rescoring calculation
    - column_name (str): the name of the column to use for the rescoring results
    **kwargs: Additional keyword arguments.

    Returns:
    - df (pandas.DataFrame): a DataFrame containing the rescoring results, with columns 'Pose ID' and 'column_name'
    """
    rescoring_folder = kwargs.get("rescoring_folder")
    software = kwargs.get("software")
    protein_file = kwargs.get("protein_file")

    tic = time.perf_counter()

    plecscore_rescoring_folder = rescoring_folder / f"{column_name}_rescoring"
    plecscore_rescoring_folder.mkdir(parents=True, exist_ok=True)
    pickle_path = f"{software}/models/PLECnn_p5_l1_pdbbind2016_s65536.pickle"
    results = plecscore_rescoring_folder / "rescored_PLECnn.sdf"
    plecscore_rescoring_command = (
        "oddt_cli "
        + str(sdf)
        + " --receptor "
        + str(protein_file)
        + " -n "
        + str(n_cpus)
        + " --score_file "
        + str(pickle_path)
        + " -O "
        + str(results)
    )
    subprocess.call(plecscore_rescoring_command, shell=True)
    PLECScore_results_df = PandasTools.LoadSDF(
        str(results),
        idName="Pose ID",
        molColName=None,
        includeFingerprints=False,
        removeHs=False,
    )
    PLECScore_results_df.rename(
        columns={"PLECnn_p5_l1_s65536": column_name}, inplace=True
    )
    PLECScore_results_df = PLECScore_results_df[["Pose ID", column_name]]
    PLECScore_rescoring_results = (
        plecscore_rescoring_folder / f"{column_name}_scores.csv"
    )
    PLECScore_results_df.to_csv(PLECScore_rescoring_results, index=False)
    toc = time.perf_counter()
    printlog(f"Rescoring with PLECScore complete in {toc-tic:0.4f}!")
    delete_files(plecscore_rescoring_folder, f"{column_name}_scores.csv")
    return PLECScore_rescoring_results
