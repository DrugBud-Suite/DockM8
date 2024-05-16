import subprocess
import sys
import time
import warnings
from pathlib import Path
from subprocess import DEVNULL, STDOUT

from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts"
                     for p in Path(__file__).resolve().parents
                     if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import delete_files, printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def oddt_nnscore_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs):
    """
    Rescores the input SDF file using the NNscore algorithm and returns a Pandas dataframe with the rescored values.

    Args:
    sdf (str): Path to the input SDF file.
    n_cpus (int): Number of CPUs to use for the rescoring.
    column_name (str): Name of the column to store the rescored values in the output dataframe.
    **kwargs: Additional keyword arguments.

    Returns:
    df (Pandas dataframe): Dataframe with the rescored values and the corresponding pose IDs.
    """
    rescoring_folder = kwargs.get("rescoring_folder")
    software = kwargs.get("software")
    protein_file = kwargs.get("protein_file")

    tic = time.perf_counter()

    nnscore_rescoring_folder = rescoring_folder / f"{column_name}_rescoring"
    nnscore_rescoring_folder.mkdir(parents=True, exist_ok=True)
    pickle_path = f"{software}/models/NNScore_pdbbind2016.pickle"
    results = nnscore_rescoring_folder / "rescored_NNscore.sdf"
    nnscore_rescoring_command = ("oddt_cli " + str(sdf) + " --receptor " +
                                 str(protein_file) + " -n " + str(n_cpus) +
                                 " --score_file " + str(pickle_path) + " -O " +
                                 str(results))
    subprocess.call(nnscore_rescoring_command,
                    shell=True,
                    stdout=DEVNULL,
                    stderr=STDOUT)
    NNScore_results_df = PandasTools.LoadSDF(str(results),
                                             idName="Pose ID",
                                             molColName=None,
                                             includeFingerprints=False,
                                             removeHs=False)
    NNScore_results_df.rename(columns={"nnscore": column_name}, inplace=True)
    NNScore_results_df = NNScore_results_df[["Pose ID", column_name]]
    NNScore_rescoring_results = nnscore_rescoring_folder / f"{column_name}_scores.csv"
    NNScore_results_df.to_csv(NNScore_rescoring_results, index=False)
    toc = time.perf_counter()
    printlog(f"Rescoring with NNscore complete in {toc-tic:0.4f}!")
    delete_files(nnscore_rescoring_folder, f"{column_name}_scores.csv")
    return NNScore_rescoring_results
