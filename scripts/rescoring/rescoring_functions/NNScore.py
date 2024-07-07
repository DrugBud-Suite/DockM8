import subprocess
import sys
import time
import warnings
from pathlib import Path

from pandas import DataFrame
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog
from scripts.utilities.utilities import delete_files

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class NNScore(ScoringFunction):

	def __init__(self):
		super().__init__("NNScore", "NNScore", "max", (0, 20))

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> DataFrame:
		tic = time.perf_counter()
		rescoring_folder = kwargs.get("rescoring_folder")
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		nnscore_rescoring_folder = rescoring_folder / f"{self.column_name}_rescoring"
		nnscore_rescoring_folder.mkdir(parents=True, exist_ok=True)
		pickle_path = f"{software}/models/NNScore_pdbbind2016.pickle"
		results = nnscore_rescoring_folder / "rescored_NNscore.sdf"
		nnscore_rescoring_command = ("oddt_cli " + str(sdf) + " --receptor " + str(protein_file) + " -n " +
										str(n_cpus) + " --score_file " + str(pickle_path) + " -O " + str(results))
		subprocess.call(nnscore_rescoring_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		NNScore_results_df = PandasTools.LoadSDF(str(results),
													idName="Pose ID",
													molColName=None,
													includeFingerprints=False,
													removeHs=False)
		NNScore_results_df.rename(columns={"nnscore": self.column_name}, inplace=True)
		NNScore_results_df = NNScore_results_df[["Pose ID", self.column_name]]
		NNScore_rescoring_results = nnscore_rescoring_folder / f"{self.column_name}_scores.csv"
		NNScore_results_df.to_csv(NNScore_rescoring_results, index=False)
		toc = time.perf_counter()
		printlog(f"Rescoring with NNscore complete in {toc-tic:0.4f}!")
		delete_files(nnscore_rescoring_folder, f"{self.column_name}_scores.csv")
		return NNScore_results_df
