import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import pandas as pd
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ConvexPLR(ScoringFunction):

	def __init__(self):
		super().__init__("ConvexPLR", "ConvexPLR", "max", (-10, 10))

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		with tempfile.TemporaryDirectory() as temp_dir:
			split_files = split_sdf_str(Path(temp_dir), sdf, n_cpus)
			split_files_sdfs = [Path(temp_dir) / f for f in os.listdir(split_files) if f.endswith(".sdf")]

			global ConvexPLR_rescoring_splitted

			def ConvexPLR_rescoring_splitted(split_file, protein_file):
				df = PandasTools.LoadSDF(str(split_file), idName="Pose ID", molColName=None)
				df = df[["Pose ID"]]
				ConvexPLR_command = (f"{software}/Convex-PL" + f" --receptor {protein_file}" +
										f" --ligand {split_file}" + " --sdf --regscore")
				process = subprocess.Popen(ConvexPLR_command,
											stdout=subprocess.PIPE,
											stderr=subprocess.PIPE,
											shell=True)
				stdout, stderr = process.communicate()
				energies = []
				output = stdout.decode().splitlines()
				for line in output:
					if line.startswith("model"):
						parts = line.split(",")
						energy = round(float(parts[1].split("=")[1]), 2)
						energies.append(energy)
				df[self.column_name] = energies
				return df

			results = parallel_executor(ConvexPLR_rescoring_splitted,
										split_files_sdfs,
										n_cpus,
										protein_file=protein_file)

		combined_scores_df = pd.concat(results, ignore_index=True)

		toc = time.perf_counter()
		printlog(f"Rescoring with ConvexPLR complete in {toc-tic:0.4f}!")
		return combined_scores_df
