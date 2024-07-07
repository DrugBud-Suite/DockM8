import sys
import time
import warnings
from pathlib import Path
from typing import List

import pandas as pd
from rdkit import RDLogger
from tqdm import tqdm

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring_functions.AAScore import AAScore
from scripts.rescoring.rescoring_functions.AD4 import AD4
from scripts.rescoring.rescoring_functions.CHEMPLP import CHEMPLP
from scripts.rescoring.rescoring_functions.ConvexPLR import ConvexPLR
from scripts.rescoring.rescoring_functions.gnina import Gnina
from scripts.rescoring.rescoring_functions.KORP_PL import KORPL
from scripts.rescoring.rescoring_functions.LinF9 import LinF9
from scripts.rescoring.rescoring_functions.NNScore import NNScore
from scripts.rescoring.rescoring_functions.PLECScore import PLECScore
from scripts.rescoring.rescoring_functions.PLP import PLP
from scripts.rescoring.rescoring_functions.RFScoreVS import RFScoreVS
from scripts.rescoring.rescoring_functions.RTMScore import RTMScore
from scripts.rescoring.rescoring_functions.SCORCH import SCORCH
from scripts.rescoring.rescoring_functions.vinardo import Vinardo
from scripts.rescoring.rescoring_functions.ITScoreAff import ITScoreAff
from scripts.rescoring.rescoring_functions.DLIGAND2 import DLIGAND2
from scripts.rescoring.rescoring_functions.CENsible import CENsible
from scripts.rescoring.rescoring_functions.PANTHER import PANTHER
from scripts.rescoring.rescoring_functions.GenScore import GenScore
from scripts.utilities.logging import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Updated RESCORING_FUNCTIONS dictionary using class instances
RESCORING_FUNCTIONS = {
	"GNINA-Affinity": Gnina("affinity"),
	"CNN-Score": Gnina("cnn_score"),
	"CNN-Affinity": Gnina("cnn_affinity"),
	"Vinardo": Vinardo(),
	"AD4": AD4(),
	"RFScoreVS": RFScoreVS(),
	"PLP": PLP(),
	"CHEMPLP": CHEMPLP(),
	"NNScore": NNScore(),
	"PLECScore": PLECScore(),
	"LinF9": LinF9(),
	"AAScore": AAScore(),
	"SCORCH": SCORCH(),
	"RTMScore": RTMScore(),
	"KORP-PL": KORPL(),
	"ConvexPLR": ConvexPLR(),
	"ITScoreAff": ITScoreAff(),
	"DLIGAND2": DLIGAND2(),
	"CENsible": CENsible(),
	"PANTHER": PANTHER("PANTHER"),
	"PANTHER-ESP": PANTHER("PANTHER-ESP"),
	"PANTHER-Shape": PANTHER("PANTHER-Shape"),
	"GenScore-scoring": GenScore("scoring"),
	"GenScore-docking": GenScore("docking"),
	"GenScore-balanced": GenScore("balanced"), }


def rescore_poses(w_dir: Path,
					protein_file: Path,
					pocket_definition: dict,
					software: Path,
					clustered_sdf: Path,
					functions: List[str],
					n_cpus: int) -> None:
	"""
    Rescores ligand poses using the specified scoring functions.
    """
	RDLogger.DisableLog("rdApp.*")
	tic = time.perf_counter()
	rescoring_folder_name = Path(clustered_sdf).stem
	rescoring_folder = w_dir / f"rescoring_{rescoring_folder_name}"
	rescoring_folder.mkdir(parents=True, exist_ok=True)

	skipped_functions = []
	for function in functions:
		scoring_function = RESCORING_FUNCTIONS.get(function)
		if not (rescoring_folder / f"{function}_rescoring" / f"{function}_scores.csv").is_file():
			try:
				scoring_function.rescore(clustered_sdf,
											n_cpus,
											protein_file=protein_file,
											pocket_definition=pocket_definition,
											software=software,
											rescoring_folder=rescoring_folder)
			except Exception as e:
				printlog(e)
				printlog(f"Failed for {function}")
		else:
			skipped_functions.append(function)
	if skipped_functions:
		printlog(f'Skipping functions: {", ".join(skipped_functions)}')

	score_files = [f"{function}_scores.csv" for function in functions]
	csv_files = [file for file in (rescoring_folder.rglob("*.csv")) if file.name in score_files]
	csv_dfs = []
	for file in csv_files:
		df = pd.read_csv(file)
		if "Unnamed: 0" in df.columns:
			df = df.drop(columns=["Unnamed: 0"])
		csv_dfs.append(df)

	if len(csv_dfs) == 1:
		combined_dfs = csv_dfs[0]
	elif len(csv_dfs) > 1:
		combined_dfs = csv_dfs[0]
		for df in tqdm(csv_dfs[1:], desc="Combining scores", unit="files"):
			combined_dfs = pd.merge(combined_dfs, df, on="Pose ID", how="inner")

	first_column = combined_dfs.pop("Pose ID")
	combined_dfs.insert(0, "Pose ID", first_column)
	columns = combined_dfs.columns
	col = columns[1:]
	for c in col.tolist():
		if c == "Pose ID":
			continue
		if combined_dfs[c].dtype != float:
			combined_dfs[c] = combined_dfs[c].apply(pd.to_numeric, errors="coerce")

	combined_dfs.to_csv(rescoring_folder / "allposes_rescored.csv", index=False)
	toc = time.perf_counter()
	printlog(f"Rescoring complete in {toc - tic:0.4f}!")


def rescore_docking(w_dir: Path,
					protein_file: Path,
					pocket_definition: dict,
					software: Path,
					function: str,
					n_cpus: int) -> pd.DataFrame:
	"""
    Rescores docking poses using the specified scoring function.
    """
	RDLogger.DisableLog("rdApp.*")
	tic = time.perf_counter()

	all_poses = Path(f"{w_dir}/allposes.sdf")
	scoring_function = RESCORING_FUNCTIONS.get(function)

	if scoring_function is None:
		raise ValueError(f"Unknown scoring function: {function}")

	score_df = scoring_function.rescore(all_poses,
										n_cpus,
										protein_file=protein_file,
										pocket_definition=pocket_definition,
										software=software,
										rescoring_folder=w_dir)

	score_df["Pose_Number"] = score_df["Pose ID"].str.split("_").str[2].astype(int)
	score_df["Docking_program"] = score_df["Pose ID"].str.split("_").str[1].astype(str)
	score_df["ID"] = score_df["Pose ID"].str.split("_").str[0].astype(str)

	if scoring_function.best_value == "min":
		best_pose_indices = score_df.groupby("ID")[scoring_function.column_name].idxmin()
	else:
		best_pose_indices = score_df.groupby("ID")[scoring_function.column_name].idxmax()

	score_file = w_dir / f"{function}_rescoring" / f"{function}_scores.csv"
	if score_file.exists():
		score_file.unlink()
	score_folder = w_dir / f"{function}_rescoring"
	if score_folder.exists():
		score_folder.rmdir()

	best_poses = pd.DataFrame(score_df.loc[best_pose_indices, "Pose ID"])
	toc = time.perf_counter()
	printlog(f"Rescoring complete in {toc - tic:0.4f}!")
	return best_poses
