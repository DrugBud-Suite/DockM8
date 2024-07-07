import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

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
	"AAScore": AAScore(),
	"AD4": AD4(),
	"CENsible": CENsible(),
	"CHEMPLP": CHEMPLP(),
	"CNN-Affinity": Gnina("cnn_affinity"),
	"CNN-Score": Gnina("cnn_score"),
	"ConvexPLR": ConvexPLR(),
	"DLIGAND2": DLIGAND2(),
	"GenScore-balanced": GenScore("balanced"),
	"GenScore-docking": GenScore("docking"),
	"GenScore-scoring": GenScore("scoring"),
	"GNINA-Affinity": Gnina("affinity"),
	"ITScoreAff": ITScoreAff(),
	"KORP-PL": KORPL(),
	"LinF9": LinF9(),
	"NNScore": NNScore(),
	"PANTHER": PANTHER("PANTHER"),
	"PANTHER-ESP": PANTHER("PANTHER-ESP"),
	"PANTHER-Shape": PANTHER("PANTHER-Shape"),
	"PLECScore": PLECScore(),
	"PLP": PLP(),
	"RFScoreVS": RFScoreVS(),
	"RTMScore": RTMScore(),
	"SCORCH": SCORCH(),
	"Vinardo": Vinardo(), }


def rescore_poses(sdf: Path,
					protein_file: Path,
					pocket_definition: dict,
					software: Path,
					functions: List[str],
					n_cpus: int,
					output_file: Optional[Path] = None) -> pd.DataFrame:
	"""
    Rescores ligand poses using the specified scoring functions.
    """
	RDLogger.DisableLog("rdApp.*")
	tic = time.perf_counter()

	skipped_functions = []
	results = []
	for function in functions:
		scoring_function = RESCORING_FUNCTIONS.get(function)
		if scoring_function:
			try:
				result = scoring_function.rescore(sdf,
													n_cpus,
													protein_file=protein_file,
													pocket_definition=pocket_definition,
													software=software)
				results.append(result)
			except Exception as e:
				printlog(e)
				printlog(f"Failed for {function}")
		else:
			skipped_functions.append(function)

	if skipped_functions:
		printlog(f'Skipping functions: {", ".join(skipped_functions)}')

	if len(results) == 1:
		combined_results = results[0]
	elif len(results) > 1:
		combined_results = results[0]
		for df in tqdm(results[1:], desc="Combining scores", unit="files"):
			combined_results = pd.merge(combined_results, df, on="Pose ID", how="inner")

	first_column = combined_results.pop("Pose ID")
	combined_results.insert(0, "Pose ID", first_column)
	columns = combined_results.columns
	col = columns[1:]
	for c in col.tolist():
		if c == "Pose ID":
			continue
		if combined_results[c].dtype != float:
			combined_results[c] = combined_results[c].apply(pd.to_numeric, errors="coerce")

	if output_file:
		combined_results.to_csv(output_file, index=False)

	toc = time.perf_counter()
	printlog(f"Rescoring complete in {toc - tic:0.4f}!")

	return combined_results


def rescore_docking(
    sdf: Path,
    protein_file: Path,
    pocket_definition: dict,
    software: Path,
    function: str,
    n_cpus: int
) -> pd.DataFrame:
    """
    Rescores docking poses using the specified scoring function.
    """
    RDLogger.DisableLog("rdApp.*")
    tic = time.perf_counter()

    scoring_function = RESCORING_FUNCTIONS.get(function)

    if scoring_function is None:
        raise ValueError(f"Unknown scoring function: {function}")

    score_df = scoring_function.rescore(
        sdf,
        n_cpus,
        protein_file=protein_file,
        pocket_definition=pocket_definition,
        software=software
    )

    score_df["Pose_Number"] = score_df["Pose ID"].str.split("_").str[2].astype(int)
    score_df["Docking_program"] = score_df["Pose ID"].str.split("_").str[1].astype(str)
    score_df["ID"] = score_df["Pose ID"].str.split("_").str[0].astype(str)

    if scoring_function.best_value == "min":
        best_pose_indices = score_df.groupby("ID")[scoring_function.column_name].idxmin()
    else:
        best_pose_indices = score_df.groupby("ID")[scoring_function.column_name].idxmax()

    best_poses = pd.DataFrame(score_df.loc[best_pose_indices, "Pose ID"])
    
    toc = time.perf_counter()
    printlog(f"Rescoring complete in {toc - tic:0.4f}!")
    return best_poses
