import sys
import traceback
from pathlib import Path

import pandas as pd
from rdkit.Chem import PandasTools

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog
from scripts.utilities.subprocess_handler import run_subprocess_command

class NNScore(ScoringFunction):
    """
    NNScore scoring function implementation.
    """

    def __init__(self, software_path: Path):
        super().__init__(
            name="NNScore", column_name="NNScore", best_value="max", score_range=(0, 20), software_path=software_path
        )

    def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
        try:
            printlog("Running NNScore...")

            results = self._temp_dir / "rescored_NNscore.sdf"
            nnscore_cmd = (
                f"oddt_cli {sdf_file}"
                f" --receptor {protein_file}"
                f" -n {n_cpus}"
                f" --score nnscore"
                f" -O {results}"
            )

            stdout, stderr = run_subprocess_command(command=nnscore_cmd)

            if not results.exists():
                printlog(f"NNScore output file not found: {results}")
                if stderr:
                    printlog(f"NNScore command output:\n{stdout}")
                    printlog(f"NNScore command error output:\n{stderr}")
                return pd.DataFrame()

            nnscore_results_df = PandasTools.LoadSDF(
                str(results),
                idName="Pose ID",
                molColName=None,
                includeFingerprints=False,
                removeHs=False
            )

            nnscore_results_df.rename(columns={"nnscore": self.column_name}, inplace=True)
            return nnscore_results_df[["Pose ID", self.column_name]]

        except Exception:
            printlog("ERROR: An unexpected error occurred during NNScore rescoring:")
            printlog(traceback.format_exc())
            return pd.DataFrame()
        finally:
            self.cleanup()
