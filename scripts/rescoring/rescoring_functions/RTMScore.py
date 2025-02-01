from pathlib import Path
import pandas as pd

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog
from scripts.utilities.subprocess_handler import run_subprocess_command

class RTMScore(ScoringFunction):
    """RTMScore scoring function implementation."""

    def __init__(self, software_path: Path):
        super().__init__(
            name="RTMScore",
            column_name="RTMScore",
            best_value="max",
            score_range=(0, 100),
            software_path=software_path
        )
        self.software_path = software_path
        self.rtmscore_script = self.software_path / "RTMScore-main/example/rtmscore.py"
        self.model_path = self.software_path / "RTMScore-main/trained_models/rtmscore_model1.pth"

    def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
        try:
            if not self.rtmscore_script.is_file():
                raise FileNotFoundError(
                    "RTMScore script not found. Please ensure RTMScore is properly installed."
                )

            rtmscore_results = Path(self._temp_dir) / f"{self.column_name}_scores"
            pocket_file = Path(str(protein_file).replace(".pdb", "_pocket.pdb"))
            if not pocket_file.is_file():
                raise FileNotFoundError(f"Pocket file not found at {pocket_file}")

            rtmscore_cmd = (
                f"cd {self.software_path}/RTMScore-main/example/ &&"
                f" python rtmscore.py"
                f" -p {pocket_file}"
                f" -l {sdf_file}"
                f" -o {rtmscore_results}"
                " -pl"
                f" -m {self.model_path}"
            )

            output_csv = rtmscore_results.with_suffix(".csv")
            stdout, stderr = run_subprocess_command(command=rtmscore_cmd)

            if not output_csv.exists():
                printlog(f"RTMScore output file not found: {output_csv}")
                if stderr:
                    printlog(f"RTMScore command output:\n{stdout}")
                    printlog(f"RTMScore command error output:\n{stderr}")
                return pd.DataFrame()

            try:
                rtmscore_df = pd.read_csv(output_csv)
                rtmscore_df = rtmscore_df.rename(columns={
                    "id": "Pose ID",
                    "score": self.column_name
                })
                rtmscore_df["Pose ID"] = rtmscore_df["Pose ID"].str.rsplit("-", n=1).str[0]
                return rtmscore_df[["Pose ID", self.column_name]]

            except Exception as e:
                printlog(f"Error processing RTMScore results: {str(e)}")
                return pd.DataFrame()

        except Exception as e:
            printlog("ERROR: An unexpected error occurred during RTMScore rescoring:")
            printlog(str(e))
            return pd.DataFrame()
        finally:
            self.cleanup()
