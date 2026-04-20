import glob
from pathlib import Path
import os
import pandas as pd

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.subprocess_handler import run_subprocess_command

class RFScoreVS(ScoringFunction):
    """RFScoreVS scoring function implementation."""

    def __init__(self, software_path: Path):
        super().__init__(
            name="RFScoreVS",
            column_name="RFScoreVS",
            best_value="max",
            score_range=(5, 10),
            software_path=software_path,
        )
        self.software_path = software_path

    def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
        """
        Rescore the molecules in the given SDF file using the RFScoreVS scoring function.

        Args:
            sdf_file (str): The path to the SDF file.
            n_cpus (int): The number of CPUs to use for parallel processing.
            protein_file (str): The path to the protein file.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: A DataFrame containing the rescored molecules.
        """
        try:
            split_files_folder = split_sdf(sdf_file, self._temp_dir, mode="cpu", splits=n_cpus)
            split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

            rescoring_results = parallel_executor(
                self._rescore_split_file,
                split_files_sdfs,
                n_cpus,
                display_name=self.name,
                protein_file=protein_file
            )

            rfscorevs_dataframes = self._load_rescoring_results(rescoring_results)
            return self._combine_rescoring_results(rfscorevs_dataframes)

        except Exception as e:
            printlog("ERROR: An unexpected error occurred during RFScoreVS rescoring:")
            printlog(str(e))
            return pd.DataFrame()
        finally:
            self._cleanup_all()

    def _cleanup_all(self):
        """Perform cleanup of temporary files including compiledtrees files."""
        try:
            self.cleanup()
            compiledtrees_files = glob.glob("/tmp/compiledtrees*")
            for file in compiledtrees_files:
                try:
                    os.remove(file)
                except OSError as e:
                    printlog(f"Failed to remove compiledtrees file {file}: {str(e)}")
        except Exception as e:
            printlog(f"Error during cleanup: {str(e)}")

    def _rescore_split_file(self, split_file: Path, protein_file: str) -> Path | None:
        results = split_file.parent / f"{split_file.stem}_RFScoreVS_scores.csv"
        rfscorevs_cmd = (
            f"{self.software_path}/rf-score-vs"
            f" --receptor {protein_file}"
            f" {split_file}"
            f" -O {results}"
            " -n 1"
        )

        stdout, stderr = run_subprocess_command(command=rfscorevs_cmd)
        
        if not results.exists():
            printlog(f"RFScoreVS output file not found: {results}")
            if stderr:
                printlog(f"RFScoreVS command output:\n{stdout}")
                printlog(f"RFScoreVS command error output:\n{stderr}")
            return None

        return results

    def _load_rescoring_results(self, result_files: list[Path]) -> list[pd.DataFrame]:
        """
        Load rescoring results from CSV files.

        Args:
            result_files (List[Path]): List of paths to rescored CSV files.

        Returns:
            List[pd.DataFrame]: List of DataFrames containing the rescoring results.
        """
        dataframes = []
        for file in result_files:
            if not file or not file.is_file():
                continue
                
            try:
                df = pd.read_csv(file, delimiter=",", header=0)
                dataframes.append(df)
            except Exception as e:
                printlog(f"ERROR: Failed to load {self.column_name} rescoring CSV file {file}: {str(e)}")
                
        return dataframes

    def _combine_rescoring_results(self, dataframes: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine rescoring results from multiple DataFrames.

        Args:
            dataframes (List[pd.DataFrame]): List of DataFrames containing rescoring results.

        Returns:
            pd.DataFrame: Combined DataFrame with rescoring results.
        """
        try:
            if not dataframes:
                printlog(f"No valid {self.name} rescoring results to combine")
                return pd.DataFrame()

            combined_results = pd.concat(dataframes, ignore_index=True)
            rfscorevs_column = next(
                (col for col in combined_results.columns if "RFScoreVS" in col),
                None
            )

            if not rfscorevs_column:
                raise ValueError("No column containing 'RFScoreVS' found in the results")

            combined_results.rename(
                columns={"name": "Pose ID", rfscorevs_column: self.column_name},
                inplace=True
            )
            return combined_results[["Pose ID", self.column_name]]

        except Exception as e:
            printlog(f"ERROR: Could not combine {self.column_name} rescored poses: {str(e)}")
            return pd.DataFrame()
