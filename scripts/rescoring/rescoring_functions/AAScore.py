from pathlib import Path
import pandas as pd
from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.subprocess_handler import run_subprocess_command

class AAScore(ScoringFunction):
    """AAScore scoring function implementation."""

    def __init__(self, software_path: Path):
        super().__init__(
            name="AAScore",
            column_name="AAScore",
            best_value="max",
            score_range=(100, -100),
            software_path=software_path,
        )

    def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
        """
        Rescore the molecules in the given SDF file using the AAScore scoring function.

        Args:
            sdf_file (str): The path to the SDF file.
            n_cpus (int): The number of CPUs to use for parallel processing.
            protein_file (str): The path to the protein file.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: A DataFrame containing the rescored molecules.
        """
        try:
            pocket_file = str(protein_file).replace(".pdb", "_pocket.pdb")

            if n_cpus == 1:
                aascore_rescoring_results = self._rescore_single_process(sdf_file, pocket_file, self._temp_dir)
            else:
                aascore_rescoring_results = self._rescore_multi_process(sdf_file, pocket_file, n_cpus, self._temp_dir)

            return aascore_rescoring_results
        except Exception as e:
            printlog("ERROR: An unexpected error occurred during AAScore rescoring:")
            printlog(str(e))
            return pd.DataFrame()
        finally:
            self.cleanup()

    def _rescore_single_process(self, sdf_file: str, pocket_file: str, temp_dir: Path) -> pd.DataFrame:
        results = temp_dir / "rescored_AAScore.csv"
        aascore_cmd = (
            f"cd {self.software_path}/AA-Score-Tool-main &&"
            f" conda run -n AAScore python AA_Score.py"
            f" --Rec {pocket_file}"
            f" --Lig {sdf_file}"
            f" --Out {results}"
        )

        stdout, stderr = run_subprocess_command(aascore_cmd)
        
        if not results.exists():
            printlog(f"AAScore output file not found: {results}")
            if stderr:
                printlog(f"AAScore command output:\n{stdout}")
                printlog(f"AAScore command error output:\n{stderr}")
            return pd.DataFrame()

        try:
            return pd.read_csv(results, delimiter="\t", header=None, names=["Pose ID", self.column_name])
        except Exception as e:
            printlog(f"Failed to read AAScore results from {results}: {str(e)}")
            return pd.DataFrame()

    def _rescore_multi_process(self, sdf_file: str, pocket_file: str, n_cpus: int, temp_dir: Path) -> pd.DataFrame:
        """
        Rescore using multiple processes.

        Args:
            sdf_file (str): The path to the SDF file.
            pocket_file (str): The path to the pocket file.
            n_cpus (int): The number of CPUs to use for parallel processing.
            temp_dir (Path): The temporary directory path.

        Returns:
            pd.DataFrame: A DataFrame containing the rescoring results.
        """
        split_files_folder = split_sdf(sdf_file, self._temp_dir, mode="cpus", splits=n_cpus)
        split_files_sdfs = list(split_files_folder.glob("*.sdf"))

        rescoring_results = parallel_executor(
            self._rescore_split_file, split_files_sdfs, n_cpus, display_name=self.name, pocket_file=pocket_file
        )

        return self._combine_rescoring_results(rescoring_results)

    def _rescore_split_file(self, split_file: Path, pocket_file: str) -> Path | None:
        results = split_file.parent / f"{split_file.stem}_AAScore.csv"
        aascore_cmd = (
            f"cd {self.software_path}/AA-Score-Tool-main &&"
            f" conda run -n AAScore python AA_Score.py"
            f" --Rec {pocket_file}"
            f" --Lig {split_file}"
            f" --Out {results}"
        )

        stdout, stderr = run_subprocess_command(aascore_cmd)
        
        if not results.exists():
            printlog(f"AAScore output file not found: {results}")
            if stderr:
                printlog(f"AAScore command output:\n{stdout}")
                printlog(f"AAScore command error output:\n{stderr}")
            return None

        return results

    def _combine_rescoring_results(self, result_files: list[Path]) -> pd.DataFrame:
        """
        Combine rescoring results from multiple CSV files.

        Args:
            result_files (List[Path]): List of paths to rescored CSV files.

        Returns:
            pd.DataFrame: Combined DataFrame with rescoring results.
        """
        try:
            dataframes = []
            for file in result_files:
                if file and file.is_file():
                    df = pd.read_csv(file, delimiter="\t", header=None, names=["Pose ID", self.column_name])
                    dataframes.append(df)

            if not dataframes:
                printlog("No valid CSV files found with AAScore scores.")
                return pd.DataFrame()

            combined_results = pd.concat(dataframes, ignore_index=True)
            return combined_results[["Pose ID", self.column_name]]
        except Exception as e:
            printlog(f"ERROR: Could not combine {self.column_name} rescored poses: {str(e)}")
            return pd.DataFrame()
