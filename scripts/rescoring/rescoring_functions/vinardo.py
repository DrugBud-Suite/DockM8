from pathlib import Path
import os
import pandas as pd
from rdkit.Chem import PandasTools

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.subprocess_handler import run_subprocess_command

class Vinardo(ScoringFunction):
    """Vinardo scoring function implementation."""

    def __init__(self, software_path: Path):
        super().__init__(
            name="Vinardo",
            column_name="Vinardo",
            best_value="min",
            score_range=(200, 20),
            software_path=software_path
        )

    def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
        """
        Rescore the molecules in the given SDF file using the Vinardo scoring function.

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

            vinardo_dataframes = self._load_rescoring_results(rescoring_results)
            return self._combine_rescoring_results(vinardo_dataframes)

        except Exception as e:
            printlog("ERROR: An unexpected error occurred during Vinardo rescoring:")
            printlog(str(e))
            return pd.DataFrame()
        finally:
            self.cleanup()

    def _rescore_split_file(self, split_file: Path, protein_file: str) -> Path | None:
        results = split_file.parent / f"{split_file.stem}_{self.column_name}.sdf"
        vinardo_cmd = (
            f"{self.software_path}/gnina"
            f" --receptor {protein_file}"
            f" --ligand {split_file}"
            f" --out {results}"
            " --score_only"
            " --scoring vinardo"
            " --cnn_scoring none"
        )

        stdout, stderr = run_subprocess_command(command=vinardo_cmd)
        
        if not results.exists():
            printlog(f"Vinardo output file not found: {results}")
            if stderr:
                printlog(f"Vinardo command output:\n{stdout}")
                printlog(f"Vinardo command error output:\n{stderr}")
            return None

        return results

    def _load_rescoring_results(self, result_files: list[Path]) -> list[pd.DataFrame]:
        """
        Load rescoring results from SDF files.

        Args:
            result_files (List[Path]): List of paths to rescored SDF files.

        Returns:
            List[pd.DataFrame]: List of DataFrames containing the rescoring results.
        """
        dataframes = []
        for file in result_files:
            if not file or not file.is_file():
                continue

            try:
                df = PandasTools.LoadSDF(
                    str(file),
                    idName="Pose ID",
                    molColName=None,
                    includeFingerprints=False,
                    embedProps=False
                )
                dataframes.append(df)
            except Exception as e:
                printlog(f"Error loading Vinardo results from {file}: {str(e)}")

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
                printlog("No valid Vinardo results to combine")
                return pd.DataFrame()

            combined_results = pd.concat(dataframes, ignore_index=True)
            combined_results.rename(columns={"minimizedAffinity": self.column_name}, inplace=True)
            return combined_results[["Pose ID", self.column_name]]

        except Exception as e:
            printlog(f"Error combining Vinardo results: {str(e)}")
            return pd.DataFrame()
