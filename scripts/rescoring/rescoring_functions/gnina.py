import subprocess
import sys
import traceback
from pathlib import Path
import os

import pandas as pd
from rdkit.Chem import PandasTools

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.path_check import get_executable_path

class Gnina(ScoringFunction):
    """
    Gnina scoring function implementation.
    """

    def __init__(self, score_type: str, software_path: Path):
        if score_type == "affinity":
            super().__init__(
                name="GNINA-Affinity",
                column_name="GNINA-Affinity",
                best_value="min",
                score_range=(100, -100),
                software_path=software_path,
            )
        elif score_type == "cnn_score":
            super().__init__(
                name="CNN-Score",
                column_name="CNN-Score",
                best_value="max",
                score_range=(0, 1),
                software_path=software_path,
            )
        elif score_type == "cnn_affinity":
            super().__init__(
                name="CNN-Affinity",
                column_name="CNN-Affinity",
                best_value="max",
                score_range=(0, 20),
                software_path=software_path,
            )
        else:
            raise ValueError("Invalid score type for Gnina")
        self.score_type = score_type
        self.software_path = software_path
        self.executable_path = get_executable_path(software_path, "gnina")

    def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
        """
        Rescore the molecules in the given SDF file using the Gnina scoring function.

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
                self._rescore_split_file, split_files_sdfs, n_cpus, display_name=self.name, protein_file=protein_file
            )

            gnina_dataframes = self._load_rescoring_results(rescoring_results)
            gnina_rescoring_results = self._combine_rescoring_results(gnina_dataframes)

            
            
            return gnina_rescoring_results
        except Exception:
            printlog(f"ERROR: An unexpected error occurred during {self.name} rescoring:")
            printlog(traceback.format_exc())
            return pd.DataFrame()
        finally:
            self.cleanup()

    def _rescore_split_file(self, split_file: Path, protein_file: str) -> Path:
        """
        Rescore a single split SDF file.

        Args:
            split_file (Path): The path to the split SDF file.
            protein_file (str): The path to the protein file.

        Returns:
            Path: The path to the rescored SDF file.
        """
        results = split_file.parent / f"{split_file.stem}_{self.column_name}.sdf"
        gnina_cmd = (
            f"{self.executable_path}"
            f" --receptor {protein_file}"
            f" --ligand {split_file}"
            f" --out {results}"
            " --cpu 1"
            " --score_only"
            " --cnn crossdock_default2018"
        )
        try:
            subprocess.run(gnina_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            printlog(f"{self.column_name} rescoring failed for {split_file}:")
            printlog(traceback.format_exc())
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
            try:
                df = PandasTools.LoadSDF(
                    str(file), idName="Pose ID", molColName=None, includeFingerprints=False, embedProps=False
                )
                dataframes.append(df)
            except Exception:
                printlog(f"ERROR: Failed to Load {self.column_name} rescoring SDF file: {file}")
                printlog(traceback.format_exc())
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
            combined_results = pd.concat(dataframes, ignore_index=True)
            combined_results.rename(
                columns={"minimizedAffinity": "GNINA-Affinity", "CNNscore": "CNN-Score", "CNNaffinity": "CNN-Affinity"},
                inplace=True,
            )
            return combined_results[["Pose ID", self.column_name]]
        except Exception:
            printlog(f"ERROR: Could not combine {self.column_name} rescored poses")
            printlog(traceback.format_exc())
            return pd.DataFrame()
