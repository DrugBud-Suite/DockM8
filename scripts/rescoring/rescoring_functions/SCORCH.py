import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.molecule_conversion import convert_molecules


class SCORCH(ScoringFunction):
    """
    SCORCH scoring function implementation.
    """

    def __init__(self, software_path: Path):
        super().__init__(
            name="SCORCH", column_name="SCORCH", best_value="max", score_range=(0, 1), software_path=software_path
        )

    def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
        """
        Rescore the molecules in the given SDF file using the SCORCH scoring function.

        Args:
        sdf_file (str): The path to the SDF file.
        n_cpus (int): The number of CPUs to use for parallel processing.
        protein_file (str): The path to the protein file.
        **kwargs: Additional keyword arguments.

        Returns:
        pd.DataFrame: A DataFrame containing the rescored molecules.
        """
        start_time = time.perf_counter()

        try:
            # Convert protein to PDBQT
            scorch_protein = Path(self._temp_dir) / "protein.pdbqt"
            try:
                convert_molecules(Path(protein_file), scorch_protein, "pdb", "pdbqt")
            except Exception:
                printlog("Error converting protein file to .pdbqt:")
                printlog(traceback.format_exc())
                return pd.DataFrame()

            # Split input SDF into individual compounds
            split_files_folder = split_sdf(sdf_file, self._temp_dir, mode="single", splits=1)
            split_files = list(split_files_folder.glob("*.sdf"))

            if not split_files:
                printlog("No split files were created")
                return pd.DataFrame()

            # Convert each split SDF to PDBQT
            for sdf_path in split_files:
                pdbqt_path = sdf_path.with_suffix(".pdbqt")
                try:
                    convert_molecules(sdf_path, pdbqt_path, "sdf", "pdbqt")
                except Exception as e:
                    printlog(f"Error converting {sdf_path} to PDBQT: {str(e)}")
                    printlog(traceback.format_exc())

            # Verify PDBQT files were created
            pdbqt_files = list(split_files_folder.glob("*.pdbqt"))
            if not pdbqt_files:
                printlog("No PDBQT files were created during conversion")
                return pd.DataFrame()

            # Run SCORCH on each compound
            rescoring_results = parallel_executor(
                self._rescore_split_file, pdbqt_files, n_cpus, display_name=self.name, scorch_protein=scorch_protein
            )

            # Process results
            scorch_dataframes = self._load_rescoring_results(rescoring_results)
            scorch_rescoring_results = self._combine_rescoring_results(scorch_dataframes)

            
            
            return scorch_rescoring_results

        except Exception:
            printlog("ERROR: An unexpected error occurred during SCORCH rescoring:")
            printlog(traceback.format_exc())
            return pd.DataFrame()
        finally:
            self.cleanup()

    def _rescore_split_file(self, pdbqt_file: Path, scorch_protein: Path) -> Path:
        """
        Rescore a single compound SDF file.

        Args:
        pdbqt_file (Path): The path to the split SDF file containing a single compound.
        scorch_protein (Path): The path to the prepared protein file.

        Returns:
        Path: The path to the rescored SDF file.
        """
        try:
            results = pdbqt_file.parent / f"{pdbqt_file.stem}_{self.column_name}.csv"

            if not pdbqt_file.exists():
                printlog(f"PDBQT file not found: {pdbqt_file}")
                return results

            scorch_cmd = (
                f"cd {self.software_path}/SCORCH-1.0.0 &&"
                " python scorch.py --receptor"
                f" {scorch_protein} --ligand"
                f" {pdbqt_file} --out"
                f" {results}"
            )

            # Run the command and capture output
            result = subprocess.run(scorch_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        except Exception:
            printlog(f"{self.column_name} rescoring failed for {pdbqt_file}:")
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
            if not file.exists():
                printlog(f"Warning: Result file does not exist: {file}")
                continue

            try:
                df = pd.read_csv(file)
                if not df.empty:
                    dataframes.append(df)
                else:
                    printlog(f"Warning: Empty DataFrame loaded from {file}")
            except Exception:
                printlog(f"ERROR: Failed to Load {self.column_name} rescoring CSV file: {file}")
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
            if not dataframes:
                printlog("Warning: No valid DataFrames to combine")
                return pd.DataFrame(columns=["Pose ID", self.column_name])

            combined_results = pd.concat(dataframes, ignore_index=True)
            combined_results.rename(columns={"SCORCH_score": self.column_name, "Ligand_ID": "Pose ID"}, inplace=True)
            return combined_results[["Pose ID", self.column_name]]
        except Exception:
            printlog(f"ERROR: Could not combine {self.column_name} rescored poses")
            printlog(traceback.format_exc())
            return pd.DataFrame(columns=["Pose ID", self.column_name])
