from pathlib import Path
import pandas as pd

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.subprocess_handler import run_subprocess_command

class SCORCH(ScoringFunction):
    """SCORCH scoring function implementation."""

    def __init__(self, software_path: Path):
        super().__init__(
            name="SCORCH",
            column_name="SCORCH",
            best_value="max",
            score_range=(0, 1),
            software_path=software_path
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
        try:
            # Convert protein to PDBQT format
            scorch_protein = Path(self._temp_dir) / "protein.pdbqt"
            try:
                convert_molecules(Path(protein_file), scorch_protein, "pdb", "pdbqt")
            except Exception as e:
                printlog(f"Error converting protein file to PDBQT: {str(e)}")
                return pd.DataFrame()

            # Split input SDF and convert to PDBQT
            split_files_folder = split_sdf(sdf_file, self._temp_dir, mode="single", splits=1)
            split_files = list(split_files_folder.glob("*.sdf"))

            if not split_files:
                printlog("No split files were created during SDF splitting")
                return pd.DataFrame()

            # Convert each split SDF to PDBQT format
            pdbqt_files = []
            for sdf_path in split_files:
                pdbqt_path = sdf_path.with_suffix(".pdbqt")
                try:
                    convert_molecules(sdf_path, pdbqt_path, "sdf", "pdbqt")
                    pdbqt_files.append(pdbqt_path)
                except Exception as e:
                    printlog(f"Error converting {sdf_path} to PDBQT: {str(e)}")
                    continue

            if not pdbqt_files:
                printlog("No PDBQT files were created during conversion")
                return pd.DataFrame()

            # Execute SCORCH scoring
            rescoring_results = parallel_executor(
                self._rescore_split_file,
                pdbqt_files,
                n_cpus,
                display_name=self.name,
                scorch_protein=scorch_protein
            )

            # Process and combine results
            scorch_dataframes = self._load_rescoring_results(rescoring_results)
            return self._combine_rescoring_results(scorch_dataframes)

        except Exception as e:
            printlog("ERROR: An unexpected error occurred during SCORCH rescoring:")
            printlog(str(e))
            return pd.DataFrame()
        finally:
            self.cleanup()

    def _rescore_split_file(self, pdbqt_file: Path, scorch_protein: Path) -> Path | None:
        try:
            results = pdbqt_file.parent / f"{pdbqt_file.stem}_{self.column_name}.csv"
            scorch_cmd = (
                f"cd {self.software_path}/SCORCH-1.0.0 &&"
                " python scorch.py --receptor"
                f" {scorch_protein} --ligand"
                f" {pdbqt_file} --out"
                f" {results}"
            )

            stdout, stderr = run_subprocess_command(command=scorch_cmd)

            if not results.exists():
                printlog(f"SCORCH output file not found: {results}")
                if stderr:
                    printlog(f"SCORCH command output:\n{stdout}")
                    printlog(f"SCORCH command error output:\n{stderr}")
                return None

            return results

        except Exception as e:
            printlog(f"Error in SCORCH rescoring for {pdbqt_file}: {str(e)}")
            return None

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
                df = pd.read_csv(file)
                if not df.empty:
                    dataframes.append(df)
                else:
                    printlog(f"Empty DataFrame found in {file}")
            except Exception as e:
                printlog(f"Error loading SCORCH results from {file}: {str(e)}")

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
                printlog("No valid SCORCH results to combine")
                return pd.DataFrame(columns=["Pose ID", self.column_name])

            combined_results = pd.concat(dataframes, ignore_index=True)
            combined_results.rename(
                columns={"SCORCH_score": self.column_name, "Ligand_ID": "Pose ID"},
                inplace=True
            )
            return combined_results[["Pose ID", self.column_name]]

        except Exception as e:
            printlog(f"Error combining SCORCH results: {str(e)}")
            return pd.DataFrame(columns=["Pose ID", self.column_name])
