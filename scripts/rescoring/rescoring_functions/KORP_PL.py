from pathlib import Path
import os
import pandas as pd
from rdkit.Chem import PandasTools

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.subprocess_handler import run_subprocess_command

class KORPL(ScoringFunction):
    """KORP-PL scoring function implementation."""

    def __init__(self, software_path: Path):
        super().__init__(
            name="KORP-PL",
            column_name="KORP-PL",
            best_value="min",
            score_range=(200, -1000),
            software_path=software_path,
        )

    def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
        """
        Rescore the molecules in the given SDF file using the KORP-PL scoring function.

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

            return self._combine_rescoring_results(rescoring_results)

        except Exception as e:
            printlog("ERROR: An unexpected error occurred during KORP-PL rescoring:")
            printlog(str(e))
            return pd.DataFrame()
        finally:
            self.cleanup()

    def _rescore_split_file(self, split_file: Path, protein_file: str) -> Path | None:
        try:
            df = PandasTools.LoadSDF(str(split_file), idName="Pose ID", molColName=None)
            df = df[["Pose ID"]]

            mol2_file = convert_molecules(split_file, split_file.with_suffix(".mol2"), "sdf", "mol2")
            output_csv = split_file.parent / f"{split_file.stem}_scores.csv"

            korpl_cmd = (
                f"{self.software_path}/KORP-PL"
                f" --receptor {protein_file}"
                f" --ligand {mol2_file}"
                " --mol2"
            )

            stdout, stderr = run_subprocess_command(command=korpl_cmd)
            
            if not stdout or "model" not in stdout:
                printlog(f"KORP-PL output not found or invalid for {split_file}")
                if stderr:
                    printlog(f"KORP-PL command output:\n{stdout}")
                    printlog(f"KORP-PL command error output:\n{stderr}")
                return None

            try:
                energies = []
                for line in stdout.splitlines():
                    if line.startswith("model"):
                        parts = line.split(",")
                        energy = round(float(parts[1].split("=")[1]), 2)
                        energies.append(energy)

                df[self.column_name] = energies
                df.to_csv(output_csv, index=False)
                return output_csv

            except (ValueError, IndexError) as e:
                printlog(f"Error parsing KORP-PL output for {split_file}: {str(e)}")
                return None

        except Exception as e:
            printlog(f"KORP-PL rescoring failed for {split_file}: {str(e)}")
            return None

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
                if not file or not file.is_file():
                    continue

                try:
                    df = pd.read_csv(file)
                    dataframes.append(df)
                except Exception as e:
                    printlog(f"Failed to read results from {file}: {str(e)}")
                    continue

            if not dataframes:
                printlog("No valid CSV files found with KORP-PL scores.")
                return pd.DataFrame()

            combined_results = pd.concat(dataframes, ignore_index=True)
            return combined_results[["Pose ID", self.column_name]]

        except Exception as e:
            printlog(f"ERROR: Could not combine {self.column_name} rescored poses: {str(e)}")
            return pd.DataFrame()
