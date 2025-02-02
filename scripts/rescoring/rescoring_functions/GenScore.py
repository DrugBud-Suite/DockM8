from pathlib import Path
import os
import pandas as pd

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.pocket_finding.utils import extract_pocket
from scripts.utilities.subprocess_handler import run_subprocess_command

class GenScore(ScoringFunction):
    """GenScore scoring function implementation."""

    def __init__(self, score_type: str, software_path: Path):
        genscore_path = software_path / "GenScore"
        self.software_path = software_path

        model_configs = {
            "scoring": ("GenScore-scoring", "GatedGCN_ft_1.0_1.pth", "gatedgcn"),
            "docking": ("GenScore-docking", "GT_0.0_1.pth", "gt"),
            "balanced": ("GenScore-balanced", "GT_ft_0.5_1.pth", "gt")
        }

        if score_type not in model_configs:
            raise ValueError(f"Invalid GenScore type: {score_type}")

        name, model_file, encoder = model_configs[score_type]
        super().__init__(
            name=name,
            column_name=name,
            best_value="max",
            score_range=(0, 200),
            software_path=genscore_path,
        )
        self.model = genscore_path / "trained_models" / model_file
        self.encoder = encoder

    def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
        """
        Rescore the molecules in the given SDF file using the GenScore scoring function.

        Args:
            sdf_file (str): The path to the SDF file.
            n_cpus (int): The number of CPUs to use for parallel processing.
            protein_file (str): The path to the protein file.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: A DataFrame containing the rescored molecules.
        """
        try:
            pocket_file = Path(str(protein_file).replace(".pdb", "_pocket.pdb"))
            if not pocket_file.is_file():
                pocket_definition = kwargs.get("pocket_definition")
                if not pocket_definition:
                    raise ValueError("Pocket definition is required when pocket file is not available")
                pocket_file = extract_pocket(pocket_definition, protein_file)

            split_files_folder = split_sdf(sdf_file, self._temp_dir, mode="count", splits=1)
            split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

            rescoring_results = parallel_executor(
                self._rescore_split_file,
                split_files_sdfs,
                n_cpus,
                display_name=self.name,
                pocket_file=pocket_file
            )

            if len(rescoring_results) > 0:
                combined_results = self._combine_rescoring_results(rescoring_results)
                return combined_results
            else:
                raise ValueError("No rescoring results found")

        except Exception as e:
            printlog(f"ERROR: An unexpected error occurred during {self.name} rescoring:")
            printlog(str(e))
            return pd.DataFrame()
        finally:
            self.cleanup()

    def _rescore_split_file(self, split_file: Path, pocket_file: Path) -> Path | None:
        """
        Rescore a single split SDF file.

        Args:
            split_file (Path): The path to the split SDF file.
            pocket_file (Path): The path to the protein pocket file.

        Returns:
            Path | None: The path to the rescored CSV file or None if rescoring failed.
        """
        try:
            output_file = split_file.parent / f"{split_file.stem}.csv"
            genscore_cmd = (
                f"cd {self.software_path}/example/ &&"
                f" conda run -n genscore python genscore.py"
                f" -p {pocket_file}"
                f" -l {split_file}"
                f" -o {split_file.parent / split_file.stem}"
                f" -m {self.model}"
                f" -e {self.encoder}"
            )

            stdout, stderr = run_subprocess_command(command=genscore_cmd)
            
            if not output_file.exists():
                printlog(f"GenScore output file not found: {output_file}")
                if stderr:
                    printlog(f"GenScore command ouput:\n{stdout}")
                    printlog(f"GenScore command error output:\n{stderr}")
                return None

            return output_file

        except Exception as e:
            printlog(f"Error in GenScore rescoring for {split_file}: {str(e)}")
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
                printlog(f"No valid CSV files found with {self.name} scores.")
                return pd.DataFrame()

            combined_results = pd.concat(dataframes, ignore_index=True)
            combined_results.rename(columns={"id": "Pose ID", "score": self.column_name}, inplace=True)
            combined_results["Pose ID"] = combined_results["Pose ID"].apply(lambda x: x.split("-")[0])
            return combined_results[["Pose ID", self.column_name]]

        except Exception as e:
            printlog(f"ERROR: Could not combine {self.column_name} rescored poses: {str(e)}")
            return pd.DataFrame()
