"""This module implements GPU-enabled GNINA docking with sequential batch processing."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.docking.docking_function import DockingFunction
from scripts.utilities.logging import printlog
from scripts.utilities.path_check import get_executable_path


class GninaGPUDocking(DockingFunction):
    """A class to perform GPU-accelerated docking using GNINA."""

    def __init__(self, software_path: Path):
        """Initializes the GninaGPUDocking class.

        Args:
            software_path (Path): The base path to the GNINA software installation.
        """
        super().__init__("GNINA_GPU", software_path)
        self.software_path = software_path
        self.executable_path = get_executable_path(software_path, "gnina")

    def dock_batch(
        self,
        batch_file: Path,
        protein_file: Path,
        pocket_definition: dict[str, list],
        exhaustiveness: int,
        n_poses: int,
    ) -> Path | None:
        RDLogger.DisableLog("rdApp.*")

        raw_results_path = self._temp_dir / "raw" / f"{batch_file.stem}_raw.sdf"
        processed_results_path = (
            self._temp_dir / "processed" / f"{batch_file.stem}_processed.sdf"
        )

        try:
            gnina_cmd = [
                str(self.executable_path),
                "--receptor",
                str(protein_file),
                "--ligand",
                str(batch_file),
                "--out",
                str(raw_results_path),
                "--center_x",
                str(pocket_definition["center"][0]),
                "--center_y",
                str(pocket_definition["center"][1]),
                "--center_z",
                str(pocket_definition["center"][2]),
                "--size_x",
                str(pocket_definition["size"][0]),
                "--size_y",
                str(pocket_definition["size"][1]),
                "--size_z",
                str(pocket_definition["size"][2]),
                "--exhaustiveness",
                str(exhaustiveness),
                "--seed",
                "1",
                "--num_modes",
                str(n_poses),
                "--cnn_scoring",
                "rescore",
                "--cnn",
                "crossdock_default2018",
            ]

            result = subprocess.run(gnina_cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                printlog(f"GNINA GPU docking failed for batch {batch_file.stem}")
                printlog(f"Error output: {result.stderr}")
                return None

            if not raw_results_path.exists() or raw_results_path.stat().st_size == 0:
                printlog(f"GNINA GPU output file is missing or empty for batch {batch_file.stem}")
                return None

            # Process results
            df = PandasTools.LoadSDF(
                str(raw_results_path), molColName="Molecule", idName="ID"
            )
            
            if df.empty:
                return None

            # Find CNN score column
            cnn_score_columns = ["CNNscore", "CNN_score", "cnn_score"]
            found_column = next(
                (col for col in cnn_score_columns if col in df.columns), None
            )
            if found_column is None:
                printlog(f"No CNN score column found for batch {batch_file.stem}")
                return None

            # Process and format results
            df["CNN-Score"] = pd.to_numeric(df[found_column], errors="coerce")
            df = df.dropna(subset=["CNN-Score"])
            if df.empty:
                return None

            # Sort and rank poses
            df = df.sort_values(["ID", "CNN-Score"], ascending=[True, False])
            df["Pose Rank"] = df.groupby("ID").cumcount() + 1
            df = df[df["Pose Rank"] <= n_poses]
            df["Pose ID"] = df.apply(
                lambda row: f"{row['ID']}_GNINA_{row['Pose Rank']}", axis=1
            )

            # Save processed results
            final_df = df[["Pose ID", "ID", "Molecule", "CNN-Score"]]
            PandasTools.WriteSDF(
                final_df,
                str(processed_results_path),
                molColName="Molecule",
                idName="Pose ID",
                properties=list(final_df.columns),
            )

            raw_results_path.unlink()
            return processed_results_path

        except Exception as e:
            printlog(
                f"ERROR: Failed to process GNINA GPU result for batch {batch_file.stem}: {str(e)}"
            )
            return None
