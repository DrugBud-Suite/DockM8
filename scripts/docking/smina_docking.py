"""Implementation of the SMINA docking function."""

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


class SminaDocking(DockingFunction):
    """A class to perform docking using the SMINA software."""

    def __init__(self, software_path: Path):
        """Initializes the SminaDocking class.

        Args:
            software_path (Path): The base path to the SMINA software installation.
        """
        super().__init__("SMINA", software_path)
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
            smina_cmd = [
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
                "--cpu",
                "1",
                "--seed",
                "1",
                "--num_modes",
                str(n_poses),
                "--cnn_scoring",
                "none",
            ]

            result = subprocess.run(
                smina_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if result.returncode != 0:
                printlog(f"SMINA docking failed for batch {batch_file.stem}")
                printlog(f"Error output: {result.stderr}")
                return None

            if not raw_results_path.exists() or raw_results_path.stat().st_size == 0:
                printlog(f"SMINA output file is missing or empty for batch {batch_file.stem}")
                print(result.stdout)
                print(result.stderr)
                return None

            # Process results
            df = PandasTools.LoadSDF(
                str(raw_results_path), molColName="Molecule", idName="ID"
            )
            
            if df.empty:
                printlog(f"No data loaded from SMINA output file for batch {batch_file.stem}")
                print(result.stdout)
                print(result.stderr)
                return None

            try:
                df["SMINA_Affinity"] = pd.to_numeric(
                    df["minimizedAffinity"], errors="raise"
                )
            except KeyError:
                printlog("Required column 'minimizedAffinity' not found in SMINA output")
                return None
            except ValueError:
                printlog("Invalid affinity values in SMINA output")
                return None

            # Sort and rank poses
            df = df.sort_values(["ID", "SMINA_Affinity"])
            df["Pose Rank"] = df.groupby("ID").cumcount() + 1
            df = df[df["Pose Rank"] <= n_poses]
            df["Pose ID"] = df.apply(
                lambda row: f"{row['ID']}_SMINA_{row['Pose Rank']}", axis=1
            )

            # Save processed results
            final_df = df[["Pose ID", "ID", "Molecule", "SMINA_Affinity"]]
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
                f"ERROR: Failed to process SMINA result for batch {batch_file.stem}: {str(e)}"
            )
            return None
