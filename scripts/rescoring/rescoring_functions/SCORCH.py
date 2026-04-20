import subprocess
from pathlib import Path
import pandas as pd

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules

class SCORCH(ScoringFunction):

    def __init__(self, software_path: Path):
        super().__init__(
            name="SCORCH",
            column_name="SCORCH",
            best_value="max",
            score_range=(0, 1),
            software_path=software_path
        )

    def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
        try:
            scorch_protein = Path(self._temp_dir) / "protein.pdbqt"
            try:
                convert_molecules(Path(protein_file), scorch_protein, "pdb", "pdbqt")
            except Exception as e:
                printlog(f"Error converting protein file to PDBQT: {str(e)}")
                return pd.DataFrame()

            pdbqt_folder = Path(self._temp_dir) / "pdbqt_ligands"
            pdbqt_folder.mkdir(exist_ok=True)

            try:
                pdbqt_files = convert_molecules(Path(sdf_file), pdbqt_folder, "sdf", "pdbqt")
            except Exception as e:
                printlog(f"Error converting ligands to PDBQT: {str(e)}")
                return pd.DataFrame()

            if not pdbqt_files:
                printlog("No PDBQT files were created during conversion")
                return pd.DataFrame()

            printlog(f"Converted {len(pdbqt_files)} molecules to PDBQT for SCORCH scoring")

            results_csv = Path(self._temp_dir) / "scorch_results.csv"
            scorch_cmd = (
                f"cd {self.software_path}/SCORCH-1.0.0 &&"
                f" python scorch.py"
                f" --receptor {scorch_protein}"
                f" --ligand {pdbqt_folder}"
                f" --threads {n_cpus}"
                f" --out {results_csv}"
            )

            timeout = max(300, len(pdbqt_files) * 120)
            try:
                result = subprocess.run(
                    scorch_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                stdout, stderr = result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                printlog(f"SCORCH timed out after {timeout}s for {len(pdbqt_files)} molecules")
                return pd.DataFrame(columns=["Pose ID", self.column_name])

            if not results_csv.exists():
                printlog(f"SCORCH output file not found: {results_csv}")
                if stdout:
                    printlog(f"SCORCH stdout:\n{stdout}")
                if stderr:
                    printlog(f"SCORCH stderr:\n{stderr}")
                return pd.DataFrame(columns=["Pose ID", self.column_name])

            df = pd.read_csv(results_csv)
            if df.empty:
                printlog("SCORCH produced empty results")
                return pd.DataFrame(columns=["Pose ID", self.column_name])

            df.rename(
                columns={"SCORCH_score": self.column_name, "Ligand_ID": "Pose ID"},
                inplace=True
            )
            return df[["Pose ID", self.column_name]]

        except Exception as e:
            printlog(f"ERROR: Unexpected error during SCORCH rescoring: {str(e)}")
            return pd.DataFrame()
        finally:
            self.cleanup()
