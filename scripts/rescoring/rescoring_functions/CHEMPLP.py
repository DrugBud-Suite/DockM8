from pathlib import Path
import pandas as pd

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.path_check import get_executable_path
from scripts.utilities.subprocess_handler import run_subprocess_command

class CHEMPLP(ScoringFunction):
    """CHEMPLP scoring function implementation."""

    def __init__(self, software_path: Path):
        super().__init__(
            name="CHEMPLP",
            column_name="CHEMPLP",
            best_value="min",
            score_range=(200, -200),
            software_path=software_path,
        )
        self.executable_path = get_executable_path(software_path, "PLANTS")

    def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
        """
        Rescore the molecules in the given SDF file using the CHEMPLP scoring function.

        Args:
            sdf_file (str): The path to the SDF file.
            n_cpus (int): The number of CPUs to use for parallel processing.
            protein_file (str): The path to the protein file.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: A DataFrame containing the rescored molecules.
        """
        try:
            printlog("Running CHEMPLP...")

            plants_protein_mol2 = self._temp_dir / "protein.mol2"
            plants_ligands_mol2 = self._temp_dir / "ligands.mol2"

            # Convert input files to MOL2 format
            for source, target, from_format in [
                (protein_file, plants_protein_mol2, "pdb"),
                (sdf_file, plants_ligands_mol2, "sdf")
            ]:
                try:
                    convert_molecules(source, target, from_format, "mol2")
                except Exception as e:
                    printlog(f"Error converting {source} to MOL2 format: {str(e)}")
                    return pd.DataFrame()

            pocket_definition = kwargs.get("pocket_definition")
            if not pocket_definition:
                printlog("Error: Pocket definition is required for CHEMPLP rescoring")
                return pd.DataFrame()

            config_file = self._create_config_file(
                self._temp_dir, plants_protein_mol2, plants_ligands_mol2, pocket_definition
            )

            results_dir = self._temp_dir / "results"
            results_csv = results_dir / "ranking.csv"

            chemplp_cmd = f"{self.executable_path} --mode rescore {config_file}"

            stdout, stderr = run_subprocess_command(command=chemplp_cmd)
        
            if not results_csv.exists():
                printlog(f"CHEMPLP output file not found: {results_csv}")
                if stderr:
                    printlog(f"CHEMPLP command output:\n{stdout}")
                    printlog(f"CHEMPLP command error output:\n{stderr}")
                return pd.DataFrame()

            try:
                chemplp_results = pd.read_csv(results_csv, sep=",", header=0)
                chemplp_results.rename(columns={"TOTAL_SCORE": self.column_name}, inplace=True)
                chemplp_results["Pose ID"] = chemplp_results["LIGAND_ENTRY"].apply(
                    lambda x: "_".join(x.split("_")[:3])
                )
                return chemplp_results[["Pose ID", self.column_name]]
            except Exception as e:
                printlog(f"Error processing CHEMPLP results: {str(e)}")
                return pd.DataFrame()

        except Exception as e:
            printlog("ERROR: An unexpected error occurred during CHEMPLP rescoring:")
            printlog(str(e))
            return pd.DataFrame()
        finally:
            self.cleanup()

    def _create_config_file(
        self, temp_dir: Path, protein_file: Path, ligands_file: Path, pocket_definition: dict
    ) -> Path:
        """
        Create a configuration file for PLANTS.

        Args:
            temp_dir (Path): The temporary directory path.
            protein_file (Path): The path to the protein file.
            ligands_file (Path): The path to the ligands file.
            pocket_definition (dict): Dictionary containing binding site parameters.

        Returns:
            Path: The path to the created configuration file.
        """
        results_dir = temp_dir / "results"
        temp_dir.mkdir(exist_ok=True)

        config_content = [
            "# search algorithm\n",
            "search_speed speed1\n",
            "aco_ants 20\n",
            "flip_amide_bonds 0\n",
            "flip_planar_n 1\n",
            "force_flipped_bonds_planarity 0\n",
            "force_planar_bond_rotation 1\n",
            "rescore_mode s\n",
            "flip_ring_corners 0\n",
            "# scoring functions\n",
            "scoring_function chemplp\n",
            "outside_binding_site_penalty 50.0\n",
            "enable_sulphur_acceptors 1\n",
            "ligand_intra_score clash2\n",
            "chemplp_clash_include_14 1\n",
            "chemplp_clash_include_HH 0\n",
            f"protein_file {protein_file}\n",
            f"ligand_file {ligands_file}\n",
            f"output_dir {results_dir}\n",
            "write_multi_mol2 1\n",
            f'bindingsite_center {pocket_definition["center"][0]} {pocket_definition["center"][1]} {pocket_definition["center"][2]}\n',
            f'bindingsite_radius {pocket_definition["size"][0] / 2}\n',
            "cluster_structures 10\n",
            "cluster_rmsd 2.0\n",
            "write_ranking_links 0\n",
            "write_protein_bindingsite 0\n",
            "write_protein_conformations 0\n",
            "write_protein_splitted 0\n",
            "write_merged_protein 0\n",
            "####\n",
        ]

        config_file = temp_dir / "config.txt"
        with config_file.open("w") as f:
            f.writelines(config_content)

        return config_file
