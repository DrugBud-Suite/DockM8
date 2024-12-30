"""Implementation of the PLANTS docking function."""

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
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.path_check import get_executable_path


class PlantsDocking(DockingFunction):
    """A class to perform docking using the PLANTS software."""

    def __init__(self, software_path: Path):
        """Initializes the PlantsDocking class.

        Args:
            software_path (Path): The base path to the PLANTS software installation.
        """
        super().__init__("PLANTS", software_path)
        self.software_path = software_path
        self.executable_path = get_executable_path(software_path, "PLANTS")

    def generate_plants_config(
        self,
        protein_mol2: Path,
        ligands_mol2: Path,
        pocket_definition: dict,
        n_poses: int,
        output_dir: Path,
        config_path: Path,
    ):
        """Generates PLANTS configuration file with docking parameters.

        Args:
            protein_mol2 (Path): Path to the protein MOL2 file.
            ligands_mol2 (Path): Path to the ligands MOL2 file.
            pocket_definition (dict): Dictionary defining the docking pocket.
            n_poses (int): Number of poses to generate.
            output_dir (Path): Directory to store the PLANTS output.
            config_path (Path): Path to save the PLANTS configuration file.
        """
        config_lines = [
            "# search algorithm\n",
            "search_speed speed1\n",
            "aco_ants 20\n",
            "flip_amide_bonds 0\n",
            "flip_planar_n 1\n",
            "force_flipped_bonds_planarity 0\n",
            "force_planar_bond_rotation 1\n",
            "rescore_mode simplex\n",
            "flip_ring_corners 0\n",
            "# scoring functions\n",
            "# Intermolecular (protein-ligand interaction scoring)\n",
            "scoring_function chemplp\n",
            "outside_binding_site_penalty 50.0\n",
            "enable_sulphur_acceptors 1\n",
            "# Intramolecular ligand scoring\n",
            "ligand_intra_score clash2\n",
            "chemplp_clash_include_14 1\n",
            "chemplp_clash_include_HH 0\n",
            "# input\n",
            f"protein_file {protein_mol2}\n",
            f"ligand_file {ligands_mol2}\n",
            "# output\n",
            f"output_dir {output_dir}\n",
            "# write single mol2 files\n",
            "write_multi_mol2 1\n",
            "# binding site definition\n",
            f'bindingsite_center {pocket_definition["center"][0]} {pocket_definition["center"][1]} {pocket_definition["center"][2]}\n',
            f'bindingsite_radius {pocket_definition["size"][0] / 2}\n',
            "# cluster algorithm\n",
            f"cluster_structures {n_poses}\n",
            "cluster_rmsd 2.0\n",
            "# write\n",
            "write_ranking_links 0\n",
            "write_protein_bindingsite 0\n",
            "write_protein_conformations 0\n",
            "write_protein_splitted 0\n",
            "write_merged_protein 0\n",
            "####\n",
        ]
        with open(config_path, "w") as config_writer:
            config_writer.writelines(config_lines)

    def dock_batch(
        self,
        batch_file: Path,
        protein_file: Path,
        pocket_definition: dict[str, list],
        exhaustiveness: int,
        n_poses: int,
    ) -> Path | None:
        RDLogger.DisableLog("rdApp.*")

        # Create only the essential directories
        batch_raw_dir = self._temp_dir / "raw" / f"{batch_file.stem}"
        processed_results_path = (
            self._temp_dir / "processed" / f"{batch_file.stem}_processed.sdf"
        )
        
        try:
            # Create batch directory for intermediate files
            batch_raw_dir.mkdir(parents=True, exist_ok=True)

            # Convert protein to mol2
            protein_mol2 = batch_raw_dir / "protein.mol2"
            try:
                convert_molecules(protein_file, protein_mol2, "pdb", "mol2")
            except Exception as e:
                printlog(f"ERROR: Failed to convert protein file to mol2: {str(e)}")
                return None

            # Convert ligands to mol2
            ligands_mol2 = batch_raw_dir / f"{batch_file.stem}.mol2"
            try:
                convert_molecules(batch_file, ligands_mol2, "sdf", "mol2")
            except Exception as e:
                printlog(f"ERROR: Failed to convert ligands file to mol2: {str(e)}")
                return None

            # Define results directory but don't create it
            results_dir = batch_raw_dir / "results"
            config_path = batch_raw_dir / "plants_config.txt"
            
            # Generate PLANTS config file
            self.generate_plants_config(
                protein_mol2=protein_mol2,
                ligands_mol2=ligands_mol2,
                pocket_definition=pocket_definition,
                n_poses=n_poses,
                output_dir=results_dir,
                config_path=config_path,
            )

            # Run PLANTS docking - let PLANTS create its own directories
            plants_cmd = [
                str(self.executable_path),
                "--mode",
                "screen",
                str(config_path),
            ]
            process = subprocess.Popen(
                plants_cmd,
                cwd=str(batch_raw_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                printlog(f"PLANTS docking failed for batch {batch_file.stem}")
                printlog(f"Error output:\n{stderr.decode()}")
                return None

            # Check for results
            mol2_results = results_dir / "docked_ligands.mol2"
            rankings_file = results_dir / "ranking.csv"
            if not (mol2_results.exists() and rankings_file.exists()):
                printlog(f"PLANTS output files missing for batch {batch_file.stem}")
                return None

            # Convert results to SDF
            sdf_results = results_dir / "docked_ligands.sdf"
            try:
                convert_molecules(mol2_results, sdf_results, "mol2", "sdf")
            except Exception as e:
                printlog(f"ERROR: Failed to convert PLANTS poses to SDF: {str(e)}")
                return None

            # Process results
            poses_df = PandasTools.LoadSDF(
                str(sdf_results), molColName="Molecule", idName="ID"
            )
            if poses_df.empty:
                return None

            # Load and process rankings
            rankings_df = pd.read_csv(rankings_file)
            rankings_df = rankings_df.rename(
                columns={"LIGAND_ENTRY": "ID", "TOTAL_SCORE": "CHEMPLP"}
            )[["ID", "CHEMPLP"]]

            # Merge poses with scores
            df = pd.merge(rankings_df, poses_df, on="ID")

            # Clean up IDs and rank poses
            df["ID"] = df["ID"].str.split("_").str[0]
            df = df.sort_values(["ID", "CHEMPLP"])
            df["Pose Rank"] = df.groupby("ID").cumcount() + 1
            df = df[df["Pose Rank"] <= n_poses]
            df["Pose ID"] = df.apply(
                lambda row: f"{row['ID']}_PLANTS_{row['Pose Rank']}", axis=1
            )

            # Save processed results
            final_df = df[["Pose ID", "ID", "Molecule", "CHEMPLP"]]
            PandasTools.WriteSDF(
                final_df,
                str(processed_results_path),
                molColName="Molecule",
                idName="Pose ID",
                properties=list(final_df.columns),
            )

            return processed_results_path

        except Exception as e:
            printlog(f"ERROR in PLANTS docking process: {str(e)}")
            return None
