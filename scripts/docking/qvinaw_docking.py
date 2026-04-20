"""Implementation of the QVINAW docking function."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
from meeko import PDBQTMolecule, RDKitMolCreate
from rdkit import RDLogger
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.docking.docking_function import DockingFunction
from scripts.utilities.file_splitting import split_pdbqt_str
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.path_check import get_executable_path


class QvinawDocking(DockingFunction):
    """A class to perform docking using the QVINAW software."""

    def __init__(self, software_path: Path):
        """Initializes the QvinawDocking class.

        Args:
            software_path (Path): The base path to the QVINAW software installation.
        """
        super().__init__("QVINAW", software_path)
        self.software_path = software_path
        self.executable_path = get_executable_path(software_path, "qvina-w")

    def dock_batch(
        self,
        batch_file: Path,
        protein_file: Path,
        pocket_definition: dict[str, list],
        exhaustiveness: int,
        n_poses: int,
    ) -> Path | None:
        """Performs docking for a single batch of ligands using QVINAW.

        Args:
            batch_file (Path): Path to the SDF file containing the batch of ligands.
            protein_file (Path): Path to the receptor file.
            pocket_definition (dict[str, list]): Dictionary defining the docking pocket.
            exhaustiveness (int): Exhaustiveness parameter for docking.
            n_poses (int): Number of poses to generate per ligand.

        Returns:
            Path | None: Path to the processed SDF file containing docking results,
                         or None if the batch failed.
        """
        RDLogger.DisableLog("rdApp.*")

        # Use consistent file naming based on batch file stem
        raw_results_path = self._temp_dir / "raw" / f"{batch_file.stem}_raw"
        raw_results_path.mkdir(parents=True, exist_ok=True)
        processed_results_path = (
            self._temp_dir / "processed" / f"{batch_file.stem}_processed.sdf"
        )

        try:
            # Convert ligand SDF to PDBQT
            ligand_pdbqt_dir = self._temp_dir / "ligand_pdbqt" / batch_file.stem
            ligand_pdbqt_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                pdbqt_files = convert_molecules(batch_file, ligand_pdbqt_dir, "sdf", "pdbqt")
                if isinstance(pdbqt_files, Path):
                    pdbqt_files = [pdbqt_files]
                pdbqt_files = [f for f in pdbqt_files if f.exists() and f.stat().st_size > 0]
                if not pdbqt_files:
                    printlog(f"No valid PDBQT files produced for batch {batch_file.stem}")
                    return None
            except Exception as e:
                error_msg = f"Failed to convert ligand SDF to PDBQT for batch {batch_file.stem}: {e}"
                printlog(error_msg)
                raise RuntimeError(error_msg)

            # Convert protein to PDBQT (skip if already converted by another worker)
            protein_pdbqt = self._temp_dir / "protein_pdbqt" / f"{protein_file.stem}.pdbqt"
            protein_pdbqt.parent.mkdir(parents=True, exist_ok=True)

            if not protein_pdbqt.exists():
                try:
                    convert_molecules(protein_file, protein_pdbqt, "pdb", "pdbqt")
                except Exception as e:
                    error_msg = f"Failed to convert protein file to PDBQT: {e}"
                    printlog(error_msg)
                    raise RuntimeError(error_msg)

            if not protein_pdbqt.exists() or protein_pdbqt.stat().st_size == 0:
                raise RuntimeError(f"Protein PDBQT file not found after conversion: {protein_pdbqt}")

            # Run docking for each ligand PDBQT file
            docked_files = []
            for ligand_file in pdbqt_files:
                output_file = raw_results_path / f"{ligand_file.stem}_docked.pdbqt"
                
                qvinaw_cmd = [
                    str(self.executable_path),
                    "--receptor", str(protein_pdbqt),
                    "--ligand", str(ligand_file),
                    "--out", str(output_file),
                    "--center_x", str(pocket_definition["center"][0]),
                    "--center_y", str(pocket_definition["center"][1]),
                    "--center_z", str(pocket_definition["center"][2]),
                    "--size_x", str(pocket_definition["size"][0]),
                    "--size_y", str(pocket_definition["size"][1]),
                    "--size_z", str(pocket_definition["size"][2]),
                    "--exhaustiveness", str(exhaustiveness),
                    "--cpu", "1",
                    "--seed", "1",
                    "--energy_range", "10",
                    "--num_modes", str(n_poses),
                ]
                
                result = subprocess.run(
                    qvinaw_cmd, 
                    capture_output=True, 
                    text=True, 
                    check=False
                )
                
                if result.returncode != 0:
                    printlog(f"QVINAW docking failed for ligand {ligand_file.stem}")
                    printlog(f"Error output: {result.stderr}")
                    continue
                
                if not output_file.exists() or output_file.stat().st_size == 0:
                    printlog(f"QVINAW output file is missing or empty for ligand {ligand_file.stem}")
                    continue
                    
                docked_files.append(output_file)

            if not docked_files:
                printlog(f"No successful docking results for batch {batch_file.stem}")
                return None

            # Split multi-model PDBQT files and collect all poses
            split_poses = []
            for docked_file in docked_files:
                try:
                    pose_files = split_pdbqt_str(docked_file)
                    split_poses.extend(pose_files)
                except Exception as e:
                    printlog(f"Failed to split PDBQT file {docked_file.name}: {e}")

            if not split_poses:
                printlog(f"No poses generated after splitting for batch {batch_file.stem}")
                return None

            # Process all pose files and create a dataframe
            pose_data = []
            for pose_file in split_poses:
                try:
                    # Filename format: {compound_name}_docked_{model_number}
                    parts = pose_file.stem.rsplit('_', 2)
                    compound_id = parts[0] if len(parts) >= 3 else pose_file.stem.split('_')[0]
                    model_number = int(parts[-1]) if len(parts) >= 2 else 1
                    
                    # Read affinity from PDBQT file
                    affinity = None
                    with open(pose_file) as f:
                        for line in f:
                            if "REMARK VINA RESULT:" in line:
                                affinity = float(line.split()[3])
                                break
                    
                    if affinity is None:
                        printlog(f"Could not find affinity in {pose_file}")
                        continue
                    
                    # Convert PDBQT to RDKit molecule
                    pdbqt_mol = PDBQTMolecule.from_file(str(pose_file), skip_typing=True)
                    rdkit_mol = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)[0]
                    
                    pose_data.append({
                        "ID": compound_id,
                        "Molecule": rdkit_mol,
                        "QVINAW_Affinity": affinity,
                        "Model": model_number
                    })
                except Exception as e:
                    printlog(f"Error processing pose file {pose_file}: {e}")

            if not pose_data:
                printlog(f"Failed to process any poses for batch {batch_file.stem}")
                return None
                
            # Create DataFrame and process results
            df = pd.DataFrame(pose_data)
            
            # Sort by ID and affinity, then rank poses
            df = df.sort_values(["ID", "QVINAW_Affinity"])
            df["Pose Rank"] = df.groupby("ID").cumcount() + 1
            
            # Keep only requested number of poses
            df = df[df["Pose Rank"] <= n_poses]
            
            # Create pose IDs
            df["Pose ID"] = df.apply(
                lambda row: f"{row['ID']}_QVINAW_{row['Pose Rank']}", axis=1
            )
            
            # Save processed results to SDF
            final_df = df[["Pose ID", "ID", "Molecule", "QVINAW_Affinity"]]
            PandasTools.WriteSDF(
                final_df,
                str(processed_results_path),
                molColName="Molecule",
                idName="Pose ID",
                properties=list(final_df.columns),
            )
            
            # Clean up raw files if successful
            return processed_results_path

        except Exception as e:
            printlog(
                f"ERROR: Failed to process QVINAW result for batch {batch_file.stem}: {str(e)}"
            )
            return None