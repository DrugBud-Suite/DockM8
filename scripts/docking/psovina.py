import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from meeko import PDBQTMolecule, RDKitMolCreate
from rdkit.Chem import PandasTools
from tqdm import tqdm

# Search for 'DockM8' in parent directories
scripts_path = next((p / 'scripts' for p in Path(__file__).resolve().parents if (p / 'scripts').is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import convert_molecules, delete_files, printlog, split_pdbqt_str


def psovina_docking(
    split_file: Path,
    w_dir: Path,
    protein_file: Path,
    pocket_definition: Dict[str, list],
    software: Path,
    exhaustiveness: int,
    n_poses: int,
):
    """
    Perform docking using the PSOVINA software for either a library of molecules or split files.

    Args:
        w_dir (Path): Working directory for docking operations.
        protein_file (Path): Path to the protein file in PDB or pdbqt format.
        pocket_definition (dict): Dictionary containing the center and size of the docking pocket.
        software (Path): Path to the PSOVINA software folder.
        exhaustiveness (int): Level of exhaustiveness for the docking search.
        n_poses (int): Number of poses to generate for each ligand.
        split_file (Path, optional): Path to the split file containing the ligands to dock. If None, uses library.

    Returns:
        Path: Path to the combined docking results file in .sdf format.
    """
    psovina_folder = w_dir / "psovina"
    results_folder = psovina_folder / Path(split_file).stem / "docked"
    if split_file:
        input_file = split_file
        pdbqt_folder = results_folder / Path(split_file).stem / "pdbqt_files"
    else:
        input_file = w_dir / "final_library.sdf"
        pdbqt_folder = psovina_folder / "pdbqt_files"

    pdbqt_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    # Convert molecules to pdbqt format
    try:
        convert_molecules(input_file, str(pdbqt_folder), "sdf", "pdbqt")
    except Exception as e:
        print("Failed to convert sdf file to .pdbqt")
        print(e)

    protein_file_pdbqt = convert_molecules(
        str(protein_file),
        str(protein_file).replace(".pdb", ".pdbqt"),
        "pdb",
        "pdbqt",
    )

    # Dock each ligand using PSOVINA
    for pdbqt_file in pdbqt_folder.glob("*.pdbqt"):
        output_file = results_folder / (pdbqt_file.stem + "_PSOVINA.pdbqt")
        psovina_cmd = (
            f"{software / 'psovina'}"
            f" --receptor {protein_file_pdbqt}"
            f" --ligand {pdbqt_file}"
            f" --out {output_file}"
            f" --center_x {pocket_definition['center'][0]}"
            f" --center_y {pocket_definition['center'][1]}"
            f" --center_z {pocket_definition['center'][2]}"
            f" --size_x {pocket_definition['size'][0]}"
            f" --size_y {pocket_definition['size'][1]}"
            f" --size_z {pocket_definition['size'][2]}"
            f" --exhaustiveness {exhaustiveness}"
            " --cpu 1 --seed 1"
            f" --num_modes {n_poses}"
        )
        subprocess.call(
            psovina_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )

    psovina_docking_results = psovina_folder / (Path(input_file).stem + "_psovina.sdf")
    # Process the docked poses
    try:
        for file in results_folder.glob("*.pdbqt"):
            split_pdbqt_str(file)
        psovina_poses = pd.DataFrame(columns=["Pose ID", "Molecule", "PSOVINA_Affinity"])
        for pose_file in results_folder.glob("*.pdbqt"):
            pdbqt_mol = PDBQTMolecule.from_file(
                pose_file, name=pose_file.stem, skip_typing=True
            )
            rdkit_mol = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
            # Extracting PSOVINA_Affinity from the file
            with open(pose_file) as file:
                affinity = next(
                    line.split()[3] for line in file if "REMARK VINA RESULT:" in line
                )
            # Use loc to append a new row to the DataFrame
            psovina_poses.loc[psovina_poses.shape[0]] = {
                "Pose ID": pose_file.stem,
                "Molecule": rdkit_mol[0],
                "PSOVINA_Affinity": affinity,
                "ID": pose_file.stem.split("_")[0],
            }
        PandasTools.WriteSDF(
            psovina_poses,
            str(psovina_docking_results),
            molColName="Molecule",
            idName="Pose ID",
            properties=list(psovina_poses.columns),
        )
    except Exception as e:
        printlog("ERROR: Failed to combine PSOVINA SDF file!")
        printlog(e)
    else:
        shutil.rmtree(psovina_folder / Path(input_file).stem, ignore_errors=True)
    return psovina_docking_results


def fetch_psovina_poses(w_dir: Union[str, Path], *args):
    """
    Fetches PSOVINA poses from the specified directory and combines them into a single SDF file.

    Args:
        w_dir (str or Path): The directory path where the PSOVINA poses are located.

    Returns:
        Path: The path to the combined PSOVINA poses SDF file.

    Raises:
        Exception: If there is an error loading or writing the PSOVINA poses SDF file.
    """
    if (w_dir / "psovina").is_dir() and not (
        w_dir / "psovina" / "psovina_poses.sdf"
    ).is_file():
        try:
            psovina_dataframes = []
            for file in tqdm(os.listdir(w_dir / "psovina"), desc="Loading PSOVINA poses"):
                if file.endswith(".sdf"):
                    df = PandasTools.LoadSDF(
                        str(w_dir / "psovina" / file),
                        idName="Pose ID",
                        molColName="Molecule",
                    )
                    psovina_dataframes.append(df)
            psovina_df = pd.concat(psovina_dataframes)
            psovina_df["ID"] = psovina_df["Pose ID"].apply(lambda x: x.split("_")[0])
        except Exception as e:
            printlog("ERROR: Failed to Load PSOVINA poses SDF file!")
            printlog(e)
        try:
            PandasTools.WriteSDF(
                psovina_df,
                str(w_dir / "psovina" / "psovina_poses.sdf"),
                molColName="Molecule",
                idName="Pose ID",
                properties=list(psovina_df.columns),
            )
        except Exception as e:
            printlog("ERROR: Failed to write combined PSOVINA poses SDF file!")
            printlog(e)
        else:
            delete_files(w_dir / "psovina", "psovina_poses.sdf")
    return w_dir / "psovina" / "psovina_poses.sdf"
