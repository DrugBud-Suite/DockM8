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


def qvinaw_docking(
    split_file: Path,
    w_dir: Path,
    protein_file: Path,
    pocket_definition: Dict[str, list],
    software: Path,
    exhaustiveness: int,
    n_poses: int,
):
    """
    Perform docking using the QVINAW software for either a library of molecules or split files.

    Args:
        w_dir (Path): Working directory for docking operations.
        protein_file (Path): Path to the protein file in PDB or pdbqt format.
        pocket_definition (dict): Dictionary containing the center and size of the docking pocket.
        software (Path): Path to the QVINAW software folder.
        exhaustiveness (int): Level of exhaustiveness for the docking search.
        n_poses (int): Number of poses to generate for each ligand.
        split_file (Path, optional): Path to the split file containing the ligands to dock. If None, uses library.

    Returns:
        Path: Path to the combined docking results file in .sdf format.
    """
    qvinaw_folder = w_dir / "qvinaw"
    results_folder = qvinaw_folder / Path(split_file).stem / "docked"
    if split_file:
        input_file = split_file
        pdbqt_folder = results_folder / Path(split_file).stem / "pdbqt_files"
    else:
        input_file = w_dir / "final_library.sdf"
        pdbqt_folder = qvinaw_folder / "pdbqt_files"

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

    # Dock each ligand using QVINAW
    for pdbqt_file in pdbqt_folder.glob("*.pdbqt"):
        output_file = results_folder / (pdbqt_file.stem + "_QVINAW.pdbqt")
        qvinaw_cmd = (
            f"{software / 'qvina-w'}"
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
            qvinaw_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )

    qvinaw_docking_results = qvinaw_folder / (Path(input_file).stem + "_qvinaw.sdf")
    # Process the docked poses
    try:
        for file in results_folder.glob("*.pdbqt"):
            split_pdbqt_str(file)
        qvinaw_poses = pd.DataFrame(columns=["Pose ID", "Molecule", "QVINAW_Affinity"])
        for pose_file in results_folder.glob("*.pdbqt"):
            pdbqt_mol = PDBQTMolecule.from_file(
                pose_file, name=pose_file.stem, skip_typing=True
            )
            rdkit_mol = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
            # Extracting QVINAW_Affinity from the file
            with open(pose_file) as file:
                affinity = next(
                    line.split()[3] for line in file if "REMARK VINA RESULT:" in line
                )
            # Use loc to append a new row to the DataFrame
            qvinaw_poses.loc[qvinaw_poses.shape[0]] = {
                "Pose ID": pose_file.stem,
                "Molecule": rdkit_mol[0],
                "QVINAW_Affinity": affinity,
                "ID": pose_file.stem.split("_")[0],
            }
        PandasTools.WriteSDF(
            qvinaw_poses,
            str(qvinaw_docking_results),
            molColName="Molecule",
            idName="Pose ID",
            properties=list(qvinaw_poses.columns),
        )
    except Exception as e:
        printlog("ERROR: Failed to combine QVINAW SDF file!")
        printlog(e)
    else:
        shutil.rmtree(qvinaw_folder / Path(input_file).stem, ignore_errors=True)
    return qvinaw_docking_results


def fetch_qvinaw_poses(w_dir: Union[str, Path], *args):
    """
    Fetches QVINAW poses from the specified directory and combines them into a single SDF file.

    Args:
        w_dir (str or Path): The directory path where the QVINAW poses are located.

    Returns:
        Path: The path to the combined QVINAW poses SDF file.

    Raises:
        Exception: If there is an error loading or writing the QVINAW poses SDF file.
    """
    if (w_dir / "qvinaw").is_dir() and not (
        w_dir / "qvinaw" / "qvinaw_poses.sdf"
    ).is_file():
        try:
            qvinaw_dataframes = []
            for file in tqdm(os.listdir(w_dir / "qvinaw"), desc="Loading QVINAW poses"):
                if (
                    file.startswith("split")
                    or file.startswith("final_library")
                    and file.endswith(".sdf")
                ):
                    df = PandasTools.LoadSDF(
                        str(w_dir / "qvinaw" / file),
                        idName="Pose ID",
                        molColName="Molecule",
                    )
                    qvinaw_dataframes.append(df)
            qvinaw_df = pd.concat(qvinaw_dataframes)
            qvinaw_df["ID"] = qvinaw_df["Pose ID"].apply(lambda x: x.split("_")[0])
        except Exception as e:
            printlog("ERROR: Failed to Load QVINAW poses SDF file!")
            printlog(e)
        try:
            PandasTools.WriteSDF(
                qvinaw_df,
                str(w_dir / "qvinaw" / "qvinaw_poses.sdf"),
                molColName="Molecule",
                idName="Pose ID",
                properties=list(qvinaw_df.columns),
            )
        except Exception as e:
            printlog("ERROR: Failed to write combined QVINAW poses SDF file!")
            printlog(e)
        else:
            delete_files(w_dir / "qvinaw", "qvinaw_poses.sdf")
    return w_dir / "qvinaw" / "qvinaw_poses.sdf"
