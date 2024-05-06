import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from rdkit.Chem import PandasTools
from tqdm import tqdm

# Search for 'DockM8' in parent directories
dockm8_path = next((p / 'DockM8' for p in Path(__file__).resolve().parents if (p / 'DockM8').is_dir()), None)
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import (
    delete_files,
    printlog,
)


def smina_docking(split_file: Path,
    w_dir: Path,
    protein_file: str,
    pocket_definition: Dict[str, list],
    software: Path,
    exhaustiveness: int,
    n_poses: int,
):
    """
    Dock ligands from a splitted file into a protein using smina.

    Args:
        split_file (str): Path to the splitted file containing the ligands to dock.
        w_dir (Path): Path to the working directory.
        protein_file (str): Path to the protein file.
        pocket_definition (Dict[str, list]): Dictionary containing the center and size of the pocket to dock into.
        software (Path): Path to the smina software.
        exhaustiveness (int): Exhaustiveness parameter for smina.
        n_poses (int): Number of poses to generate.

    Returns:
        None
    """
    # Create a folder to store the smina results
    smina_folder = w_dir / "smina"
    smina_folder.mkdir(parents=True, exist_ok=True)
    if split_file:
        input_file = split_file
        results_path = (
            smina_folder / f"{os.path.basename(split_file).split('.')[0]}_smina.sdf"
        )
    else:
        input_file = w_dir / "final_library.sdf"
        results_path = smina_folder / "docked.sdf"
    # Construct the smina command
    smina_cmd = (
        f'{software / "gnina"}'
        + f" --receptor {protein_file}"
        + f" --ligand {input_file}"
        + f" --out {results_path}"
        + f' --center_x {pocket_definition["center"][0]}'
        + f' --center_y {pocket_definition["center"][1]}'
        + f' --center_z {pocket_definition["center"][2]}'
        + f' --size_x {pocket_definition["size"][0]}'
        + f' --size_y {pocket_definition["size"][1]}'
        + f' --size_z {pocket_definition["size"][2]}'
        + f" --exhaustiveness {exhaustiveness}"
        + " --cpu 1"
        + f" --num_modes {n_poses}"
        + " --cnn_scoring none --no_gpu"
    )
    try:
        # Execute the smina command
        subprocess.call(smina_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except Exception as e:
        printlog(f"SMINA docking failed: {e}")
    return


def fetch_smina_poses(w_dir: Union[str, Path], n_poses: int, *args):
    """
    Fetches SMINA poses from the specified directory and combines them into a single SDF file.

    Args:
        w_dir (str or Path): The directory path where the SMINA poses are located.
        n_poses (int): The number of poses to fetch.

    Returns:
        None
    """
    if (w_dir / "smina").is_dir() and not (
        w_dir / "smina" / "smina_poses.sdf"
    ).is_file():
        try:
            smina_dataframes = []
            for file in tqdm(os.listdir(w_dir / "smina"), desc="Loading SMINA poses"):
                if file.startswith("split"):
                    df = PandasTools.LoadSDF(
                        str(w_dir / "smina" / file),
                        idName="ID",
                        molColName="Molecule",
                        includeFingerprints=False,
                        embedProps=False,
                        removeHs=False,
                        strictParsing=True,
                    )
                    smina_dataframes.append(df)
            smina_df = pd.concat(smina_dataframes)
            list_ = [*range(1, int(n_poses) + 1, 1)]
            ser = list_ * (len(smina_df) // len(list_))
            smina_df["Pose ID"] = [
                f'{row["ID"]}_SMINA_{num}'
                for num, (_, row) in zip(
                    ser + list_[: len(smina_df) - len(ser)], smina_df.iterrows()
                )
            ]
            smina_df.rename(
                columns={"minimizedAffinity": "SMINA_Affinity"}, inplace=True
            )
        except Exception as e:
            printlog("ERROR: Failed to Load SMINA poses SDF file!")
            printlog(e)
        try:
            PandasTools.WriteSDF(
                smina_df,
                str(w_dir / "smina" / "smina_poses.sdf"),
                molColName="Molecule",
                idName="Pose ID",
                properties=list(smina_df.columns),
            )
        except Exception as e:
            printlog("ERROR: Failed to write combined SMINA poses SDF file!")
            printlog(e)
        else:
            delete_files(w_dir / "smina", "smina_poses.sdf")
