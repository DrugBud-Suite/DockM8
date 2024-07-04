import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from rdkit.Chem import PandasTools
from tqdm import tqdm

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import delete_files
from scripts.utilities.logging import printlog


def gnina_docking(split_file: Path,
	w_dir: Path,
	protein_file: str,
	pocket_definition: Dict[str, list],
	software: Path,
	exhaustiveness: int,
	n_poses: int,
	):
	"""
    Dock ligands from a splitted file into a protein using gnina.

    Args:
        split_file (str): Path to the splitted file containing the ligands to dock.
        w_dir (Path): Path to the working directory.
        protein_file (str): Path to the protein file.
        pocket_definition (Dict[str, list]): Dictionary containing the center and size of the pocket to dock into.
        software (Path): Path to the gnina software.
        exhaustiveness (int): Exhaustiveness parameter for gnina.
        n_poses (int): Number of poses to generate.

    Returns:
        None
    """
	# Create a folder to store the gnina results
	gnina_folder = w_dir / "gnina"
	gnina_folder.mkdir(parents=True, exist_ok=True)
	if split_file:
		input_file = split_file
		results_path = gnina_folder / f"{os.path.basename(split_file).split('.')[0]}_gnina.sdf"
	else:
		input_file = w_dir / "final_library.sdf"
		results_path = gnina_folder / "docked.sdf"
	log = gnina_folder / f"{os.path.basename(split_file).split('.')[0]}_gnina.log"
	# Construct the gnina command
	gnina_cmd = (f"{software / 'gnina'}"
		f" --receptor {protein_file}"
		f" --ligand {input_file}"
		f" --out {results_path}"
		f" --center_x {pocket_definition['center'][0]}"
		f" --center_y {pocket_definition['center'][1]}"
		f" --center_z {pocket_definition['center'][2]}"
		f" --size_x {pocket_definition['size'][0]}"
		f" --size_y {pocket_definition['size'][1]}"
		f" --size_z {pocket_definition['size'][2]}"
		f" --exhaustiveness {exhaustiveness}"
		f" --log {log}"
		" --cpu 1 --seed 1"
		f" --num_modes {n_poses}"
		" --cnn_scoring rescore --cnn crossdock_default2018 --no_gpu")
	try:
		# Execute the gnina command
		subprocess.call(gnina_cmd, shell=True          #, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
			)
	except Exception as e:
		printlog(f"GNINA docking failed: {e}")
	return


def fetch_gnina_poses(w_dir: Union[str, Path], n_poses: int, *args):
	"""
    Fetches GNINA poses from the specified directory and combines them into a single SDF file.

    Args:
        w_dir (str or Path): The directory path where the GNINA poses are located.
        n_poses (int): The number of poses to fetch.

    Returns:
        None
    """
	if (w_dir / "gnina").is_dir() and not (w_dir / "gnina" / "gnina_poses.sdf").is_file():
		try:
			gnina_dataframes = []
			for file in tqdm(os.listdir(w_dir / "gnina"), desc="Loading GNINA poses"):
				if file.startswith("split"):
					df = PandasTools.LoadSDF(str(w_dir / "gnina" / file),
							idName="ID",
							molColName="Molecule",
							strictParsing=False)
					gnina_dataframes.append(df)
			gnina_df = pd.concat(gnina_dataframes)
			list_ = [*range(1, int(n_poses) + 1, 1)]
			ser = list_ * (len(gnina_df) // len(list_))
			gnina_df["Pose ID"] = [
				f'{row["ID"]}_GNINA_{num}'
				for num, (_, row) in zip(ser + list_[:len(gnina_df) - len(ser)], gnina_df.iterrows())]
			gnina_df.rename(columns={"minimizedAffinity": "GNINA_Affinity"}, inplace=True)
		except Exception as e:
			printlog("ERROR: Failed to Load GNINA poses SDF file!")
			printlog(e)
		try:
			PandasTools.WriteSDF(gnina_df,
					str(w_dir / "gnina" / "gnina_poses.sdf"),
					molColName="Molecule",
					idName="Pose ID",
					properties=list(gnina_df.columns))

		except Exception as e:
			printlog("ERROR: Failed to write combined GNINA poses SDF file!")
			printlog(e)
		else:
			delete_files(w_dir / "gnina", ["gnina_poses.sdf", "*.log"])
	return w_dir / "gnina" / "gnina_poses.sdf"
