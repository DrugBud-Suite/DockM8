import sys
from pathlib import Path

import requests

# Search for 'DockM8' in parent directories
scripts_path = next((p / 'scripts' for p in Path(__file__).resolve().parents if (p / 'scripts').is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import printlog

def fetch_pdb_structure(pdb_id: str, output_dir: Path):
    """
    Fetches a PDB file from the RCSB PDB database based on the given PDB ID.

    Args:
        pdb_id (str): The PDB ID of the file to be fetched.
        output_dir (Path): The directory where the fetched PDB file will be saved.

    Returns:
        output_path (Path): The path to the downloaded PDB file.

    Raises:
        None
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)

    if response.status_code == 200:
        pdb_data = response.text
        output_path = output_dir / f"{pdb_id}.pdb"
        with open(output_path, "w") as file:
            file.write(pdb_data)
        printlog(f"PDB file {output_path} downloaded successfully.")
        return output_path
    else:
        raise Exception(f"Failed to fetch PDB file {pdb_id}.")
    