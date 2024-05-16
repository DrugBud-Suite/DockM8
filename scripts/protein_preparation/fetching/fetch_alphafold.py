from pathlib import Path
import requests
import sys

# Search for 'DockM8' in parent directories
scripts_path = next((p / 'scripts'
                     for p in Path(__file__).resolve().parents
                     if (p / 'scripts').is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import printlog


def fetch_alphafold_structure(uniprot_code: str, output_dir: Path):
    """
    Fetches the Alphafold structure prediction for a given UniProt code and saves the corresponding PDB file.

    Args:
        uniprot_code (str): The UniProt code for the protein of interest.
        output_dir (Path): The directory where the PDB file will be saved.

    Returns:
        Path: The file path of the downloaded PDB file, or None if the download failed.
    """
    url = f'https://alphafold.ebi.ac.uk/api/prediction/{uniprot_code}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()  # Parse the JSON response
        if data:  # Check if the data is not empty
            pdb_url = data[0][
                'pdbUrl']  # Extract the pdbUrl from the first item in the list
            pdb_response = requests.get(pdb_url)
            if pdb_response.status_code == 200:
                output_file_path = output_dir / f"{uniprot_code}.pdb"
                with open(output_file_path, 'wb') as file:
                    file.write(pdb_response.content)
                printlog(
                    f"AlphaFold structure downloaded and saved to: {output_file_path}"
                )
                return output_file_path
            else:
                printlog("Failed to download the AlphaFold structure.")
        else:
            printlog(f"No data available for UniProt code: {uniprot_code}")
    else:
        printlog(
            f"Error: Failed to fetch Alphafold structure for UniProt code: {uniprot_code}"
        )
