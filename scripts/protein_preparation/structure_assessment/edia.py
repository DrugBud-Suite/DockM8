import sys
import warnings
from pathlib import Path

import pandas as pd
import requests

# Search for 'DockM8' in parent directories
scripts_path = next((p / 'scripts'
                     for p in Path(__file__).resolve().parents
                     if (p / 'scripts').is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.fetching.fetch_pdb import fetch_pdb_structure
from scripts.protein_preparation.utils import extract_chain
from scripts.utilities.utilities import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class EDIA_API_Consts:
    BASE_URL = "https://proteins.plus/api/edia_rest"
    HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

    @staticmethod
    def get_post_data(pdb_code):
        return {"edia": {"pdbCode": pdb_code}}


def submit_edia_job(pdb_code):
    """
    Submit a new job to the EDIA server and return the location URL of the results.

    Parameters
    ----------
    pdb_code : str
        PDB code of the protein structure.

    Returns
    -------
    str
        URL location of the job results.
    """
    try:
        printlog(f"Submitting EDIA job for PDB code {pdb_code}")
        url = EDIA_API_Consts.BASE_URL
        data = EDIA_API_Consts.get_post_data(pdb_code)
        response = requests.post(url,
                                 json=data,
                                 headers=EDIA_API_Consts.HEADERS)
        response.raise_for_status()
        response_data = response.json()
        return response_data["location"]
    except requests.exceptions.RequestException as e:
        printlog(f"Error submitting EDIA job: {e}")
        raise
    except (KeyError, ValueError) as e:
        printlog(f"Error parsing EDIA job response: {e}")
        raise


def get_edia_job_results(url, output_dir: Path):
    """
    Retrieves the EDIA job results from the given URL and saves the structure scores as a CSV file.

    Args:
        url (str): The URL of the EDIA job results.
        output_dir (Path): The directory where the CSV file will be saved.

    Returns:
        Path: The path to the saved CSV file.
    """
    try:
        printlog(f"Retrieving EDIA job results from {url}")
        response = requests.get(url)
        response.raise_for_status()

        structure_scores_url = response.json()["structure_scores"]

        structure_scores_response = requests.get(structure_scores_url)

        with open(output_dir / "EDIA_scores.csv", "wb") as f:
            f.write(structure_scores_response.content)

        return output_dir / "EDIA_scores.csv"
    except requests.exceptions.RequestException as e:
        printlog(f"Error retrieving EDIA job results: {e}")
        raise
    except (KeyError, ValueError) as e:
        printlog(f"Error parsing EDIA job response: {e}")
        raise


def get_best_chain_edia(pdb_code: str, output_dir: Path):
    """
    Retrieves the best chain from a PDB structure based on the EDIA scores.

    Args:
        pdb_code (str): The PDB code of the structure.
        output_dir (Path): The directory to save the output files.

    Returns:
        Path: The path to the extracted chain file.
    """
    try:
        printlog(f"Getting best chain for PDB code {pdb_code}")
        response = submit_edia_job(pdb_code)
        scores_csv = get_edia_job_results(response, output_dir)

        scores_df = pd.read_csv(scores_csv)
        average_scores = scores_df.groupby("Chain")["EDIAm"].mean()
        average_scores = average_scores.sort_values(ascending=False)
        for chain, score in average_scores.items():
            print(f"Chain {chain}: EDIA score = {score}")
        best_chain = average_scores.idxmax()
        # Delete the scores CSV file
        scores_csv.unlink()

        best_chain = average_scores.idxmax()
        printlog(
            f"Best EDIA scoring chain for PDB code {pdb_code}: {best_chain}")

        pdb_file = fetch_pdb_structure(pdb_code, output_dir)

        printlog(f"Extracting best chain {best_chain} from PDB file {pdb_file}")
        extracted_chain_path = extract_chain(pdb_file, best_chain)

        return extracted_chain_path
    except Exception as e:
        printlog(f"Error in determining best chain using EDIA: {e}")
        raise
