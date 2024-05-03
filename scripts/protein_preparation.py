import io
import os
import sys
import time
import warnings
from pathlib import Path
from urllib.parse import urljoin

import requests
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO

cwd = Path.cwd()
dockm8_path = cwd.parents[0] / "DockM8"
sys.path.append(str(dockm8_path))

from scripts.utilities import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROTEINS_PLUS_URL = 'https://proteins.plus/api/v2/'
UPLOAD = urljoin(PROTEINS_PLUS_URL, 'molecule_handler/upload/')
UPLOAD_JOBS = urljoin(PROTEINS_PLUS_URL, 'molecule_handler/upload/jobs/')
PROTEINS = urljoin(PROTEINS_PLUS_URL, 'molecule_handler/proteins/')
LIGANDS = urljoin(PROTEINS_PLUS_URL, 'molecule_handler/ligands/')
PROTOSS = urljoin(PROTEINS_PLUS_URL, 'protoss/')
PROTOSS_JOBS = urljoin(PROTEINS_PLUS_URL, 'protoss/jobs/')

def poll_job(job_id, poll_url, poll_interval=1, max_polls=10):
    """
    Poll the progress of a job by continuously polling the server in regular intervals and updating the job information.

    Args:
    job_id (str): UUID of the job to poll.
    poll_url (str): URL to send the polling request to.
    poll_interval (int): Time interval between polls in seconds. Default is 1 second.
    max_polls (int): Maximum number of times to poll before exiting. Default is 10.

    Returns:
    dict: Polled job information.
    """
    # Get the initial job information
    job = requests.get(poll_url + job_id + '/').json()
    status = job['status']
    current_poll = 0

    # Continuously poll the job until it is completed or maximum polls reached
    while status == 'pending' or status == 'running':
        print(f'Job {job_id} is {status}')
        current_poll += 1

        # Check if maximum polls reached
        if current_poll >= max_polls:
            print(f'Job {job_id} has not completed after {max_polls} polling requests and {poll_interval * max_polls} seconds')
            return job

        # Wait for the specified interval before polling again
        time.sleep(poll_interval)

        # Poll the job again to get updated status
        job = requests.get(poll_url + job_id + '/').json()
        status = job['status']

    print(f'Job {job_id} completed with {status}')
    return job


def prepare_protein_protoss(receptor : Path) -> Path :
    """
    Prepares a protein using ProtoSS.

    Args:
    receptor (Path): Path to the protein file in PDB format.

    Returns:
    Path: Path to the prepared protein file in PDB format.
    """
    # Print log message
    printlog('Preparing protein with ProtoSS ...')

    # Open the receptor protein file
    with open(receptor) as upload_file:
        # Create the query with the protein file
        query = {'protein_file': upload_file}
        # Submit the job to ProtoSS and get the job submission response
        job_submission = requests.post(PROTOSS, files=query).json()

    # Poll the job status until it is completed
    protoss_job = poll_job(job_submission['job_id'], PROTOSS_JOBS)

    # Get the output protein information from the job
    protossed_protein = requests.get(PROTEINS + protoss_job['output_protein'] + '/').json()

    # Create a StringIO object with the protein file string
    protein_file = io.StringIO(protossed_protein['file_string'])

    # Parse the protein structure from the StringIO object
    protein_structure = PDBParser().get_structure(protossed_protein['name'], protein_file)
    
    # Create the output file path by replacing the extension of the receptor file
    output_file = Path(str(receptor).replace('.pdb', '_protoss.pdb'))

    # Open the output file in write mode
    with output_file.open('w') as output_file_handle:
        # Create a PDBIO object
        pdbio = PDBIO()
        # Set the protein structure for saving
        pdbio.set_structure(protein_structure)
        # Save the protein structure to the output file
        pdbio.save(output_file_handle)
    
    # Return the path to the prepared protein file
    return output_file

