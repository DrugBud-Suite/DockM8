import os
import io
from pathlib import Path
import requests
import sys
import time
from urllib.parse import urljoin
import warnings
from scripts.utilities import *
from IPython.display import Image
from Bio.PDB import *
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.PDBIO import PDBIO
from rdkit import Chem

PROTEINS_PLUS_URL = 'https://proteins.plus/api/v2/'
UPLOAD = urljoin(PROTEINS_PLUS_URL, 'molecule_handler/upload/')
UPLOAD_JOBS = urljoin(PROTEINS_PLUS_URL, 'molecule_handler/upload/jobs/')
PROTEINS = urljoin(PROTEINS_PLUS_URL, 'molecule_handler/proteins/')
LIGANDS = urljoin(PROTEINS_PLUS_URL, 'molecule_handler/ligands/')
PROTOSS = urljoin(PROTEINS_PLUS_URL, 'protoss/')
PROTOSS_JOBS = urljoin(PROTEINS_PLUS_URL, 'protoss/jobs/')

def poll_job(job_id, poll_url, poll_interval=1, max_polls=10):
    """Poll the progress of a job
    
    Continuosly polls the server in regular intervals and updates the job information, especially the status.
    
    :param job_id: UUID of the job to poll
    :type job_id: str
    :param poll_url: URl to send the polling request to
    :type poll_url: str
    :param poll_interval: time interval between polls in seconds
    :type poll_interval: int
    :param max_polls: maximum number of times to poll before exiting
    :type max_polls: int
    :return: polled job
    :rtype: dict
    """
    job = requests.get(poll_url + job_id + '/').json()
    status = job['status']
    current_poll = 0
    while status == 'pending' or status == 'running':
        print(f'Job {job_id} is { status }')
        current_poll += 1
        if current_poll >= max_polls:
            print(f'Job {job_id} has not completed after {max_polls} polling requests' \
                  f' and {poll_interval * max_polls} seconds')
            return job
        time.sleep(poll_interval)
        job = requests.get(poll_url + job_id + '/').json()
        status = job['status']
    print(f'Job {job_id} completed with { status }')
    return job
def prepare_protein_protoss(receptor : Path) -> Path :
    """
    Prepares a protein using ProtoSS.

    Args:
    receptor (Path): Path to the protein file in PDB format.

    Returns:
    Path: Path to the prepared protein file in PDB format.
    """
    printlog('Preparing protein with ProtoSS ...')
    with open(receptor) as upload_file:
        query = {'protein_file': upload_file}
        job_submission = requests.post(PROTOSS, files=query).json()
    protoss_job = poll_job(job_submission['job_id'], PROTOSS_JOBS)
    protossed_protein = requests.get(PROTEINS + protoss_job['output_protein'] + '/').json()
    protein_file = io.StringIO(protossed_protein['file_string'])
    protein_structure = PDBParser().get_structure(protossed_protein['name'], protein_file)
    
    output_file = Path(str(receptor).replace('.pdb', '_protoss.pdb'))
    with output_file.open('w') as output_file_handle:
        pdbio = PDBIO()
        pdbio.set_structure(protein_structure)
        pdbio.save(output_file_handle)
    
    return output_file