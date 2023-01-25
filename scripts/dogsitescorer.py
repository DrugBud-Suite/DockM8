"""
Class containing all the required functions and constants
to communicate with the DoGSiteScorer's Rest-API.
This can be used to submit binding site detection jobs,
either by providing the PDB code of a protein structure,
or by uploading its PDB file.
It returns a table of all detected pockets and sub-pockets
and their corresponding descriptors.
For each detected (sub-)pocket, a PDB file is provided
and a CCP4 map file is generated.
These can be downloaded and used to define the coordinates of
the (sub-)pocket needed for the docking calculation and visualization.
The function `select_best_pocket` is also defined which provides
several methods for selecting the most suitable binding site.
"""

import io  # for creating file-like objects from strings (needed as input for some functions)
import gzip  # for decompressing .gz files downloaded from DoGSiteScorer
import time  # for creating pauses during runtime (e.g. to wait for the response of API requests)
from pathlib import Path # for handling local paths
import re # for filtering floats from a list of strings

import requests  # for communicating with web-service APIs
import pandas as pd  # for creating dataframes and handling data
from biopandas.pdb import PandasPdb  # for working with PDB files
import redo # for retrying API queries if they fail


class APIConsts:
    """
    Constants for DoGSiteScorer's API.

    Notes
    -----
    API specifications described here:
    - https://proteins.plus/help/
    - https://proteins.plus/help/dogsite_rest
    """

    class FileUpload:
        URL = "https://proteins.plus/api/pdb_files_rest"
        REQUEST_MSG = "pdb_file[pathvar]"
        RESPONSE_MSG = {
            "status": "status_code",
            "status_codes": {"accepted": "accepted", "denied": "bad_request"},
            "message": "message",
            "url_of_id": "location",
        }
        RESPONSE_MSG_FETCH_ID = {"message": "message", "id": "id"}

    class SubmitJob:
        URL = "https://proteins.plus/api/dogsite_rest"
        QUERY_HEADERS = {
            "Content-type": "application/json",
            "Accept": "application/json",
        }

        RESPONSE_MSG = {"url_of_job": "location"}

        RESPONSE_MSG_FETCH_BINDING_SITES = {
            "result_table": "result_table",
            "pockets_pdb_files": "residues",
            "pockets_ccp4_files": "pockets",
        }

@redo.retriable(attempts=30, sleeptime=1, sleepscale=1.1, max_sleeptime=20)
def _send_request_get_results(
    request_type, 
    keys_list, 
    url, 
    task="Fetching results from DoGSiteScorer API", 
    **kwargs
):
    '''
    Send a request and get the keyword values from json response.
    
    Parameters
    ----------
    request_type : str
        Type of request, i.e. name of a function from the `requests` module,
        e.g. "get", "post".
    keys_list : list of strings
        List of keys in the json response to return.
    url : str
        URL to send the request to.
    task : str
        Textual description of the request's purpose to print in the error message if one is raised.
        Optional; default : "Fetching results from DoGSiteScorer API"
    **kwargs
        Additional arguments to send with the request.
    
    Returns
    -------
    list
        List of values in the json response corresponding to the input list of keys. 
    '''
    
    request_function = getattr(requests, request_type)
    response = request_function(url, **kwargs)
    response.raise_for_status()
    response_values = response.json()
    results=[]
    for key in keys_list:
        try:
            results.append(response_values[key])
        except KeyError:
            raise ValueError(
                f"{task} failed.\n"
                +f"Expected key {key} not found in the response.\n"
                +f"The response message is as follows: {response_values}"
            )
    return results
    
def upload_pdb_file(filepath):
    """
    Upload a PDB file to the DoGSiteScorer webserver using their API
    and get back a dummy PDB code, which can be used to submit a detection job.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Relative or absolute path of the PDB file.

    Returns
    -------
    str
        Dummy PDB code of the uploaded structure, which can be used instead of a PDB code.
    """

    # Open the local PDB file for reading in binary mode
    with open(Path(filepath).with_suffix(".pdb"), "rb") as f:
        # Post API query and get the response
        url_of_id = _send_request_get_results(
            "post",
            [APIConsts.FileUpload.RESPONSE_MSG["url_of_id"]],
            APIConsts.FileUpload.URL, 
            files={APIConsts.FileUpload.REQUEST_MSG: f}
        )[0]
    
    protein_id = _send_request_get_results(
        "get",
        [APIConsts.FileUpload.RESPONSE_MSG_FETCH_ID["id"]],
        url_of_id
    )[0]
    return protein_id

def get_dogsitescorer_metadata(job_location, attempts=30):
    """
    Get results from a DoGSiteScorer query, i.e., the binding sites which are found over the protein surface,
    in the form of a table with the details about all detected pockets.

    Parameters
    ----------
    job_location : str
        Consists of the location of a finished DoGSiteScorer job on the proteins.plus web server.
    attempts : int
        The time waiting for the feedback from DoGSiteScorer service.

    Returns
    -------
    pandas.DataFrame
        Table with metadata on detected binding sites.
    """

    print(f"Querying for job at URL {job_location}...", end="")

    while attempts:
        # Get job results
        result = requests.get(job_location)
        result.raise_for_status()
        # Get URL of result table file
        response = result.json()
        if "result_table" in response:
            result_file = response["result_table"]
            break
        attempts -= 1
        print(".", end="")
        time.sleep(10)
    # Get result table (as string)
    result_table = requests.get(result_file).text
    # Load the table (csv format using "\t" as separator) with pandas DataFrame
    # We cannot load the table from a string directly but from a file
    # Use io.StringIO to wrap this string as file-like object as needed for read_csv method
    # See more: https://docs.python.org/3/library/io.html#io.StringIO
    result_table_df = pd.read_csv(io.StringIO(result_table), sep="\t").set_index("name")
    result_table_df = result_table_df[["lig_cov","poc_cov","lig_name","volume","enclosure","surface","depth","surf/vol","accept","donor","hydrophobic_interactions","hydrophobicity","metal","simpleScore","drugScore"]]
    return result_table_df

def submit_dogsitescorer_job_with_pdbid(pdb_code, chain_id, ligand=""):
    """
    Submit PDB ID to DoGSiteScorer webserver using their API and get back URL for job location.

    Parameters
    ----------
    pdb_code : str
        4-letter valid PDB ID, e.g. '3w32'.
    chain_id : str
        Chain ID, e.g. 'A'.
    ligand : str
        Name of ligand bound to PDB structure with pdb_id, e.g. 'W32_A_1101'.
        Currently, the ligand name must be checked manually on the DoGSiteScorer website.

    Returns
    -------
    str
        Job location URL for submitted query.

    References
    ----------
    Function is adapted from: https://github.com/volkamerlab/TeachOpenCADD/pull/3 (@jaimergp)
    """

    # Submit job to proteins.plus
    # For details on parameters see: https://proteins.plus/help/dogsite_rest
    r = requests.post(
        "https://proteins.plus/api/dogsite_rest",
        json={
            "dogsite": {
                "pdbCode": pdb_code,  # PDB code of protein
                "analysisDetail": "1",  # 1 = include subpockets in results
                "bindingSitePredictionGranularity": "1",  # 1 = include drugablity scores
                "ligand": ligand,  # if name is specified, ligand coverage is calculated
                "chain": chain_id,  # if chain is specified, calculation is only performed on this chain
            }
        },
        headers={"Content-type": "application/json", "Accept": "application/json"},
    )

    r.raise_for_status()

    return r.json()["location"]

def sort_binding_sites(dataframe, method):
    if method == 'drugScore':
        print('Sorting binding sites by drug score')
        dataframe=dataframe.sort_values(by=["drugScore"], ascending=False)
    elif method == 'volume':
        print('Sorting binding sites by volume')
        dataframe=dataframe.sort_values(by=["volume"], ascending=False)
    else:
        print('Sorting binding sites by {}'.format(method))
        dataframe=dataframe.sort_values(by=method, ascending=False)
    best_pocket_name = dataframe.iloc[0, :].name
    return best_pocket_name

def get_url_for_pockets(job_location, file_type="pdb"):
    """
    Get all pocket file locations for a finished DoGSiteScorer job
    for a selected file type (pdb/ccp4).

    Parameters
    ----------
    job_location : str
        URL of finished job submitted to the DoGSiteScorer web server.
    file_type : str
        Type of file to be returned (pdb/ccp4).

    Returns
    -------
    list
        List of all respective pocket file URLs.
    """

    # Get job results
    result = requests.get(job_location)

    if file_type == "pdb":
        # Get pocket residues
        return result.json()["residues"]
    elif file_type == "ccp4":
        # Get pocket volumes
        return result.json()["pockets"]
    else:
        raise ValueError(f"File type {file_type} not available.")
    
def get_selected_pocket_location(job_location, best_pocket, file_type="pdb"):
    """
    Get the selected binding site file location.

    Parameters
    ----------
    job_location : str
        URL of finished job submitted to the DoGSiteScorer web server.
    best_pocket : str
        Selected pocket id.
    file_type : str
        Type of file to be returned (pdb/ccp4).

    Returns
    ------
    str
        URL of selected pocket file on the DoGSiteScorer web server.
    """
    result = []

    # Get URL for all available pdb or ccp4 files
    pocket_files = get_url_for_pockets(job_location, file_type)

    for pocket_file in pocket_files:
        if file_type == "pdb":
            if f"{best_pocket}_res" in pocket_file:
                result.append(pocket_file)
        elif file_type == "ccp4":
            if f"{best_pocket}_gpsAll" in pocket_file:
                result.append(pocket_file)

    if len(result) > 1:
        raise TypeError(f'Multiple strings detected: {", ".join(result)}.')
    elif len(result) == 0:
        raise TypeError(f"No string detected.")
    else:
        pass

    return result[0]

def save_binding_site_to_file(binding_site_url, output_path):
    """
    Download and save the PDB and CCP4 files corresponding to the calculated binding sites.

    Parameters
    ----------
    binding_site_df : pandas.DataFrame
        Binding site data retrieved from the DoGSiteScorer webserver.
    output_path : str or pathlib.Path
        Local folder path to save the files in.
    """
    response = requests.get(binding_site_url)
    response.raise_for_status()
    response_file_content = response.content
    file_extension = ".pdb"
    file_name = 'DogSiteScorer_binding_site' + file_extension
    with open(Path(output_path) / file_name, "wb") as f:
        f.write(response_file_content)
    return

def calculate_pocket_coordinates_from_pocket_pdb_file(filepath):
    """
    Calculate the coordinates of a binding site using the binding site's PDB file
    downloaded from DoGSiteScorer.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Local filepath of the binding site's PDB file.

    Returns
    -------
    dict of list of int
        Binding site coordinates in format:
        `{'center': [x, y, z], 'size': [x, y, z]}`
    """
    def load_pdb_file_as_dataframe(pdb_file_text_content):
        ppdb = PandasPdb().read_pdb_from_list(pdb_file_text_content.splitlines(True))
        pdb_df = ppdb.df
        return pdb_df
    with open(Path(filepath).with_suffix(".pdb")) as f:
        pdb_file_text_content = f.read()
    pdb_file_df = load_pdb_file_as_dataframe(pdb_file_text_content)
    pocket_coordinates_data = pdb_file_df["OTHERS"].loc[5, "entry"]
    coordinates_data_as_list = pocket_coordinates_data.split()
    # select strings representing floats from a list of strings
    coordinates = [float(element) for element in coordinates_data_as_list if re.compile(r'\d+(?:\.\d*)').match(element)]
    pocket_coordinates = {
        "center": coordinates[:3],
        "size": [coordinates[-1] * 2 for dim in range(3)],
    }
    return pocket_coordinates

def binding_site_coordinates_dogsitescorer(pdbpath, w_dir, method='volume'):
    pdb_upload=upload_pdb_file(pdbpath)
    job_location = submit_dogsitescorer_job_with_pdbid(pdb_upload, 'A', '')
    binding_site_df = get_dogsitescorer_metadata(job_location)
    best_binding_site = sort_binding_sites(binding_site_df, method)
    pocket = get_selected_pocket_location(job_location, best_binding_site)
    save_binding_site_to_file(pocket, w_dir)
    pocket_coordinates=calculate_pocket_coordinates_from_pocket_pdb_file(w_dir+'/DogSiteScorer_binding_site.pdb')
    return pocket_coordinates