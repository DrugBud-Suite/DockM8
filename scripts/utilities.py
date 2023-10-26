from pathlib import Path
from tqdm import tqdm
from rdkit.Chem import PandasTools
import math
import subprocess
from rdkit import Chem
from meeko import MoleculePreparation
import openbabel
import datetime
from subprocess import DEVNULL, STDOUT
from rdkit.Chem import AllChem
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import multiprocessing
import concurrent.futures
from joblib import Parallel, delayed

def split_sdf(dir, sdf_file, ncpus):
    """
    Split an SDF file into multiple smaller SDF files, each containing a subset of the original compounds.

    Args:
        dir (str): The directory where the split SDF files will be saved.
        sdf_file (str): The path to the original SDF file to be split.
        ncpus (int): The number of CPUs to use for the splitting process.

    Returns:
        Path: The path to the directory containing the split SDF files.
    """
    sdf_file_name = Path(sdf_file).name.replace('.sdf', '')
    print(f'Splitting SDF file {sdf_file_name}.sdf ...')
    split_files_folder = Path(dir) / f'split_{sdf_file_name}'
    split_files_folder.mkdir(parents=True, exist_ok=True)
    for file in split_files_folder.iterdir():
        file.unlink()
    df = PandasTools.LoadSDF(str(sdf_file),
                            molColName='Molecule',
                            idName='ID',
                            includeFingerprints=False,
                            strictParsing=True)
    compounds_per_core = math.ceil(len(df['ID']) / (ncpus * 2))
    used_ids = set()  # keep track of used 'ID' values
    file_counter = 1
    for i in tqdm(range(0, len(df), compounds_per_core), desc='Splitting files'):
        chunk = df[i:i + compounds_per_core]
        # remove rows with 'ID' values that have already been used
        chunk = chunk[~chunk['ID'].isin(used_ids)]
        used_ids.update(set(chunk['ID']))  # add new 'ID' values to used_ids
        output_file = split_files_folder / f'split_{file_counter}.sdf'
        PandasTools.WriteSDF(chunk,
                            str(output_file),
                            molColName='Molecule',
                            idName='ID')
        file_counter += 1
    print(f'Split docking library into {file_counter - 1} files each containing {compounds_per_core} compounds')
    return split_files_folder

def split_sdf_single(dir, sdf_file):
    """
    Split a single SDF file into multiple SDF files, each containing one compound.

    Args:
    - dir (str): The directory where the split SDF files will be saved.
    - sdf_file (str): The path to the input SDF file.

    Returns:
    - split_files_folder (Path): The path to the directory containing the split SDF files.
    """
    sdf_file_name = Path(sdf_file).name.replace('.sdf', '')
    print(f'Splitting SDF file {sdf_file_name}.sdf ...')
    split_files_folder = Path(dir) / f'split_{sdf_file_name}'
    split_files_folder.mkdir(exist_ok=True)
    for file in split_files_folder.iterdir():
        file.unlink()
    df = PandasTools.LoadSDF(str(sdf_file),
                            molColName='Molecule',
                            idName='ID',
                            includeFingerprints=False,
                            strictParsing=True)
    compounds_per_core = 1
    file_counter = 1
    for i, chunk in enumerate(tqdm(df, desc='Splitting files')):
        output_file = split_files_folder / f'split_{i+1}.sdf'
        PandasTools.WriteSDF(chunk,
                            str(output_file),
                            molColName='Molecule',
                            idName='ID')
    print(f'Split SDF file into {file_counter - 1} files each containing 1 compound')
    return split_files_folder


def Insert_row(row_number, df, row_value):
    """
    Inserts a row into a pandas DataFrame at the specified row number, shifting all other rows down.

    Parameters:
    row_number (int): The index of the row to insert.
    df (pandas.DataFrame): The DataFrame to insert the row into.
    row_value (list): The values to insert into the new row.

    Returns:
    pandas.DataFrame: The DataFrame with the new row inserted.
    """
    start_index = 0
    last_index = row_number
    start_lower = row_number
    last_lower = df.shape[0]
    # Create a list of upper_half index and lower half index
    upper_half = [*range(start_index, last_index, 1)]
    lower_half = [*range(start_lower, last_lower, 1)]
    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]
    # Combine the two lists
    index_ = upper_half + lower_half
    # Update the index of the dataframe
    df.index = index_
    # Insert a row at the end
    df.loc[row_number] = row_value
    # Sort the index labels
    df = df.sort_index()
    return df


def printlog(message):
    def timestamp_generator():
        dateTimeObj = datetime.datetime.now()
        return "[" + dateTimeObj.strftime("%Y-%b-%d %H:%M:%S") + "]"
    timestamp = timestamp_generator()
    msg = "\n" + str(timestamp) + ": " + str(message)
    print(msg)
    log_file_path = Path(__file__).resolve().parent / '../log.txt'
    with open(log_file_path, 'a') as f_out:
        f_out.write(msg)


def parallel_sdf_to_pdbqt(input_file: str, output_dir: str, ncpus: int) -> int:
    """
    Convert an SDF file to PDBQT format in parallel using multiple CPUs.

    Args:
        input_file (str): Path to the input SDF file.
        output_dir (str): Path to the output directory where the PDBQT files will be saved.
        ncpus (int): Number of CPUs to use for parallel processing.

    Returns:
        int: Number of molecules converted to PDBQT format.
    """
    def convert_molecule(mol, output_dir):
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("sdf", "pdbqt")
        # Calculate Gasteiger charges
        charge_model = openbabel.OBChargeModel.FindType("gasteiger")
        charge_model.ComputeCharges(mol)
        mol_name = mol.GetTitle()

        if not mol_name:
            mol_name = f"molecule_{mol.GetIdx()}"

        valid_filename = "".join(c for c in mol_name if c.isalnum() or c in (' ', '.', '_')).rstrip()
        output_file = Path(output_dir) / f"{valid_filename}.pdbqt"
        obConversion.WriteFile(mol, str(output_file))

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "sdf")

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    mol = openbabel.OBMol()
    not_at_end = obConversion.ReadFile(mol, str(input_file))
    molecules = []

    while not_at_end:
        molecules.append(openbabel.OBMol(mol))
        not_at_end = obConversion.Read(mol)

    try:
        with ThreadPoolExecutor(max_workers=ncpus) as executor:
            tasks = [executor.submit(convert_molecule, m, output_dir) for m in molecules]
            _ = [t.result() for t in tasks]
    except Exception as e:
        print(f"ERROR: Could note convert SDF file to .pdbqt: {e}")
    return len(molecules)


def meeko_to_pdbqt(sdf_path: str, output_dir: str) -> None:
    """
    Convert an SDF file containing Meeko molecules to PDBQT format and save the output files to the specified directory.

    Args:
        sdf_path (str): The path to the input SDF file.
        output_dir (str): The path to the output directory.

    Returns:
        None
    """
    for mol in Chem.SDMolSupplier(sdf_path, removeHs=False):
        preparator = MoleculePreparation(min_ring_size=10)
        mol = Chem.AddHs(mol)
        preparator.prepare(mol)
        pdbqt_string = preparator.write_pdbqt_string()

        # Extract the molecule name from the SDF file
        mol_name = mol.GetProp('_Name')

        # Create the output file path
        output_path = Path(output_dir) / f"{mol_name}.pdbqt"

        # Write the pdbqt string to the file
        with open(output_path, 'w') as f:
            f.write(pdbqt_string)


def load_molecule(molecule_file):
    """Load a molecule from a file.
    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format '.mol2', '.mol', '.sdf',
        '.pdbqt', or '.pdb'.
    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the loaded molecule.
    """
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    if molecule_file.endswith('.mol'):
        mol = Chem.MolFromMolFile(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as f:
            pdbqt_data = f.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError(
            f'Expect the format of the molecule_file to be '
            'one of .mol2, .mol, .sdf, .pdbqt and .pdb, got {molecule_file}')
    return mol
def convert_pdb_to_pdbqt(protein_file: Path) -> Path:
    """
    Converts a PDB file to PDBQT format using Open Babel.

    Args:
        protein_file (Path): Path to the input PDB file.

    Returns:
        Path: Path to the output PDBQT file.
    """
    # Define output file name
    pdbqt_file = protein_file.with_suffix('.pdbqt')

    # Open Babel command
    obabel_command = f'obabel {protein_file} -O {pdbqt_file} -partialcharges Gasteiger -xr'

    # Execute command
    try:
        subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        print(f'Conversion from PDB to PDBQT failed: {e}')
    return pdbqt_file
        

def delete_files(folder_path: str, save_file: str) -> None:
    """
    Deletes all files in a folder except for a specified save file.

    Args:
        folder_path (str): The path to the folder to delete files from.
        save_file (str): The name of the file to save.

    Returns:
        None
    """
    folder = Path(folder_path)
    for item in folder.iterdir():
        if item.is_file() and item.name != save_file:
            item.unlink()
        elif item.is_dir():
            delete_files(item, save_file)
            if not any(item.iterdir()) and item.name != save_file:
                item.rmdir()
                
def parallel_executor(function, split_files_sdfs, ncpus, **kwargs):
    """
    Executes a function in parallel using multiple processes.

    Args:
        function (function): The function to execute in parallel.
        split_files_sdfs (list): A list of input arguments to pass to the function.
        ncpus (int): The number of CPUs to use for parallel execution.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        The result of the function execution.
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
        jobs = []
        for split_file in split_files_sdfs:
            try:
                job = executor.submit(function, split_file, **kwargs)
                jobs.append(job)
            except Exception as e:
                printlog("Error in concurrent futures job creation: " + str(e))
        for job in tqdm(concurrent.futures.as_completed(jobs), total=len(split_files_sdfs)):
            try:
                res = job.result()
            except Exception as e:
                printlog("Error in concurrent futures job run: " + str(e))
    return res

def parallel_executor_joblib(function, split_files_sdfs, ncpus, **kwargs):
    """
    Executes a function in parallel using joblib library.

    Args:
        function (function): The function to be executed in parallel.
        split_files_sdfs (list): A list of split files.
        ncpus (int): The number of CPUs to be used for parallel execution.
        **kwargs: Additional keyword arguments to be passed to the function.

    Returns:
        list: A list of results from the parallel execution.
    """
    jobs = []
    for split_file in split_files_sdfs:
        try:
            job = delayed(function)(split_file, **kwargs)
            jobs.append(job)
        except Exception as e:
            printlog("Error in joblib job creation: " + str(e))
    results = Parallel(n_jobs=ncpus)(jobs)
    return results