import argparse
import concurrent.futures
import pebble
import datetime
import math
import os
import warnings
from pathlib import Path

import openbabel
import pandas as pd
from joblib import Parallel, delayed
from meeko import MoleculePreparation, PDBQTWriterLegacy
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    for i in range(0, len(df), compounds_per_core):
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
    #printlog(f'Split docking library into {file_counter - 1} files each containing {compounds_per_core} compounds')
    return split_files_folder

def split_sdf_str(dir, sdf_file, ncpus):
    sdf_file_name = Path(sdf_file).name.replace('.sdf', '')
    split_files_folder = Path(dir) / f'split_{sdf_file_name}'
    split_files_folder.mkdir(parents=True, exist_ok=True)
    
    with open(sdf_file, 'r') as infile:
        sdf_lines = infile.readlines()

    total_compounds = sdf_lines.count("$$$$\n")
    
    n = math.ceil(total_compounds // ncpus * 2)

    compound_count = 0
    current_compound_lines = []

    for line in sdf_lines:
        current_compound_lines.append(line)

        if line.startswith("$$$$"):
            compound_count += 1

            if compound_count % n == 0:
                output_file = split_files_folder / f"split_{compound_count // n}.sdf"
                with open(output_file, 'w') as outfile:
                    outfile.writelines(current_compound_lines)
                current_compound_lines = []

    # Write the remaining compounds to the last file
    if current_compound_lines:
        output_file = split_files_folder / f"split_{compound_count // n + 1}.sdf"
        with open(output_file, 'w') as outfile:
            outfile.writelines(current_compound_lines)
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
    split_files_folder = Path(dir) / f'split_{sdf_file_name}'
    split_files_folder.mkdir(exist_ok=True)
    for file in split_files_folder.iterdir():
        file.unlink()
    df = PandasTools.LoadSDF(str(sdf_file),
                            molColName='Molecule',
                            idName='ID',
                            includeFingerprints=False,
                            strictParsing=True)
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Splitting SDF file'):
        # Extract compound information from the row
        compound = row['Molecule']
        compound_id = row['ID']
        # Create a new DataFrame with a single compound
        compound_df = pd.DataFrame({'Molecule': [compound], 'ID': [compound_id]})
        # Output file path
        output_file = split_files_folder / f'split_{i + 1}.sdf'
        # Write the single compound DataFrame to an SDF file
        PandasTools.WriteSDF(compound_df,
                             str(output_file),
                             molColName='Molecule',
                             idName='ID')
    print(f'Split SDF file into {len(df)} files, each containing 1 compound')
    return split_files_folder

def split_sdf_single_str(dir, sdf_file):
    sdf_file_name = Path(sdf_file).name.replace('.sdf', '')
    split_files_folder = Path(dir) / f'split_{sdf_file_name}'
    split_files_folder.mkdir(parents=True, exist_ok=True)
    
    with open(sdf_file, 'r') as infile:
        sdf_lines = infile.readlines()

    n = sdf_lines.count("$$$$\n")

    compound_count = 0
    current_compound_lines = []

    for line in sdf_lines:
        current_compound_lines.append(line)

        if line.startswith("$$$$"):
            compound_count += 1

            if compound_count % n == 0:
                output_file = split_files_folder / f"split_{compound_count // n}.sdf"
                with open(output_file, 'w') as outfile:
                    outfile.writelines(current_compound_lines)
                current_compound_lines = []

    # Write the remaining compounds to the last file
    if current_compound_lines:
        output_file = split_files_folder / f"split_{compound_count // n + 1}.sdf"
        with open(output_file, 'w') as outfile:
            outfile.writelines(current_compound_lines)
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
    """
    Prints the given message along with a timestamp to the console and appends it to a log file.

    Args:
        message (str): The message to be logged.

    Returns:
        None
    """
    def timestamp_generator():
        dateTimeObj = datetime.datetime.now()
        return "[" + dateTimeObj.strftime("%Y-%b-%d %H:%M:%S") + "]"
    timestamp = timestamp_generator()
    msg = str(timestamp) + ": " + str(message)
    print(msg)
    log_file_path = Path(__file__).resolve().parent / '../log.txt'
    with open(log_file_path, 'a') as f_out:
        f_out.write(msg)


def convert_molecules(input_file : Path, output_file : Path, input_format : str, output_format : str):
    """
    Convert molecules from one file format to another.

    Args:
        input_file (Path): The path to the input file.
        output_file (Path): The path to the output file.
        input_format (str): The format of the input file.
        output_format (str): The format of the output file.

    Returns:
        Path: The path to the converted output file.
    """
    # For protein conversion to pdbqt file format using OpenBabel
    if input_format == 'pdb' and output_format == 'pdbqt':
        try:
            obConversion = openbabel.OBConversion()
            mol = openbabel.OBMol()
            obConversion.ReadFile(mol, str(input_file))
            obConversion.SetInAndOutFormats("pdb", "pdbqt")
            # Calculate Gasteiger charges
            charge_model = openbabel.OBChargeModel.FindType("gasteiger")
            charge_model.ComputeCharges(mol)
            obConversion.WriteFile(mol, str(output_file))
            # Remove all torsions from pdbqt output
            with open(output_file, 'r') as file:
                lines = file.readlines()
                lines = [line for line in lines if all(keyword not in line for keyword in ['between atoms:', 'BRANCH', 'ENDBRANCH', 'torsions', 'Active', 'ENDROOT', 'ROOT'])]
                lines = [line.replace(line, 'TER\n') if line.startswith('TORSDOF') else line for line in lines]
                with open(output_file, 'w') as file:
                    file.writelines(lines)
        except Exception as e:
            printlog(f"Error occurred during conversion using OpenBabel: {str(e)}")
        return output_file
    # For compound conversion to pdbqt file format using RDKit and Meeko
    if input_format == 'sdf' and output_format == 'pdbqt':
        try:
            for mol in Chem.SDMolSupplier(str(input_file), removeHs=False):
                preparator = MoleculePreparation(min_ring_size=10)
                mol = Chem.AddHs(mol)
                setup_list = preparator.prepare(mol)
                pdbqt_string = PDBQTWriterLegacy.write_string(setup_list[0])
                mol_name = mol.GetProp('_Name')
                output_path = Path(output_file) / f"{mol_name}.pdbqt"
                # Write the pdbqt string to the file
                with open(output_path, 'w') as f:
                    f.write(pdbqt_string[0])
        except Exception as e:
            printlog(f"Error occurred during conversion using Meeko: {str(e)}")
        return output_file
    # For general conversion using Pybel
    else:
        try:
            output = pybel.Outputfile(output_format, str(output_file), overwrite=True)
            for mol in pybel.readfile(input_format, str(input_file)):
                output.write(mol)
            output.close()
        except Exception as e:
            printlog(f"Error occurred during conversion using Pybel: {str(e)}")
        return output_file

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
            f'Expect the format of the molecule_file to be one of .mol2, .mol, .sdf, .pdbqt and .pdb, got {molecule_file}')
    return mol

        

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
                
def parallel_executor(function, list_of_objects : list, ncpus : int, backend = 'concurrent_process', **kwargs):
    
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
    if backend == "concurrent_process":
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
            jobs = [executor.submit(function, obj, **kwargs) for obj in list_of_objects]
            results = [job.result() for job in tqdm(concurrent.futures.as_completed(jobs), total=len(list_of_objects), desc=f"Running {function}")]
    
    if backend == "concurrent_process_silent":
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
            jobs = [executor.submit(function, obj, **kwargs) for obj in list_of_objects]
            results = [job.result() for job in concurrent.futures.as_completed(jobs)]
    
    if backend == "concurrent_thread":
        with concurrent.futures.ThreadPoolExecutor(max_workers=ncpus) as executor:
            jobs = [executor.submit(function, obj, **kwargs) for obj in list_of_objects]
            results = [job.result() for job in tqdm(concurrent.futures.as_completed(jobs), total=len(list_of_objects), desc=f"Running {function}")]
    
    if backend == 'joblib':
        jobs = [delayed(function)(obj, **kwargs) for obj in list_of_objects]
        results = Parallel(n_jobs=ncpus)(tqdm(jobs, total=len(list_of_objects), desc=f"Running {function}"))
    
    if backend == 'pebble_process':
        print(kwargs)
        with pebble.ProcessPool(max_workers=ncpus) as executor:
            jobs = [executor.schedule(function, args=(obj,), kwargs = kwargs) for obj in list_of_objects]
            results = [job.result() for job in jobs]
            
    if backend == 'pebble_thread':
        with pebble.ThreadPool(max_workers=ncpus) as executor:
            jobs = [executor.schedule(function, args=(obj,), kwargs = kwargs) for obj in list_of_objects]
            results = [job.result() for job in jobs]
    return results

def str2bool(v):
    """
    Converts a string representation of a boolean to a boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parallel_SDF_loader(sdf_path : Path, molcolName : str, idName : str, ncpus = os.cpu_count()-2, SMILES = None) -> pd.DataFrame:
    """
    Loads a SDF file in parallel using joblib library.

    Args:
        sdf_path (Path): The path to the SDF file.
        molcolName (str): The name of the molecule column in the SDF file.
        idName (str): The name of the ID column in the SDF file.
        includeFingerprints (bool): Whether to include fingerprints in the loaded DataFrame.
        strictParsing (bool): Whether to use strict parsing when loading the SDF file.

    Returns:
        DataFrame: The loaded SDF file as a DataFrame.
    """
    try:
        # Load the molecules from the SDF file
        mols = [m for m in Chem.MultithreadedSDMolSupplier(sdf_path,
                                                        numWriterThreads=ncpus, 
                                                        removeHs=False, 
                                                        strictParsing=True) if m is not None]
        data = []
        # Iterate over each molecule
        for mol in mols:
            # Get the properties of the molecule
            mol_props = {'Pose ID': mol.GetProp('_Name')}
            for prop in mol.GetPropNames():
                mol_props[prop] = mol.GetProp(prop)
                mol_props['Molecule'] = mol
            # Append the properties to the list
            data.append(mol_props)
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data).drop(columns=['mol_cond'])
        # Detect the columns that should be numeric
        for col in df.columns:
            if col not in [idName, molcolName, 'ID', 'Pose ID']:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
    except Exception as e:
        printlog(f"Error occurred during loading of SDF file: {str(e)}")
    return df
