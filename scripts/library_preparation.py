from typing import Optional
import os
import subprocess
from subprocess import DEVNULL, STDOUT, PIPE
import multiprocessing
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from chembl_structure_pipeline import standardizer
from pkasolver.query import calculate_microstate_pka_values
from scripts.utilities import *
from pathlib import Path
import tqdm
import concurrent.futures


def standardize_molecule(molecule):
    standardized_molecule = standardizer.standardize_mol(molecule)
    standardized_molecule = standardizer.get_parent_mol(standardized_molecule)
    return standardized_molecule


def standardize_library(input_sdf, id_column):
    """
    Standardizes a docking library using the ChemBL Structure Pipeline.

    Args:
        input_sdf (str): The path to the input SDF file containing the docking library.
        id_column (str): The name of the column in the SDF file that contains the compound IDs.

    Returns:
        None. The function writes the standardized molecules to a new SDF file.

    Raises:
        Exception: If there is an error loading the library SDF file.
        Exception: If there is an error converting SMILES to RDKit molecules.
        Exception: If there is an error writing the standardized library SDF file.
    """
    printlog('Standardizing docking library using ChemBL Structure Pipeline...')
    # Load Original Library SDF into Pandas
    try:
        df = PandasTools.LoadSDF(
            str(input_sdf),
            idName=id_column,
            molColName='Molecule',
            includeFingerprints=False,
            embedProps=True,
            removeHs=True,
            strictParsing=True,
            smilesName='SMILES')
        df.rename(columns={id_column: 'ID'}, inplace=True)
        n_cpds_start = len(df)
    except BaseException:
        printlog('ERROR: Failed to Load library SDF file!')
        raise Exception('Failed to Load library SDF file!')
    try:
        df.drop(columns='Molecule', inplace=True)
        df['Molecule'] = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]
    except Exception as e:
        printlog('ERROR: Failed to convert SMILES to RDKit molecules!' + e)
    # Standardize molecules using ChemBL Pipeline
    df['Molecule'] = [standardizer.get_parent_mol(standardizer.standardize_mol(mol)) for mol in df['Molecule']]
    df[['Molecule', 'flag']] = pd.DataFrame(df['Molecule'].tolist(), index=df.index)
    df = df.drop(columns='flag')
    df = df.loc[:, ~df.columns.duplicated()].copy()
    n_cpds_end = len(df)
    printlog(
        f'Standardization of compound library finished: Started with {n_cpds_start}, ended with {n_cpds_end} : {n_cpds_start-n_cpds_end} compounds lost')
    # Write standardized molecules to standardized SDF file
    wdir = Path(input_sdf).parent
    output_sdf = wdir / 'temp' / 'standardized_library.sdf'
    try:
        PandasTools.WriteSDF(
            df,
            str(output_sdf),
            molColName='Molecule',
            idName='ID',
            allNumeric=True)
    except BaseException:
        printlog('ERROR: Failed to write standardized library SDF file!')
        raise Exception('Failed to write standardized library SDF file!')


def standardize_library_futures(input_sdf, id_column, ncpus):
    """
    Standardizes a docking library using the ChemBL Structure Pipeline.

    Args:
        input_sdf (str): The path to the input SDF file containing the docking library.
        id_column (str): The name of the column in the SDF file that contains the compound IDs.
        ncpus (int): The number of CPUs to use for parallelization.

    Returns:
        pandas.DataFrame: The standardized DataFrame containing the standardized molecules.
    """

    printlog('Standardizing docking library using ChemBL Structure Pipeline...')
    try:
        df = PandasTools.LoadSDF(
            str(input_sdf),
            molColName='Molecule',
            idName=id_column,
            removeHs=True,
            strictParsing=True,
            smilesName='SMILES')
        df.rename(columns={id_column: 'ID'}, inplace=True)
        df['Molecule'] = [Chem.MolFromSmiles(
            smiles) for smiles in df['SMILES']]
        n_cpds_start = len(df)
    except Exception as e:
        printlog(
            'ERROR: Failed to Load library SDF file or convert SMILES to RDKit molecules!')
        printlog(e)
    with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
        df['Molecule'] = list(
            tqdm.tqdm(
                executor.map(
                    standardize_molecule,
                    df['Molecule']),
                total=len(
                    df['Molecule']),
                desc='Standardizing molecules',
                unit='mol'))
    df[['Molecule', 'flag']] = pd.DataFrame(
        df['Molecule'].tolist(), index=df.index)
    df = df.drop(columns='flag')
    df = df.loc[:, ~df.columns.duplicated()].copy()
    n_cpds_end = len(df)
    printlog(
        f'Standardization of compound library finished: Started with {n_cpds_start}, ended with {n_cpds_end} : {n_cpds_start-n_cpds_end} compounds lost')
    wdir = Path(input_sdf).parent
    output_sdf = wdir / 'temp' / 'standardized_library.sdf'
    PandasTools.WriteSDF(
        df,
        str(output_sdf),
        molColName='Molecule',
        idName='ID')
    return df


def protonate_library_pkasolver(input_sdf):
    printlog('Calculating protonation states using pkaSolver...')
    try:
        input_df = PandasTools.LoadSDF(
            str(input_sdf),
            molColName='Molecule',
            idName='ID',
            removeHs=True,
            strictParsing=True,
            smilesName='SMILES')
        input_df['Molecule'] = [Chem.MolFromSmiles(
            smiles) for smiles in input_df['SMILES']]
        n_cpds_start = len(input_df)
    except Exception as e:
        printlog(
            'ERROR: Failed to Load library SDF file or convert SMILES to RDKit molecules!')
        printlog(e)
    microstate_pkas = pd.DataFrame(
        calculate_microstate_pka_values(mol) for mol in input_df['Molecule'])
    missing_prot_state = microstate_pkas[microstate_pkas[0].isnull(
    )].index.tolist()
    microstate_pkas = microstate_pkas.iloc[:, 0].dropna()
    protonated_df = pd.DataFrame(
        {"Molecule": [mol.ph7_mol for mol in microstate_pkas]})
    try:
        for x in missing_prot_state:
            if x > protonated_df.index.max() + 1:
                printlog("Invalid insertion")
            else:
                protonated_df = Insert_row(
                    x, protonated_df, input_df.loc[x, 'Rdkit_mol'])
    except Exception as e:
        printlog('ERROR in adding missing protonating state')
        printlog(e)
    protonated_df['ID'] = input_df['ID']
    protonated_df = protonated_df.loc[:, ~
                                      protonated_df.columns.duplicated()].copy()
    n_cpds_end = len(input_df)
    printlog(
        f'Protonation of compound library finished: Started with {n_cpds_start}, ended with {n_cpds_end} : {n_cpds_start-n_cpds_end} compounds lost')
    output_sdf = Path(input_sdf).parent / 'protonated_library.sdf'
    PandasTools.WriteSDF(
        protonated_df,
        str(output_sdf),
        molColName='Molecule',
        idName='ID')
    return

# NOT WORKING


def generate_confomers_RDKit(input_sdf, ID):
    output_sdf = Path(input_sdf).parent / '3D_library_RDkit.sdf'
    try:
        genConf_command = f'python software/genConf.py -isdf {input_sdf} -osdf {output_sdf} -n 1'
        os.system(genConf_command)
    except Exception as e:
        printlog('ERROR: Failed to generate conformers!')
        printlog(e)
    return output_sdf


def generate_conformers_GypsumDL_withprotonation(
        input_sdf, ncpus):
    printlog(
        'Calculating protonation states and generating 3D conformers using GypsumDL...')
    try:
        gypsum_dl_command = f'python software/gypsum_dl-1.2.0/run_gypsum_dl.py -s {input_sdf} -o {Path(input_sdf).parent} --job_manager multiprocessing -p {ncpus} -m 1 -t 10 --min_ph 6.5 --max_ph 7.5 --pka_precision 1 --skip_alternate_ring_conformations --skip_making_tautomers --skip_enumerate_chiral_mol --skip_enumerate_double_bonds --max_variants_per_compound 1'
        subprocess.call(
            gypsum_dl_command,
            shell=True,
            stdout=DEVNULL,
            stderr=STDOUT)
    except Exception as e:
        printlog('ERROR: Failed to generate protomers and conformers!')
        printlog(e)


def generate_conformers_GypsumDL_noprotonation(
        input_sdf, ncpus):
    printlog('Generating 3D conformers using GypsumDL...')
    try:
        gypsum_dl_command = f'python software/gypsum_dl-1.2.0/run_gypsum_dl.py -s {input_sdf} -o {Path(input_sdf).parent} --job_manager multiprocessing -p {ncpus} -m 1 -t 10 --skip_adding_hydrogen --skip_alternate_ring_conformations --skip_making_tautomers --skip_enumerate_chiral_mol --skip_enumerate_double_bonds --max_variants_per_compound 1'
        subprocess.call(
            gypsum_dl_command,
            shell=True,
            stdout=DEVNULL,
            stderr=STDOUT)
    except Exception as e:
        printlog('ERROR: Failed to generate conformers!')
        printlog(e)


def cleanup(input_sdf: str) -> pd.DataFrame:
    """
    Cleans up the temporary files generated during the library preparation process.

    Args:
        input_sdf (str): The path to the input SDF file containing the compound library.

    Returns:
        pd.DataFrame: The final DataFrame containing the cleaned-up library with only the 'Molecule' and 'ID' columns.
    """
    printlog('Cleaning up files...')
    wdir = Path(input_sdf).parent

    # Load the successfully generated conformers from the GypsumDL process into a pandas DataFrame
    gypsum_df = PandasTools.LoadSDF(str(wdir / 'temp' / 'gypsum_dl_success.sdf'), molColName='Molecule',idName='ID', removeHs=False, strictParsing=True)

    # Remove the first row of the DataFrame, which contains the original input molecule
    final_df = gypsum_df.iloc[1:, :]

    # Select only the 'Molecule' and 'ID' columns from the DataFrame
    final_df = final_df[['Molecule', 'ID']]

    # Get the number of compounds in the final DataFrame
    n_cpds_end = len(final_df)

    # Write the final DataFrame to a new SDF file
    PandasTools.WriteSDF(final_df, str(wdir / 'temp' / 'final_library.sdf'), molColName='Molecule', idName='ID')

    # Delete the temporary files generated during the library preparation process
    (wdir / 'temp' / 'gypsum_dl_success.sdf').unlink(missing_ok=True)
    (wdir / 'temp' / 'protonated_library.sdf').unlink(missing_ok=True)
    (wdir / 'temp' / 'standardized_library.sdf').unlink(missing_ok=True)
    (wdir / 'temp' / 'gypsum_dl_failed.smi').unlink(missing_ok=True)

    printlog(f'Preparation of compound library finished: ended with {n_cpds_end}')

    return


def prepare_library(input_sdf: str, id_column: str, protonation: str, ncpus: int) -> pd.DataFrame:
    """
    Prepares a docking library for further analysis.
    
    Args:
        input_sdf (str): The path to the input SDF file containing the docking library.
        id_column (str): The name of the column in the SDF file that contains the compound IDs.
        protonation (str): The method to use for protonation. Can be 'pkasolver', 'GypsumDL', or any other value for no protonation.
        ncpus (int): The number of CPUs to use for parallelization.
        
    Returns:
        pd.DataFrame: The final cleaned DataFrame containing the standardized, protonated (if applicable), and 3D conformer-generated molecules.
    """
    wdir = Path(input_sdf).parent
    standardized_sdf = wdir / 'temp' / 'standardized_library.sdf'
    
    if not standardized_sdf.is_file():
        standardize_library_futures(input_sdf, id_column, ncpus)
    
    protonated_sdf = wdir / 'temp' / 'protonated_library.sdf'
    
    if not protonated_sdf.is_file():
        if protonation == 'pkasolver':
            protonate_library_pkasolver(standardized_sdf)
            generate_conformers_GypsumDL_noprotonation(protonated_sdf, ncpus)
        elif protonation == 'GypsumDL':
            generate_conformers_GypsumDL_withprotonation(standardized_sdf, ncpus)
        else:
            generate_conformers_GypsumDL_noprotonation(standardized_sdf, ncpus)
    else:
        generate_conformers_GypsumDL_noprotonation(protonated_sdf, ncpus)
    
    cleanup(input_sdf)
    return

