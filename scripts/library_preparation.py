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
    printlog('Standardizing docking library using ChemBL Structure Pipeline...')
    #Load Original Library SDF into Pandas
    try:
        df = PandasTools.LoadSDF(input_sdf, idName=id_column, molColName='Molecule', includeFingerprints=False, embedProps=True, removeHs=True, strictParsing=True, smilesName='SMILES')
        df.rename(columns = {id_column:'ID'}, inplace = True)
        n_cpds_start = len(df)
    except:
        printlog('ERROR: Failed to Load library SDF file!')
    try:
        df.drop(columns = 'Molecule', inplace = True)
        df['Molecule'] = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]
    except:
        printlog('ERROR: Failed to convert SMILES to RDKit molecules!')
    #Standardize molecules using ChemBL Pipeline
    df['Molecule'] = [standardizer.standardize_mol(mol) for mol in df['Molecule']]
    df['Molecule'] = [standardizer.get_parent_mol(mol) for mol in df['Molecule']]
    df[['Molecule', 'flag']]=pd.DataFrame(df['Molecule'].tolist(), index=df.index)
    df=df.drop(columns='flag')
    df = df.loc[:,~df.columns.duplicated()].copy()
    n_cpds_end = len(df)
    printlog(f'Standardization of compound library finished: Started with {n_cpds_start}, ended with {n_cpds_end} : {n_cpds_start-n_cpds_end} compounds lost')
    #Write standardized molecules to standardized SDF file
    wdir = os.path.dirname(input_sdf)
    output_sdf = wdir+'/temp/standardized_library.sdf'
    try:
        PandasTools.WriteSDF(df, output_sdf, molColName='Molecule', idName='ID', allNumeric=True)
    except:
        printlog('ERROR: Failed to write standardized library SDF file!')
    return

def standardize_library_multiprocessing(input_sdf, id_column, ncpus):
    printlog('Standardizing docking library using ChemBL Structure Pipeline...')
    try:
        df = PandasTools.LoadSDF(input_sdf, molColName=None, idName=id_column, removeHs=True, strictParsing=True, smilesName='SMILES')
        df.rename(columns = {id_column:'ID'}, inplace = True)
        df['Molecule'] = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]
        n_cpds_start = len(df)
    except Exception as e: 
        printlog('ERROR: Failed to Load library SDF file or convert SMILES to RDKit molecules!')
        printlog(e)
    with multiprocessing.Pool(processes=ncpus) as p:
        df['Molecule'] = tqdm.tqdm(p.imap(standardize_molecule, df['Molecule']), total=len(df['Molecule']), desc='Standardizing molecules', unit='mol')
    df[['Molecule', 'flag']]=pd.DataFrame(df['Molecule'].tolist(), index=df.index)
    df=df.drop(columns='flag')
    df = df.loc[:,~df.columns.duplicated()].copy()
    n_cpds_end = len(df)
    printlog(f'Standardization of compound library finished: Started with {n_cpds_start}, ended with {n_cpds_end} : {n_cpds_start-n_cpds_end} compounds lost')
    wdir = os.path.dirname(input_sdf)
    output_sdf = wdir+'/temp/standardized_library.sdf'
    PandasTools.WriteSDF(df, output_sdf, molColName='Molecule', idName='ID')
    return

def standardize_library_futures(input_sdf, id_column, ncpus):
    printlog('Standardizing docking library using ChemBL Structure Pipeline...')
    try:
        df = PandasTools.LoadSDF(input_sdf, molColName='Molecule', idName=id_column, removeHs=True, strictParsing=True, smilesName='SMILES')
        df.rename(columns = {id_column:'ID'}, inplace = True)
        df['Molecule'] = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]
        n_cpds_start = len(df)
    except Exception as e: 
        printlog('ERROR: Failed to Load library SDF file or convert SMILES to RDKit molecules!')
        printlog(e)
    with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
        df['Molecule'] = list(tqdm.tqdm(executor.map(standardize_molecule, df['Molecule']), total=len(df['Molecule']), desc='Standardizing molecules', unit='mol'))
    df[['Molecule', 'flag']]=pd.DataFrame(df['Molecule'].tolist(), index=df.index)
    df=df.drop(columns='flag')
    df = df.loc[:,~df.columns.duplicated()].copy()
    n_cpds_end = len(df)
    printlog(f'Standardization of compound library finished: Started with {n_cpds_start}, ended with {n_cpds_end} : {n_cpds_start-n_cpds_end} compounds lost')
    wdir = os.path.dirname(input_sdf)
    output_sdf = wdir+'/temp/standardized_library.sdf'
    PandasTools.WriteSDF(df, output_sdf, molColName='Molecule', idName='ID')
    return

def protonate_library_pkasolver(input_sdf):
    printlog('Calculating protonation states using pkaSolver...')
    try:
        input_df = PandasTools.LoadSDF(input_sdf, molColName='Molecule', idName='ID', removeHs=True, strictParsing=True, smilesName='SMILES')
        input_df['Molecule'] = [Chem.MolFromSmiles(smiles) for smiles in input_df['SMILES']]
        n_cpds_start = len(input_df)
    except Exception as e: 
        printlog('ERROR: Failed to Load library SDF file or convert SMILES to RDKit molecules!')
        printlog(e)
    microstate_pkas = pd.DataFrame(calculate_microstate_pka_values(mol) for mol in input_df['Molecule'])
    missing_prot_state = microstate_pkas[microstate_pkas[0].isnull()].index.tolist()
    microstate_pkas = microstate_pkas.iloc[:, 0].dropna()
    protonated_df = pd.DataFrame({"Molecule" : [mol.ph7_mol for mol in microstate_pkas]})
    try:
        for x in missing_prot_state:
            if x > protonated_df.index.max()+1:
                printlog("Invalid insertion")
            else:
                protonated_df = Insert_row(x, protonated_df, input_df.loc[x, 'Rdkit_mol'])
    except Exception as e: 
        printlog('ERROR in adding missing protonating state')
        printlog(e)
    protonated_df['ID'] = input_df['ID']
    df = df.loc[:,~df.columns.duplicated()].copy()
    n_cpds_end = len(input_df)
    printlog(f'Standardization of compound library finished: Started with {n_cpds_start}, ended with {n_cpds_end} : {n_cpds_start-n_cpds_end} compounds lost')
    output_sdf = os.path.dirname(input_sdf)+'/protonated_library.sdf'
    PandasTools.WriteSDF(protonated_df, output_sdf, molColName='Molecule', idName='ID')
    return


#NOT WORKING
def generate_confomers_RDKit(input_sdf, ID, software_path):
    output_sdf = +os.path.dirname(input_sdf)+'/3D_library_RDkit.sdf'
    try:
        genConf_command = 'python '+software_path+'/genConf.py -isdf '+input_sdf+' -osdf '+output_sdf+' -n 1'
        os.system(genConf_command)
    except Exception as e: 
        printlog('ERROR: Failed to generate conformers!')
        printlog(e)
    return output_sdf

def generate_conformers_GypsumDL_withprotonation(input_sdf, software_path, ncpus):
    printlog('Calculating protonation states and generating 3D conformers using GypsumDL...')
    try:
        gypsum_dl_command = 'python '+software_path+'/gypsum_dl-1.2.0/run_gypsum_dl.py -s '+input_sdf+' -o '+os.path.dirname(input_sdf)+' --job_manager multiprocessing -p '+str(ncpus)+' -m 1 -t 10 --min_ph 6.5 --max_ph 7.5 --pka_precision 1 --skip_alternate_ring_conformations --skip_making_tautomers --skip_enumerate_chiral_mol --skip_enumerate_double_bonds --max_variants_per_compound 1'
        subprocess.call(gypsum_dl_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e: 
        printlog('ERROR: Failed to generate protomers and conformers!')
        printlog(e)
        
def generate_conformers_GypsumDL_noprotonation(input_sdf, software_path, ncpus):
    printlog('Generating 3D conformers using GypsumDL...')
    try:
        gypsum_dl_command = 'python '+software_path+'/gypsum_dl-1.2.0/run_gypsum_dl.py -s '+input_sdf+' -o '+os.path.dirname(input_sdf)+' --job_manager multiprocessing -p '+str(ncpus)+' -m 1 -t 10 --skip_adding_hydrogen --skip_alternate_ring_conformations --skip_making_tautomers --skip_enumerate_chiral_mol --skip_enumerate_double_bonds --max_variants_per_compound 1'
        subprocess.call(gypsum_dl_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e: 
        printlog('ERROR: Failed to generate conformers!')
        printlog(e)     

def cleanup(input_sdf):
    printlog('Cleaning up files...')
    wdir = os.path.dirname(input_sdf)
    gypsum_df = PandasTools.LoadSDF(wdir+'/temp/gypsum_dl_success.sdf', idName='ID', molColName='Molecule', strictParsing=True)
    final_df = gypsum_df.iloc[1:, :]
    final_df = final_df[['Molecule', 'ID']]
    n_cpds_end = len(final_df)
    printlog(f'Preparation of compound library finished: ended with {n_cpds_end}')
    PandasTools.WriteSDF(final_df, wdir+'/temp/final_library.sdf', molColName='Molecule', idName='ID')
    Path(wdir+'/temp/gypsum_dl_success.sdf').unlink(missing_ok=True)
    Path(wdir+'/temp/protonated_library.sdf').unlink(missing_ok=True)
    Path(wdir+'/temp/standardized_library.sdf').unlink(missing_ok=True)
    Path(wdir+'/temp/gypsum_dl_failed.smi').unlink(missing_ok=True)
    return final_df

def prepare_library(input_sdf, id_column, software_path, protonation, ncpus):
    wdir = os.path.dirname(input_sdf)
    standardized_sdf = wdir+'/temp/standardized_library.sdf'
    if os.path.isfile(standardized_sdf) == False:
        standardize_library_futures(input_sdf, id_column, ncpus)
    protonated_sdf = wdir+'/temp/protonated_library.sdf'
    if os.path.isfile(protonated_sdf) == False:
        if protonation == 'pkasolver':
            protonate_library_pkasolver(standardized_sdf)
            generate_conformers_GypsumDL_noprotonation(protonated_sdf, software_path, ncpus)
        elif protonation == 'GypsumDL':
            generate_conformers_GypsumDL_withprotonation(standardized_sdf, software_path, ncpus)
        else:
            generate_conformers_GypsumDL_noprotonation(standardized_sdf, software_path, ncpus)
    else:
        generate_conformers_GypsumDL_noprotonation(protonated_sdf, software_path, ncpus)
    cleaned_df = cleanup(input_sdf)
    return cleaned_df