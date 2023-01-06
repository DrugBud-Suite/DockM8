import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from chembl_structure_pipeline import standardizer
from pkasolver.query import calculate_microstate_pka_values
from scripts.utilities import Insert_row
import tqdm

def standardize_library(input_sdf, id_column):
    print('Standardizing docking library using ChemBL Structure Pipeline...')
    #Load Original Library SDF into Pandas
    try:
        df = PandasTools.LoadSDF(input_sdf, idName=id_column, molColName='Molecule',includeFingerprints=False, embedProps=True, removeHs=True, strictParsing=True, smilesName='SMILES')
        df.rename(columns = {id_column:'ID'}, inplace = True)
    except:
        print('ERROR: Failed to Load library SDF file!')
    #Standardize molecules using ChemBL Pipeline
    df['Molecule'] = [standardizer.standardize_mol(mol) for mol in df['Molecule']]
    df['Molecule'] = [standardizer.get_parent_mol(mol) for mol in df['Molecule']]
    df[['Molecule', 'flag']]=pd.DataFrame(df['Molecule'].tolist(), index=df.index)
    df=df.drop(columns='flag')
    #Write standardized molecules to standardized SDF file
    wdir = os.path.dirname(input_sdf)
    output_sdf = wdir+'/temp/standardized_library.sdf'
    #try:
    PandasTools.WriteSDF(df, output_sdf, molColName='Molecule', idName='ID', allNumeric=True)
    #except:
        #print('ERROR: Failed to write standardized library SDF file!')
    return

def protonate_library(input_sdf):
    try:
        input_df = PandasTools.LoadSDF(input_sdf, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=True, removeHs=True, strictParsing=True, smilesName='SMILES')
    except:
        print('ERROR: Failed to Load library SDF file!')
    #apply pkasolver package to get protonation states of standardized molecules
    try:
        #create a new column of the calculated microstate pka values of converted rdkit molecules
        #input_df['Rdkit_mol'] = [Chem.MolFromSmiles(mol) for mol in input_df['SMILES']]
        #generate protonation states and choosing the first recommended protonation state 
        mols_df = pd.DataFrame(calculate_microstate_pka_values(mol, only_dimorphite=False) for mol in input_df['Molecule'])
        #generate list of indecies of all missing protonating states
        missing_prot_state = mols_df[mols_df[0].isnull()].index.tolist()
        #drop all missing values
        mols_df = mols_df.iloc[:, 0].dropna()
    except:
        print('ERROR: no SMILES structure found')
    #choosing ph7_mol in our dataframe
    try:
        protonated_df= pd.DataFrame({"Molecule" : [mol.ph7_mol for mol in mols_df]})
    except:
        print('ERROR: protonation state not found')
    #adding original molecule for molecules that has no protonation state.
    try:
        for x in missing_prot_state:
            if x > protonated_df.index.max()+1:
                print("Invalid insertion")
            else:
                protonated_df = Insert_row(x, protonated_df, input_df.loc[x, 'Molecule'])
    except:
        print('ERROR in adding missing protonating state')
    id_list = input_df['ID']
    protonated_df['ID'] = id_list
    # Write protonated SDF file
    wdir = os.path.dirname(input_sdf)
    output_sdf = wdir+'/protonated_library.sdf'
    #try:
    PandasTools.WriteSDF(protonated_df, output_sdf, molColName='Molecule', idName='ID')
    #except:
        #print('\n**\n**\n**ERROR: Failed to write protonated library SDF file!')
    return protonated_df

def generate_confomers_RDKit(input_sdf, ID, software_path):
    wdir = os.path.dirname(input_sdf)
    output_sdf = wdir+'/3D_library_RDkit.sdf'
    try:
        genConf_command = 'python '+software_path+'/genConf.py -isdf '+input_sdf+' -osdf '+output_sdf+' -n 1'
        os.system(genConf_command)
    except:
        print('ERROR: Failed to generate conformers!')
    return output_sdf

def generate_confomers_GypsumDL(input_sdf, software_path):
    try:
        gypsum_dl_command = 'python '+software_path+'/gypsum_dl-1.2.0/run_gypsum_dl.py -s '+input_sdf+' --job_manager multiprocessing -p -1 -m 1 -t 10 --skip_adding_hydrogen --skip_alternate_ring_conformations --skip_making_tautomers --skip_enumerate_chiral_mol --skip_enumerate_double_bonds'
        os.system(gypsum_dl_command)
    except:
        print('ERROR: Failed to generate protomers and conformers!')
def cleanup(input_sdf):
    print('Cleaning up files...')
    wdir = os.path.dirname(input_sdf)
    gypsum_sdf = wdir+'/gypsum_dl_success.sdf'
    gypsum_df = PandasTools.LoadSDF(gypsum_sdf, idName='ID', molColName='Molecule', includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True, smilesName='SMILES')
    final_df = gypsum_df.iloc[1:, :]
    final_df = final_df[['Molecule', 'ID']]
    #Write prepared library to SDF file
    try:
        PandasTools.WriteSDF(final_df, wdir+'/temp/final_library.sdf', molColName='Molecule', idName='ID')
    except:
        print('ERROR: Failed to write prepared library SDF file!')
    #Remove temporary files
    try:
        os.remove(gypsum_sdf)
        os.remove(wdir+'/temp/protonated_library.sdf')
        os.remove(wdir+'/temp/standardized_library.sdf')
        os.remove(gypsum_sdf.replace('_success.sdf', '_failed.smi'))
    except:
        print('ERROR: Could not remove temporary library files completely!')
    return final_df

def prepare_library_GypsumDL(input_sdf, id_column, software_path):
    wdir = os.path.dirname(input_sdf)
    standardized_sdf = wdir+'/temp/standardized_library.sdf'
    protonated_sdf = wdir+'/temp/protonated_library.sdf'
    gypsum_sdf = wdir+'/temp/gypsum_dl_success.sdf'
    standardize_library(input_sdf, id_column)
    protonated_df = protonate_library(standardized_sdf)
    generate_confomers_GypsumDL(protonated_sdf, software_path)
    cleaned_df = cleanup(input_sdf)
    return cleaned_df