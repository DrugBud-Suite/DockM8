import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from chembl_structure_pipeline import standardizer
from pkasolver.query import calculate_microstate_pka_values
from scripts.utilities import Insert_row
import multiprocessing
from IPython.display import display
from pathlib import Path
import dask.dataframe as dd
from dask import delayed

def standardize_library(input_sdf, id_column):
    print('Standardizing docking library using ChemBL Structure Pipeline...')
    #Load Original Library SDF into Pandas
    try:
        df = PandasTools.LoadSDF(input_sdf, idName=id_column, molColName='Molecule', includeFingerprints=False, embedProps=True, removeHs=True, strictParsing=True, smilesName='SMILES')
        df.rename(columns = {id_column:'ID'}, inplace = True)
    except:
        print('ERROR: Failed to Load library SDF file!')
    try:
        df.drop(columns = 'Molecule', inplace = True)
        df['Molecule'] = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]
    except:
        print('ERROR: Failed to convert SMILES to RDKit molecules!')
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

def standardize_library_multiprocessing(input_sdf, id_column):
    print('Standardizing docking library using ChemBL Structure Pipeline...')
    #Load Original Library SDF into Pandas
    try:
        df = PandasTools.LoadSDF(input_sdf, idName=id_column, molColName='Molecule', includeFingerprints=False, embedProps=True, removeHs=True, strictParsing=True, smilesName='SMILES')
        df.rename(columns = {id_column:'ID'}, inplace = True)
    except:
        print('ERROR: Failed to Load library SDF file!')
    try:
        df.drop(columns = 'Molecule', inplace = True)
        df['Molecule'] = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]
    except:
        print('ERROR: Failed to convert SMILES to RDKit molecules!')
    #Standardize molecules using ChemBL Pipeline
    molecules = df['Molecule']
    def standardize_molecule(molecule):
        standardized_molecule = standardizer.standardize_mol(molecule)
        standardized_molecule = standardizer.get_parent_mol(standardized_molecule)
        return standardized_molecule
    with multiprocessing.Pool() as p:
        standardized_molecules = p.map(standardize_molecule, molecules)
    df['Molecule'] = standardized_molecules
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

def standardize_library_multiprocessing_simplified(input_sdf, id_column):
    print('Standardizing docking library using ChemBL Structure Pipeline...')
    try:
        df = PandasTools.LoadSDF(input_sdf, molColName='Molecule', idName=id_column, removeHs=True, strictParsing=True, smilesName='SMILES')
        df.rename(columns = {id_column:'ID'}, inplace = True)
        df['Molecule'] = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]
    except:
        print('ERROR: Failed to Load library SDF file or convert SMILES to RDKit molecules!')
    def standardize_molecule(molecule):
        standardized_molecule = standardizer.standardize_mol(molecule)
        standardized_molecule = standardizer.get_parent_mol(standardized_molecule)
        return standardized_molecule
    pool = multiprocessing.Pool()
    df['Molecule'] = pool.map(standardize_molecule, df['Molecule'])
    pool.close()
    pool.join()
    df[['Molecule', 'flag']]=pd.DataFrame(df['Molecule'].tolist(), index=df.index)
    df=df.drop(columns='flag')
    wdir = os.path.dirname(input_sdf)
    output_sdf = wdir+'/temp/standardized_library.sdf'
    PandasTools.WriteSDF(df, output_sdf, molColName='Molecule', idName='ID')
    return

def standardize_library_dask(input_sdf, id_column):
    wdir = os.path.dirname(input_sdf)
    output_sdf = wdir+'/temp/standardized_library.sdf'
    print('Standardizing docking library using ChemBL Structure Pipeline...')
    try:
        df = dd.from_pandas(PandasTools.LoadSDF(input_sdf, molColName='Molecule', idName=id_column, removeHs=True, strictParsing=True, smilesName='SMILES'), npartitions=500)
        df = df.rename(columns = {id_column:'ID'})
        df['Molecule'] = df['SMILES'].map(Chem.MolFromSmiles)
    except:
        print('ERROR: Failed to Load library SDF file or convert SMILES to RDKit molecules!')
    def standardize_molecule(molecule):
        standardized_molecule = standardizer.standardize_mol(molecule)
        standardized_molecule = standardizer.get_parent_mol(standardized_molecule)
        return standardized_molecule
    df['Molecule'] = df['Molecule'].map(standardize_molecule, meta=('Molecule', object))
    #df[['Molecule', 'flag']]= df['Molecule'].map(delayed(lambda x: pd.Series(x.tolist())), meta=[('Molecule', object), ('flag', int)])
    final_df = df.compute()
    final_df[['Molecule', 'flag']]=pd.DataFrame(final_df['Molecule'].tolist(), index=final_df.index)
    final_df=final_df.drop(columns='flag')
    PandasTools.WriteSDF(final_df, output_sdf, molColName='Molecule', idName='ID')
    return

def protonate_library_pkasolver(input_sdf):
    try:
        input_df = PandasTools.LoadSDF(input_sdf, idName='ID', molColName='Molecule', includeFingerprints=False, embedProps=True, removeHs=True, strictParsing=True, smilesName='SMILES')
    except:
        print('ERROR: Failed to Load library SDF file!')
    #apply pkasolver package to get protonation states of standardized molecules
    try:
        #create a new column of the calculated microstate pka values of converted rdkit molecules
        input_df['Rdkit_mol'] = [Chem.MolFromSmiles(mol) for mol in input_df['SMILES']]
        #generate protonation states and choosing the first recommended protonation state 
        mols_df = pd.DataFrame(calculate_microstate_pka_values(mol, only_dimorphite=False) for mol in input_df['Rdkit_mol'])
        #generate list of indecies of all missing protonating states
        missing_prot_state = mols_df[mols_df[0].isnull()].index.tolist()
        #drop all missing values
        mols_df = mols_df.iloc[:, 0].dropna()
    except:
        print('ERROR: no SMILES structure found')
    #choosing ph7_mol in our dataframe
    try:
        protonated_df= pd.DataFrame({"Molecule" : [mol.ph7_mol for mol in mols_df]})
        print(protonated_df)
    except:
        print('ERROR: protonation state not found')
    #adding original molecule for molecules that has no protonation state.
    try:
        for x in missing_prot_state:
            if x > protonated_df.index.max()+1:
                print("Invalid insertion")
            else:
                protonated_df = Insert_row(x, protonated_df, input_df.loc[x, 'Rdkit_mol'])
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
    return

def protonate_library_pkasolver_simplified(input_sdf):
    input_df = PandasTools.LoadSDF(input_sdf, idName='ID', removeHs=True, strictParsing=True, smilesName='SMILES')
    input_df['Rdkit_mol'] = [Chem.MolFromSmiles(mol) for mol in input_df['SMILES']]
    microstate_pkas = pd.DataFrame(calculate_microstate_pka_values(mol) for mol in input_df['Rdkit_mol'])
    missing_prot_state = microstate_pkas[microstate_pkas[0].isnull()].index.tolist()
    microstate_pkas = microstate_pkas.iloc[:, 0].dropna()
    print(microstate_pkas)
    protonated_df = pd.DataFrame({"Molecule" : [mol.ph7_mol for mol in microstate_pkas]})
    try:
        for x in missing_prot_state:
            if x > protonated_df.index.max()+1:
                print("Invalid insertion")
            else:
                protonated_df = Insert_row(x, protonated_df, input_df.loc[x, 'Rdkit_mol'])
    except:
        print('ERROR in adding missing protonating state')
    protonated_df['ID'] = input_df['ID']
    output_sdf = os.path.dirname(input_sdf)+'/protonated_library.sdf'
    PandasTools.WriteSDF(protonated_df, output_sdf, molColName='Molecule', idName='ID')
    return

def protonate_library_pkasolver_dask(input_sdf):
    input_df = dd.from_pandas(PandasTools.LoadSDF(input_sdf, idName='ID', removeHs=True, strictParsing=True, smilesName='SMILES'), npartitions=8)
    print('Reading molecules...')
    input_df['Rdkit_mol'] = input_df['SMILES'].map(Chem.MolFromSmiles)
    print('Calculating protonation...')
    microstate_pkas = [delayed(calculate_microstate_pka_values)(mol) for mol in input_df['Rdkit_mol']]
    microstate_pkas = delayed(pd.DataFrame)(microstate_pkas)
    missing_prot_state = delayed(microstate_pkas[microstate_pkas[0].isnull()].index.tolist())
    microstate_pkas = delayed(microstate_pkas.iloc[:, 0].dropna())
    microstate_pkas = microstate_pkas.compute()
    display(microstate_pkas)
    protonated_df = pd.DataFrame({"Molecule" : [mol.ph7_mol for mol in microstate_pkas]})
    display(protonated_df)
    #microstate_pkas = pd.DataFrame(calculate_microstate_pka_values(mol) for mol in input_df['Rdkit_mol'])
    #missing_prot_state = microstate_pkas[microstate_pkas[0].isnull()].index.tolist()
    #microstate_pkas = microstate_pkas.iloc[:, 0].dropna()
    #print(microstate_pkas)
    #protonated_df = pd.DataFrame({"Molecule" : [mol.ph7_mol for mol in microstate_pkas]})
    try:
        for x in missing_prot_state:
            if x > protonated_df.index.max()+1:
                print("Invalid insertion")
            else:
                protonated_df = Insert_row(x, protonated_df, input_df.loc[x, 'Rdkit_mol'])
    except:
        print('ERROR in adding missing protonating state')
    protonated_df['ID'] = input_df['ID']
    output_sdf = os.path.dirname(input_sdf)+'/protonated_library.sdf'
    PandasTools.WriteSDF(protonated_df, output_sdf, molColName='Molecule', idName='ID')
    return

def calc_pka(x):
    new_mol = calculate_microstate_pka_values(x, only_dimorphite=False)
    return new_mol

def protonate_library_pkasolver_multiprocessing(input_sdf):
    try:
        input_df = PandasTools.LoadSDF(input_sdf, idName='ID', molColName='Molecule', includeFingerprints=False, embedProps=True, removeHs=True, strictParsing=True, smilesName='SMILES')
    except:
        print('ERROR: Failed to Load library SDF file!')
    #apply pkasolver package to get protonation states of standardized molecules
    #try:
        #create a new column of the calculated microstate pka values of converted rdkit molecules
    input_df['Rdkit_mol'] = [Chem.MolFromSmiles(mol) for mol in input_df['SMILES']]
    toprocess = input_df['Rdkit_mol']
    with multiprocessing.Pool() as p:
        newmols = p.map(calculate_microstate_pka_values, toprocess)
        print(newmols)
    mols_df =pd.DataFrame(newmols)
    #generate list of indecies of all missing protonating states
    missing_prot_state = mols_df[mols_df[0].isnull()].index.tolist()
    #drop all missing values
    mols_df = mols_df.iloc[:, 0].dropna()
    #except:
        #print('ERROR: multiprocess failed')
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
                protonated_df = Insert_row(x, protonated_df, input_df.loc[x, 'Rdkit_mol'])
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
    return

def generate_confomers_RDKit(input_sdf, ID, software_path):
    wdir = os.path.dirname(input_sdf)
    output_sdf = wdir+'/3D_library_RDkit.sdf'
    try:
        genConf_command = 'python '+software_path+'/genConf.py -isdf '+input_sdf+' -osdf '+output_sdf+' -n 1'
        os.system(genConf_command)
    except:
        print('ERROR: Failed to generate conformers!')
    return output_sdf

def generate_conformers_GypsumDL_withprotonation(input_sdf, software_path):
    try:
        gypsum_dl_command = 'python '+software_path+'/gypsum_dl-1.2.0/run_gypsum_dl.py -s '+input_sdf+' --job_manager multiprocessing -p -1 -m 1 -t 10 --min_ph 6.5 --max_ph 7.5 --pka_precision 1 --skip_alternate_ring_conformations --skip_making_tautomers --skip_enumerate_chiral_mol --skip_enumerate_double_bonds --max_variants_per_compound 1'
        os.system(gypsum_dl_command)
    except:
        print('ERROR: Failed to generate protomers and conformers!')
        
def generate_conformers_GypsumDL_noprotonation(input_sdf, software_path):
    try:
        gypsum_dl_command = 'python '+software_path+'/gypsum_dl-1.2.0/run_gypsum_dl.py -s '+input_sdf+' --job_manager multiprocessing -p -1 -m 1 -t 10 --skip_adding_hydrogen --skip_alternate_ring_conformations --skip_making_tautomers --skip_enumerate_chiral_mol --skip_enumerate_double_bonds --max_variants_per_compound 1'
        os.system(gypsum_dl_command)
    except:
        print('ERROR: Failed to generate conformers!')        

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

def cleanup_simplified(input_sdf):
    print('Cleaning up files...')
    wdir = os.path.dirname(input_sdf)
    gypsum_df = PandasTools.LoadSDF(wdir+'/gypsum_dl_success.sdf', idName='ID', molColName='Molecule', strictParsing=True)
    final_df = gypsum_df.iloc[1:, :]
    final_df = final_df[['Molecule', 'ID']]
    PandasTools.WriteSDF(final_df, wdir+'/temp/final_library.sdf', molColName='Molecule', idName='ID')
    Path(wdir+'/gypsum_dl_success.sdf').unlink(missing_ok=True)
    Path(wdir+'/temp/protonated_library.sdf.sdf').unlink(missing_ok=True)
    Path(wdir+'/temp/standardized_library.sdf').unlink(missing_ok=True)
    Path(wdir+'/gypsum_dl_failed.smi').unlink(missing_ok=True)
    return final_df

def prepare_library_pkasolver_GypsumDL(input_sdf, id_column, software_path):
    wdir = os.path.dirname(input_sdf)
    standardized_sdf = wdir+'/temp/standardized_library.sdf'
    protonated_sdf = wdir+'/temp/protonated_library.sdf'
    standardize_library(input_sdf, id_column)
    protonate_library_pkasolver(standardized_sdf)
    generate_conformers_GypsumDL_noprotonation(protonated_sdf, software_path)
    cleaned_df_pkasolver = cleanup(input_sdf)
    return cleaned_df_pkasolver

def prepare_library_pkasolver_multiprocessing_GypsumDL(input_sdf, id_column, software_path):
    wdir = os.path.dirname(input_sdf)
    standardized_sdf = wdir+'/temp/standardized_library.sdf'
    protonated_sdf = wdir+'/temp/protonated_library.sdf'
    standardize_library_multiprocessing(input_sdf, id_column)
    protonate_library_pkasolver(standardized_sdf)
    generate_conformers_GypsumDL_noprotonation(protonated_sdf, software_path)
    cleaned_df_pkasolver = cleanup(input_sdf)
    return cleaned_df_pkasolver

def prepare_library_GypsumDL(input_sdf, id_column, software_path):
    wdir = os.path.dirname(input_sdf)
    standardized_sdf = wdir+'/temp/standardized_library.sdf'
    standardize_library_multiprocessing(input_sdf, id_column)
    generate_conformers_GypsumDL_withprotonation(standardized_sdf, software_path)
    cleaned_df_gypsum = cleanup(input_sdf)
    return cleaned_df_gypsum

def prepare_library_noprotonation(input_sdf, id_column, software_path):
    wdir = os.path.dirname(input_sdf)
    standardized_sdf = wdir+'/temp/standardized_library.sdf'
    standardize_library_multiprocessing(input_sdf, id_column)
    generate_conformers_GypsumDL_noprotonation(standardized_sdf, software_path)
    cleaned_df_noprotonation = cleanup(input_sdf)
    return cleaned_df_noprotonation

def prepare_library_pkasolver_GypsumDL_simplified(input_sdf, id_column, software_path):
    wdir = os.path.dirname(input_sdf)
    standardized_sdf = wdir+'/temp/standardized_library.sdf'
    protonated_sdf = wdir+'/temp/protonated_library.sdf'
    standardize_library_multiprocessing_simplified(input_sdf, id_column)
    protonate_library_pkasolver_simplified(standardized_sdf)
    generate_conformers_GypsumDL_noprotonation(protonated_sdf, software_path)
    cleaned_df_pkasolver = cleanup_simplified(input_sdf)
    return cleaned_df_pkasolver