import os
import subprocess
from subprocess import DEVNULL, STDOUT, PIPE
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from chembl_structure_pipeline import standardizer
from pkasolver.query import calculate_microstate_pka_values
from scripts.utilities import Insert_row
from IPython.display import display
from pathlib import Path
import tqdm
import concurrent.futures


class Ligand:
    def __init__(
            self,
            docking_library,
            id_column,
            software_path,
            protonation_option,
            number_cpus):
        self.docking_library = docking_library
        self.id_column = id_column
        self.software_path = software_path
        self.protonation_option = protonation_option
        self.number_cpus = number_cpus
        self.standardized_sdf = os.path.dirname(
            self.docking_library) + '/temp/standardized_library.sdf'
        self.protonated_sdf = os.path.dirname(
            self.docking_library) + '/temp/protonated_library.sdf'
        self.standardized_df = None

    def standardization(self):
        if os.path.isfile(self.standardized_sdf) == False:
            self.standardize_library_futures(
                self.docking_library, self.id_column, self.number_cpus)

    def protonation(self):
        if os.path.isfile(self.protonated_sdf) == False:
            if self.protonation_option == 'pkasolver':
                self.protonate_library_pkasolver()
                self.generate_conformers_GypsumDL_noprotonation(
                    self.protonated_sdf)
            elif self.protonation_option == 'GypsumDL':
                self.generate_conformers_GypsumDL_withprotonation(
                    self.standardized_sdf)
            else:
                self.generate_conformers_GypsumDL_noprotonation(
                    self.standardized_sdf)
        else:
            self.generate_conformers_GypsumDL_noprotonation(
                self.protonated_sdf)
        self.cleanup()

    def standardize_library_futures(self):
        print('Standardizing docking library using ChemBL Structure Pipeline...')
        try:
            df = PandasTools.LoadSDF(
                self.docking_library,
                molColName='Molecule',
                idName=self.id_column,
                removeHs=True,
                strictParsing=True,
                smilesName='SMILES')  # gotta rename id column to ID
            df.rename(columns={self.id_column: 'ID'}, inplace=True)
            df['Molecule'] = [Chem.MolFromSmiles(
                smiles) for smiles in df['SMILES']]
            n_cpds_start = len(df)
        except Exception as e:
            print(
                'ERROR: Failed to Load library SDF file or convert SMILES to RDKit molecules!')
            print(e)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.number_cpus) as executor:
            df['Molecule'] = list(
                tqdm.tqdm(
                    executor.map(
                        self.standardize_molecule,
                        df['Molecule']),
                    total=len(
                        df['Molecule']),
                    desc='Standardizing molecules',
                    unit='mol'))
        df[['Molecule', 'flag']] = pd.DataFrame(
            df['Molecule'].tolist(), index=df.index)
        df = df.drop(columns='flag')
        n_cpds_end = len(df)
        print(
            f'Standardization of compound library finished: Started with {n_cpds_start}, ended with {n_cpds_end} : {n_cpds_start-n_cpds_end} compounds lost')
        PandasTools.WriteSDF(
            df,
            self.standardized_sdf,
            molColName='Molecule',
            idName='ID')
        self.standardized_df = df

    def standardize_molecule(self, molecule):
        standardized_molecule = standardizer.standardize_mol(molecule)
        standardized_molecule = standardizer.get_parent_mol(
            standardized_molecule)
        return standardized_molecule

    def protonate_library_pkasolver(self):
        print('Calculating protonation states using pkaSolver...')
        try:
            if self.standardized_df is None:
                input_df = PandasTools.LoadSDF(
                    self.standardized_sdf,
                    molColName=None,
                    idName='ID',
                    removeHs=True,
                    strictParsing=True,
                    smilesName='SMILES')
                input_df['Molecule'] = [Chem.MolFromSmiles(
                    smiles) for smiles in input_df['SMILES']]
            else:
                input_df = self.standardized_df
                n_cpds_start = len(input_df)
        except Exception as e:
            print(
                'ERROR: Failed to Load library SDF file or convert SMILES to RDKit molecules!')
            print(e)
        microstate_pkas = pd.DataFrame(
            calculate_microstate_pka_values(mol) for mol in input_df['Molecule'])
        missing_prot_state = microstate_pkas[microstate_pkas[0].isnull(
        )].index.tolist()
        microstate_pkas = microstate_pkas.iloc[:, 0].dropna()
        print(microstate_pkas)
        protonated_df = pd.DataFrame(
            {"Molecule": [mol.ph7_mol for mol in microstate_pkas]})
        try:
            for x in missing_prot_state:
                if x > protonated_df.index.max() + 1:
                    print("Invalid insertion")
                else:
                    protonated_df = Insert_row(
                        x, protonated_df, input_df.loc[x, 'Rdkit_mol'])
        except Exception as e:
            print('ERROR in adding missing protonating state')
            print(e)
        protonated_df['ID'] = input_df['ID']
        n_cpds_end = len(input_df)
        print(
            f'Standardization of compound library finished: Started with {n_cpds_start}, ended with {n_cpds_end} : {n_cpds_start-n_cpds_end} compounds lost')
        PandasTools.WriteSDF(
            protonated_df,
            self.protonated_sdf,
            molColName='Molecule',
            idName='ID')

    def generate_conformers_GypsumDL_withprotonation(self, sdf_path):
        print(
            'Calculating protonation states and generating 3D conformers using GypsumDL...')
        try:
            gypsum_dl_command = 'python ' + self.software_path + '/gypsum_dl-1.2.0/run_gypsum_dl.py -s ' + sdf_path + ' -o ' + os.path.dirname(sdf_path) + ' --job_manager multiprocessing -p ' + str(
                self.number_cpus) + ' -m 1 -t 10 --min_ph 6.5 --max_ph 7.5 --pka_precision 1 --skip_alternate_ring_conformations --skip_making_tautomers --skip_enumerate_chiral_mol --skip_enumerate_double_bonds --max_variants_per_compound 1'
            subprocess.call(
                gypsum_dl_command,
                shell=True,
                stdout=DEVNULL,
                stderr=STDOUT)
        except Exception as e:
            print('ERROR: Failed to generate protomers and conformers!')
            print(e)

    def generate_conformers_GypsumDL_noprotonation(self, sdf_path):
        print('Generating 3D conformers using GypsumDL...')
        try:
            gypsum_dl_command = 'python ' + software_path + '/gypsum_dl-1.2.0/run_gypsum_dl.py -s ' + sdf_path + ' -o ' + os.path.dirname(sdf_path) + ' --job_manager multiprocessing -p ' + str(
                self.number_cpus) + ' -m 1 -t 10 --skip_adding_hydrogen --skip_alternate_ring_conformations --skip_making_tautomers --skip_enumerate_chiral_mol --skip_enumerate_double_bonds --max_variants_per_compound 1'
            subprocess.call(
                gypsum_dl_command,
                shell=True,
                stdout=DEVNULL,
                stderr=STDOUT)
        except Exception as e:
            print('ERROR: Failed to generate conformers!')
            print(e)

    def cleanup(self):
        print('Cleaning up files...')
        wdir = os.path.dirname(self.protonated_sdf)
        gypsum_df = PandasTools.LoadSDF(
            wdir + '/temp/gypsum_dl_success.sdf',
            idName='ID',
            molColName='Molecule',
            strictParsing=True)
        final_df = gypsum_df.iloc[1:, :]
        final_df = final_df[['Molecule', 'ID']]
        n_cpds_end = len(final_df)
        print(
            f'Preparation of compound library finished: ended with {n_cpds_end}')
        PandasTools.WriteSDF(
            final_df,
            wdir +
            '/temp/final_library.sdf',
            molColName='Molecule',
            idName='ID')
        Path(wdir + '/temp/gypsum_dl_success.sdf').unlink(missing_ok=True)
        Path(wdir + '/temp/protonated_library.sdf').unlink(missing_ok=True)
        Path(wdir + '/temp/standardized_library.sdf').unlink(missing_ok=True)
        Path(wdir + '/temp/gypsum_dl_failed.smi').unlink(missing_ok=True)
