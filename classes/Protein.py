import warnings
from random import randint
import pandas as pd
import os
import glob
import numpy as np
import copy
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import AllChem
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class Protein:
    def __init__(self, ligand_file, protein_file, cut):
        """
        ligand_file: format mol
        protein_file: format pdb
        """
        print(
            f'Extracting pocket from {protein_file} using {ligand_file} as reference ligand')
        self.ligand_file = ligand_file
        self.protein_file = protein_file
        self.ligand_mol = None
        self.cut = cut
        self.pocket_path = os.path.basename(protein_file).split(".")[
            0] + "_pocket.pdb"

        os.remove(self.temp_file)
        print(
            f'Finished extracting pocket from {protein_file} using {ligand_file} as reference ligand')

    def load_molecule(self):
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
        if self.ligand_file.endswith('.mol2'):
            self.ligand_mol = Chem.MolFromMol2File(
                self.ligand_file, sanitize=False, removeHs=False)
        if self.ligand_file.endswith('.mol'):
            self.ligand_mol = Chem.MolFromMolFile(
                self.ligand_file, sanitize=False, removeHs=False)
        elif self.ligand_file.endswith('.sdf'):
            supplier = Chem.SDMolSupplier(
                self.ligand_file, sanitize=False, removeHs=False)
            self.ligand_molself.ligand_mol = supplier[0]
        elif mself.ligand_file.endswith('.pdbqt'):
            with open(self.ligand_file) as f:
                pdbqt_data = f.readlines()
            pdb_block = ''
            for line in pdbqt_data:
                pdb_block += '{}\n'.format(line[:66])
            self.ligand_mol = Chem.MolFromPDBBlock(
                pdb_block, sanitize=False, removeHs=False)
        elif self.ligand_file.endswith('.pdb'):
            self.ligand_mol = Chem.MolFromPDBFile(
                self.ligand_file, sanitize=False, removeHs=False)
        else:
            return ValueError(
                f'Expect the format of the molecule_file to be '
                'one of .mol2, .mol, .sdf, .pdbqt and .pdb, got {self.ligand_file}')
        return 1

    def process_pro_and_lig(self):
        ppdb = PandasPdb()
        ppdb = ppdb.read_pdb(self.protein_file)
        protein_biop = ppdb.df['ATOM']
        protein_cut = self.select_cut_residue(
            protein_biop, self.ligand_mol, self.cut)
        ppdb.df['ATOM'] = protein_cut
        newmolname = str(randint(1, 1000000)).zfill(10)
        name = 'pocket_{}.pdb'.format(newmolname)
        ppdb.to_pdb(path=name, records=['ATOM'])
        self.pocket_mol = Chem.MolFromPDBFile(name, removeHs=False)
        self.temp_file = name
        Chem.MolToPDBFile(self.pocket_mol, self.pocket_path)
        return 1

    def select_cut_residue(self, protein_biop, ligand_mol, cut):
        """
        pro: biopandas DataFrame
        lig: rdkit mol
        """
        protein_df = self.cal_pro_min_dist(protein_biop, ligand_mol)
        protein_df['chain_rid'] = protein_df.apply(lambda row: str(
            row['chain_id']) + str(row['residue_number']), axis=1)
        protein_df = protein_df[protein_df['min_dist'] < cut]
        use_res = list(set(list(protein_df['chain_rid'])))
        protein_df = protein_df[protein_df['chain_rid'].isin(use_res)]
        protein_df = protein_df.drop(['chain_rid'], axis=1)
        return protein_df

    def get_ligu(self, ligand_mol):
        mol_ligand_conf = ligand_mol.GetConformers()[0]
        pos = mol_ligand_conf.GetPositions()
        df = pd.DataFrame(pos)
        df.columns = ["x_coord", "y_coord", "z_coord"]
        return df

    def cal_pro_min_dist(self, protein_biop, ligand_mol):
        protein_biop = self.add_xyz(protein_biop)
        ligu = self.get_ligu(ligand_mol)
        ligu = self.add_xyz(ligu)
        protein_biop['min_dist'] = protein_biop.apply(
            lambda row: self.get_min_dist(row['xyz'], ligu), axis=1)
        return protein_biop

    def conc(self, a, b, c):
        return [a, b, c]

    def add_xyz(self, ligu):
        ligu['xyz'] = ligu.apply(
            lambda row: self.conc(
                row['x_coord'],
                row['y_coord'],
                row['z_coord']),
            axis=1)
        return ligu

    def cal_dist(self, a, b):
        a1 = np.array(a)
        b1 = np.array(b)
        dist = np.linalg.norm(a1 - b1)
        dist = round(dist, 2)
        return dist

    def get_min_dist(self, am, ligu):
        ligu['pro_xyz'] = [am] * ligu.shape[0]
        ligu['dist'] = ligu.apply(
            lambda row: self.cal_dist(
                row['xyz'], row['pro_xyz']), axis=1)
        md = min(ligu['dist'])
        return md


if __name__ == "__main__":
    import sys
    protein_file = sys.argv[1]
    ligand_file = sys.argv[2]
    cut = sys.argv[3]
    get_pocket = GetPocket(ligand_file, protein_file, cut)
