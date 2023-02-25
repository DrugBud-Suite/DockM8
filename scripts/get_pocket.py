from random import randint
import pandas as pd
import os,glob
import numpy as np
import copy
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import AllChem
pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter(action='ignore',category=UserWarning)

def conc(a,b,c):
    return [a,b,c]

def add_xyz(ligu):
    ligu['xyz'] = ligu.apply(lambda row: conc(row['x_coord'],row['y_coord'],row['z_coord']),axis=1)
    return ligu

def cal_dist(a,b):
    a1 = np.array(a)
    b1 = np.array(b)
    dist = np.linalg.norm(a1-b1)
    dist =round(dist,2)
    return dist

def get_min_dist(am, ligu):
    ligu['pro_xyz']=[am]*ligu.shape[0]
    ligu['dist']= ligu.apply(lambda row: cal_dist(row['xyz'], row['pro_xyz']), axis =1)
    md = min(ligu['dist'])
    return md

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
        return ValueError(f'Expect the format of the molecule_file to be '
                          'one of .mol2, .mol, .sdf, .pdbqt and .pdb, got {molecule_file}')
    return mol

class GetPocket:
    def __init__(self, ligand_file, protein_file, cut):
        """
        ligand_file: format mol
        protein_file: format pdb
        """
        print(f'Extracting pocket from {protein_file} using {ligand_file} as reference ligand')
        self.ligand_file = ligand_file
        self.protein_file = protein_file
        self.ligand_mol = load_molecule(ligand_file)
        self.pocket_mol,  self.temp_file = self.process_pro_and_lig(cut)
        self.pocket_path = protein_file.replace('.pdb', '_pocket.pdb')
        Chem.MolToPDBFile(self.pocket_mol, self.pocket_path)
        os.remove(self.temp_file)
        print(f'Finished extracting pocket from {protein_file} using {ligand_file} as reference ligand')
        
    def process_pro_and_lig(self, cut):
        ppdb = PandasPdb()
        ppdb.read_pdb(self.protein_file)
        protein_biop = ppdb.df['ATOM']
        pro_cut, pros_near_lig = self.select_cut_residue(protein_biop, self.ligand_mol, cut)
        ppdb.df['ATOM'] = pro_cut
        newmolname = str(randint(1,1000000)).zfill(10)
        name = 'pocket_{}.pdb'.format(newmolname)
        ppdb.to_pdb(path=name, records=['ATOM'])
        pmol = Chem.MolFromPDBFile(name, removeHs=False)
        return pmol, name
    
    def select_cut_residue(self, protein_biop, ligand_mol, cut):
        """
        pro: biopandas DataFrame
        lig: rdkit mol
        """
        pro = self.cal_pro_min_dist(protein_biop, ligand_mol)
        pro['chain_rid'] = pro.apply(lambda row: 
                                     str(row['chain_id'])+str(row['residue_number']), axis=1)
        pros = pro[pro['min_dist'] < cut]
        pros_near_lig = copy.deepcopy(pros)
        use_res = list(set(list(pros['chain_rid'])))
        pro= pro[pro['chain_rid'].isin(use_res)]
        pro = pro.drop(['chain_rid'],axis=1)
        return pro, pros_near_lig
    
    def get_ligu(self, ligand_mol):
        mol_ligand_conf = ligand_mol.GetConformers()[0]
        pos = mol_ligand_conf.GetPositions()
        df = pd.DataFrame(pos)
        df.columns = ["x_coord", "y_coord","z_coord"]
        return df
    
    def cal_pro_min_dist(self, protein_biop, ligand_mol):
        protein_biop =add_xyz(protein_biop)
        ligu = self.get_ligu(ligand_mol)
        ligu = add_xyz(ligu)
        protein_biop['min_dist']= protein_biop.apply(lambda row: get_min_dist(row['xyz'], ligu), axis=1)
        return protein_biop


if __name__=="__main__":
    import sys
    protein_file=sys.argv[1]
    ligand_file=sys.argv[2]
    cut=sys.argv[3]
    get_pocket = GetPocket(ligand_file, protein_file, cut)
