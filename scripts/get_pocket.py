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
from scripts.utilities import *
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter(action='ignore',category=UserWarning)

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

def get_pocket(ligand_file, protein_file, radius):
    """
    ligand_file: format mol
    protein_file: format pdb
    """
    printlog(f'Extracting pocket from {protein_file} using {ligand_file} as reference ligand')
    ligand_mol = load_molecule(ligand_file)
    pocket_mol,  temp_file = process_protein_and_ligand(protein_file, ligand_mol, radius)
    pocket_path = protein_file.replace('.pdb', '_pocket.pdb')
    Chem.MolToPDBFile(pocket_mol, pocket_path)
    os.remove(temp_file)
    printlog(f'Finished extracting pocket from {protein_file} using {ligand_file} as reference ligand')
    
    ligu = get_ligand_coordinates(ligand_mol)
    center_x = ligu['x_coord'].mean().round(2)
    center_y = ligu['y_coord'].mean().round(2)
    center_z = ligu['z_coord'].mean().round(2)
    pocket_coordinates = {
        "center": [center_x, center_y, center_z],
        "size": [float(radius)*2, float(radius)*2, float(radius)*2]}
    return pocket_coordinates

from rdkit.Chem import Descriptors3D

def get_pocket_RoG(ligand_file, protein_file):
    """
    ligand_file: format mol
    protein_file: format pdb
    """
    printlog(f'Extracting pocket from {protein_file} using {ligand_file} as reference ligand')
    ligand_mol = load_molecule(ligand_file)
    radius_of_gyration = Descriptors3D.RadiusOfGyration(ligand_mol)
    printlog(f'Radius of Gyration of reference ligand is: {radius_of_gyration}')
    pocket_mol,  temp_file = process_protein_and_ligand(protein_file, ligand_mol, round(0.5*2.857*float(radius_of_gyration), 2))
    pocket_path = protein_file.replace('.pdb', '_pocket.pdb')
    Chem.MolToPDBFile(pocket_mol, pocket_path)
    os.remove(temp_file)
    printlog(f'Finished extracting pocket from {protein_file} using {ligand_file} as reference ligand')
    
    ligu = get_ligand_coordinates(ligand_mol)
    center_x = ligu['x_coord'].mean().round(2)
    center_y = ligu['y_coord'].mean().round(2)
    center_z = ligu['z_coord'].mean().round(2)
    pocket_coordinates = {
        "center": [center_x, center_y, center_z],
        "size": [round(2.857*float(radius_of_gyration),2), round(2.857*float(radius_of_gyration),2), round(2.857*float(radius_of_gyration),2)]}
    return pocket_coordinates

def process_protein_and_ligand(protein_file, ligand_molecule, cutoff):
    ppdb = PandasPdb()
    ppdb.read_pdb(protein_file)
    protein_dataframe = ppdb.df['ATOM']
    protein_cut, residues_near_ligand = select_cutoff_residues(protein_dataframe, ligand_molecule, cutoff)
    ppdb.df['ATOM'] = protein_cut
    random_mol_name = str(randint(1, 1000000)).zfill(10)
    pocket_file_name = 'pocket_{}.pdb'.format(random_mol_name)
    ppdb.to_pdb(path=pocket_file_name, records=['ATOM'])
    protein_molecule = Chem.MolFromPDBFile(pocket_file_name, removeHs=False)
    return protein_molecule, pocket_file_name

# Function to add coordinates column in the DataFrame
def add_coordinates(dataframe):
    dataframe['coordinates'] = dataframe.apply(
        lambda row: [row['x_coord'], row['y_coord'], row['z_coord']],
        axis=1)
    return dataframe

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return round(np.linalg.norm(point1 - point2), 2)

# Function to calculate the minimum distance
def calculate_min_distance(aminoacid, dataframe):
    dataframe['protein_coordinates'] = [aminoacid] * dataframe.shape[0]
    dataframe['distance'] = dataframe.apply(
        lambda row: calculate_distance(row['coordinates'], row['protein_coordinates']), 
        axis=1)
    return min(dataframe['distance'])

# Get coordinates of a ligand molecule
def get_ligand_coordinates(ligand_molecule):
    ligand_conformer = ligand_molecule.GetConformers()[0]
    coordinates = ligand_conformer.GetPositions()
    dataframe = pd.DataFrame(coordinates, columns=["x_coord", "y_coord","z_coord"])
    return add_coordinates(dataframe)

# Calculate minimum distance for protein
def calculate_min_distance_protein(protein_dataframe, ligand_molecule):
    protein_dataframe = add_coordinates(protein_dataframe)
    ligand_coordinates = add_coordinates(get_ligand_coordinates(ligand_molecule))
    protein_dataframe['min_dist'] = protein_dataframe.apply(
        lambda row: calculate_min_distance(row['coordinates'], ligand_coordinates), 
        axis=1)
    return protein_dataframe

# Select cutoff residues
def select_cutoff_residues(protein_dataframe, ligand_molecule, cutoff):
    protein_dataframe = calculate_min_distance_protein(protein_dataframe, ligand_molecule)
    protein_dataframe['chain_residue_id'] = protein_dataframe.apply(
        lambda row: str(row['chain_id']) + str(row['residue_number']), 
        axis=1)
    residues_within_cutoff = protein_dataframe[protein_dataframe['min_dist'] < cutoff]
    selected_residues = list(set(list(residues_within_cutoff['chain_residue_id'])))
    protein_dataframe = protein_dataframe[protein_dataframe['chain_residue_id'].isin(selected_residues)]
    protein_dataframe = protein_dataframe.drop(['chain_residue_id'], axis=1)
    return protein_dataframe, residues_within_cutoff

