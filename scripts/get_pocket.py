import os
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import Descriptors3D

from scripts.utilities import load_molecule, printlog

pd.options.mode.chained_assignment = None
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_pocket(ligand_file: Path, protein_file: Path, radius: int):
    """
    Extracts the pocket from a protein file using a reference ligand.

    Args:
        ligand_file (Path): The path to the reference ligand file in mol format.
        protein_file (Path): The path to the protein file in pdb format.
        radius (int): The radius of the pocket to be extracted.

    Returns:
        dict: A dictionary containing the coordinates and size of the extracted pocket.
            The dictionary has the following structure:
            {
                "center": [center_x, center_y, center_z],
                "size": [size_x, size_y, size_z]
            }
    """
    printlog(f'Extracting pocket from {protein_file.stem} using {ligand_file.stem} as reference ligand')
    # Load the reference ligand molecule
    ligand_mol = load_molecule(str(ligand_file))
    if not os.path.exists(str(protein_file).replace('.pdb', '_pocket.pdb')):
        # Process the protein and ligand to extract the pocket
        pocket_mol, temp_file = process_protein_and_ligand(str(protein_file), ligand_mol, radius)
        pocket_path = str(protein_file).replace('.pdb', '_pocket.pdb')
        # Convert the pocket molecule to PDB file format and save it
        Chem.MolToPDBFile(pocket_mol, pocket_path)
        os.remove(temp_file)
        printlog(f'Finished extracting pocket from {protein_file.stem} using {ligand_file.stem} as reference ligand')
    else:
        pass
    # Calculate the center coordinates of the pocket
    ligu = get_ligand_coordinates(ligand_mol)
    center_x = ligu['x_coord'].mean().round(2)
    center_y = ligu['y_coord'].mean().round(2)
    center_z = ligu['z_coord'].mean().round(2)
    # Create a dictionary with the pocket coordinates and size
    pocket_coordinates = {
        "center": [center_x, center_y, center_z],
        "size": [float(radius) * 2, float(radius) * 2, float(radius) * 2]
    }
    return pocket_coordinates

def get_pocket_RoG(ligand_file : Path, protein_file : Path):
    """
    Extracts the pocket from a protein using a reference ligand and calculates the radius of gyration.

    Args:
        ligand_file (Path): The path to the reference ligand file in mol format.
        protein_file (Path): The path to the protein file in pdb format.

    Returns:
        dict: A dictionary containing the pocket coordinates and size.
            The dictionary has the following keys:
                - 'center': A list of three floats representing the x, y, and z coordinates of the pocket center.
                - 'size': A list of three floats representing the size of the pocket in each dimension.
    """
    printlog(f'Extracting pocket from {protein_file.stem} using {ligand_file.stem} as reference ligand')
    # Load the reference ligand molecule and calculate its radius of gyration
    ligand_mol = load_molecule(str(ligand_file))
    radius_of_gyration = Descriptors3D.RadiusOfGyration(ligand_mol)
    if not os.path.exists(str(protein_file).replace('.pdb', '_pocket.pdb')):
        printlog(f'Radius of Gyration of reference ligand is: {radius_of_gyration}')
        # Process the protein and ligand to extract the pocket
        pocket_mol, temp_file = process_protein_and_ligand(str(protein_file), ligand_mol, round(0.5 * 2.857 * float(radius_of_gyration), 2))
        pocket_path = str(protein_file).replace('.pdb', '_pocket.pdb')
        Chem.MolToPDBFile(pocket_mol, pocket_path)
        os.remove(temp_file)
        printlog(f'Finished extracting pocket from {protein_file.stem} using {ligand_file.stem} as reference ligand')
    else:
        pass
    # Calculate the center coordinates of the pocket
    ligu = get_ligand_coordinates(ligand_mol)
    center_x = ligu['x_coord'].mean().round(2)
    center_y = ligu['y_coord'].mean().round(2)
    center_z = ligu['z_coord'].mean().round(2)
    # Create a dictionary with the pocket coordinates and size
    pocket_coordinates = {
        "center": [center_x,center_y,center_z],
        "size": [round(2.857 * float(radius_of_gyration), 2), round(2.857 * float(radius_of_gyration), 2), round(2.857 * float(radius_of_gyration), 2)]}
    return pocket_coordinates

def process_protein_and_ligand(protein_file, ligand_molecule, cutoff):
    """
    Process the protein and ligand to select cutoff residues and generate a pocket file.

    Args:
        protein_file (str): Path to the protein file in PDB format.
        ligand_molecule (Chem.Mol): Ligand molecule.
        cutoff (float): Cutoff distance for selecting residues near the ligand.

    Returns:
        protein_molecule (Chem.Mol): Processed protein molecule with selected residues.
        pocket_file_name (str): Name of the generated pocket file.
    """
    # Read the protein file using PandasPdb
    ppdb = PandasPdb()
    ppdb.read_pdb(protein_file)
    protein_dataframe = ppdb.df['ATOM']
    # Select cutoff residues near the ligand
    protein_cut, residues_near_ligand = select_cutoff_residues(protein_dataframe, ligand_molecule, cutoff)
    # Update the protein dataframe with the selected residues
    ppdb.df['ATOM'] = protein_cut
    # Generate a random name for the pocket file
    random_mol_name = str(randint(1, 1000000)).zfill(10)
    pocket_file_name = 'pocket_{}.pdb'.format(random_mol_name)
    # Save the updated protein dataframe as a pocket file
    ppdb.to_pdb(path=pocket_file_name, records=['ATOM'])
    # Create a molecule object from the pocket file
    protein_molecule = Chem.MolFromPDBFile(pocket_file_name, removeHs=False)
    # Return the processed protein molecule and the pocket file name
    return protein_molecule, pocket_file_name


def add_coordinates(dataframe):
    """
    Add coordinates column to the given dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The dataframe with the coordinates column added.
    """
    dataframe['coordinates'] = dataframe.apply(lambda row: [row['x_coord'], row['y_coord'], row['z_coord']],axis=1)
    return dataframe

def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    point1 (list or tuple): The coordinates of the first point.
    point2 (list or tuple): The coordinates of the second point.

    Returns:
    float: The distance between the two points, rounded to 2 decimal places.
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    return round(np.linalg.norm(point1 - point2), 2)

def calculate_min_distance(aminoacid, dataframe):
    """
    Calculate the minimum distance between the given amino acid and the coordinates in the dataframe.

    Parameters:
    aminoacid (str): The amino acid to calculate the distance from.
    dataframe (pandas.DataFrame): The dataframe containing the coordinates.

    Returns:
    float: The minimum distance between the amino acid and the coordinates.
    """
    dataframe['protein_coordinates'] = [aminoacid] * dataframe.shape[0]
    dataframe['distance'] = dataframe.apply(
        lambda row: calculate_distance(
            row['coordinates'],
            row['protein_coordinates']),
        axis=1)
    return min(dataframe['distance'])


def get_ligand_coordinates(ligand_molecule):
    """
    Get the coordinates of a ligand molecule.

    Parameters:
    ligand_molecule (Molecule): The ligand molecule.

    Returns:
    DataFrame: A DataFrame containing the x, y, and z coordinates of the ligand molecule.
    """
    ligand_conformer = ligand_molecule.GetConformers()[0]
    coordinates = ligand_conformer.GetPositions()
    dataframe = pd.DataFrame(coordinates, columns=["x_coord", "y_coord", "z_coord"])
    return add_coordinates(dataframe)


def calculate_min_distance_protein(protein_dataframe, ligand_molecule):
    """
    Calculates the minimum distance between each protein coordinate and the ligand molecule.

    Args:
        protein_dataframe (pandas.DataFrame): DataFrame containing protein coordinates.
        ligand_molecule (Molecule): Ligand molecule object.

    Returns:
        pandas.DataFrame: DataFrame with an additional column 'min_dist' containing the minimum distance for each protein coordinate.
    """
    protein_dataframe = add_coordinates(protein_dataframe)
    ligand_coordinates = add_coordinates(get_ligand_coordinates(ligand_molecule))
    protein_dataframe['min_dist'] = protein_dataframe.apply(lambda row: calculate_min_distance(row['coordinates'], ligand_coordinates), axis=1)
    return protein_dataframe

def select_cutoff_residues(protein_dataframe, ligand_molecule, cutoff):
    """
    Selects residues within a specified cutoff distance from a ligand molecule in a protein dataframe.

    Args:
        protein_dataframe (pandas.DataFrame): The protein dataframe containing information about the protein residues.
        ligand_molecule: The ligand molecule.
        cutoff (float): The cutoff distance for selecting residues.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame]: A tuple containing the updated protein dataframe and a dataframe
        containing the residues within the cutoff distance.
    """
    # Calculate the minimum distance between each protein coordinate and the ligand molecule
    protein_dataframe = calculate_min_distance_protein(protein_dataframe, ligand_molecule)
    # Create a new column 'chain_residue_id' by concatenating chain_id and residue_number
    protein_dataframe['chain_residue_id'] = protein_dataframe.apply(lambda row: str(row['chain_id']) + str(row['residue_number']), axis=1)
    # Select residues within the cutoff distance
    residues_within_cutoff = protein_dataframe[protein_dataframe['min_dist'] < cutoff]
    # Get the unique selected residues
    selected_residues = list(set(list(residues_within_cutoff['chain_residue_id'])))
    # Filter the protein dataframe to include only the selected residues
    protein_dataframe = protein_dataframe[protein_dataframe['chain_residue_id'].isin(selected_residues)]
    # Drop the 'chain_residue_id' column
    protein_dataframe = protein_dataframe.drop(['chain_residue_id'], axis=1)
    return protein_dataframe, residues_within_cutoff
