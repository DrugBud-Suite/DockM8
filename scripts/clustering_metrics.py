from spyrmsd import rmsd, molecule
from espsim import GetEspSim
import oddt
import oddt.shape
import oddt.fingerprints
import oddt.toolkits.rdk
from rdkit import Chem
from rdkit.Chem import rdFMCS
import pandas as pd
import numpy as np
import math


def simpleRMSD_calc(*args):
    '''
    Calculates the root mean square deviation (RMSD) metric between two molecules.

    Args:
        *args: Variable number of arguments representing the two molecules for which the RMSD is calculated.

    Returns:
        float: The calculated RMSD value between the two molecules.
    '''
    # Find the maximum common substructure (MCS) between the reference and target molecules
    mcs = rdFMCS.FindMCS([args[0], args[1]])
    
    # Get the atom map for the reference and target molecules
    ref_atoms = args[0].GetSubstructMatch(Chem.MolFromSmarts(mcs.smartsString))
    target_atoms = args[1].GetSubstructMatch(Chem.MolFromSmarts(mcs.smartsString))
    
    # Generate the atom map by zipping the atom indices from the reference and target molecules
    atom_map = list(zip(ref_atoms, target_atoms))
    
    # Calculate the distances between corresponding atoms in the two molecules
    distances = [np.linalg.norm(np.array(args[0].GetConformer().GetAtomPosition(ref)) - np.array(args[1].GetConformer().GetAtomPosition(target))) for ref, target in atom_map]
    
    # Apply the RMSD formula
    rmsd = np.sqrt(np.mean(np.square(distances)))
    
    return round(rmsd, 3)


def spyRMSD_calc(*args):
    """
    Calculates the symmetry-corrected RMSD metric between two molecules.

    Args:
        *args: Variable number of arguments representing the two molecules for which the RMSD is calculated.

    Returns:
        float: The calculated symmetry-corrected RMSD value between the two molecules, rounded to 3 decimal places.
    """
    mol = args[0][0] if isinstance(args[0], tuple) else args[0]
    jmol = args[0][1] if isinstance(args[0], tuple) else args[1]

    spyrmsd_mol = molecule.Molecule.from_rdkit(mol)
    spyrmsd_jmol = molecule.Molecule.from_rdkit(jmol)

    spyrmsd_mol.strip()
    spyrmsd_jmol.strip()

    coords_ref = spyrmsd_mol.coordinates
    anum_ref = spyrmsd_mol.atomicnums
    adj_ref = spyrmsd_mol.adjacency_matrix

    coords_test = spyrmsd_jmol.coordinates
    anum_test = spyrmsd_jmol.atomicnums
    adj_test = spyrmsd_jmol.adjacency_matrix

    spyRMSD = rmsd.symmrmsd(coords_ref, coords_test, anum_ref, anum_test, adj_ref, adj_test)

    return round(spyRMSD, 3)


def espsim_calc(*args):
    '''Calculates the electrostatic shape similarity metric between two molecules'''
    return GetEspSim(args[0], args[1])


def SPLIF_calc(molecule1: str, molecule2: str, pocket_file: str) -> float:
    '''
    Calculates the Protein-Ligand Interaction fingerprint similarity metric between two molecules

    Args:
        molecule1 (str): Path to the first molecule file
        molecule2 (str): Path to the second molecule file
        pocket_file (str): Path to the pocket file

    Returns:
        float: Rounded similarity score between the two molecules
    '''
    # Generate the path to the pocket file
    pocket_file = pocket_file.replace('.pdb', '_pocket.pdb')

    # Read the protein structure from the pocket file
    protein = next(oddt.toolkit.readfile('pdb', pocket_file))
    protein.protein = True

    # Create Molecule objects for the two molecules
    splif_mol = oddt.toolkits.rdk.Molecule(molecule1)
    splif_jmol = oddt.toolkits.rdk.Molecule(molecule2)

    # Calculate the Simple Interaction Fingerprint (SIF) for the two molecules
    mol_fp = oddt.fingerprints.SimpleInteractionFingerprint(splif_mol, protein)
    jmol_fp = oddt.fingerprints.SimpleInteractionFingerprint(splif_jmol, protein)

    # Calculate the Tanimoto similarity between the two SIFs
    SPLIF_sim = oddt.fingerprints.tanimoto(mol_fp, jmol_fp)

    # Round the similarity score to 3 decimal places
    return round(SPLIF_sim, 3)


def USRCAT_calc(*args):
    """
    Calculates the shape similarity metric between two molecules using the USR-CAT method.

    Args:
        *args: Variable number of arguments representing the two molecule files for which the shape similarity is calculated.

    Returns:
        float: The shape similarity score between the two molecules, rounded to 3 decimal places.
    """
    # Create Molecule objects for the two molecules
    shape_mol = oddt.toolkits.rdk.Molecule(args[0])
    shape_jmol = oddt.toolkits.rdk.Molecule(args[1])

    # Calculate the USR-CAT fingerprint for the two molecules
    mol_fp = oddt.shape.usr_cat(shape_mol)
    jmol_fp = oddt.shape.usr_cat(shape_jmol)

    # Calculate the USR-CAT similarity between the two fingerprints
    usr_sim = oddt.shape.usr_similarity(mol_fp, jmol_fp)

    # Round the similarity score to 3 decimal places
    return round(usr_sim, 3)
