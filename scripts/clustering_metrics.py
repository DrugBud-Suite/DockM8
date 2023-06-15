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
    '''Calculates the RMSD metric between two molecules'''
    # MCS identification between reference pose and target pose
    r = rdFMCS.FindMCS([args[0], args[1]])
    # Atom map for reference and target
    a = args[0].GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
    b = args[1].GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
    # Atom map generation
    amap = list(zip(a, b))
    # distance calculation per atom pair
    distances = []
    for atomA, atomB in amap:
        pos_A = args[0].GetConformer().GetAtomPosition(atomA)
        pos_B = args[1].GetConformer().GetAtomPosition(atomB)
        coord_A = np.array((pos_A.x, pos_A.y, pos_A.z))
        coord_B = np.array((pos_B.x, pos_B.y, pos_B.z))
        dist_numpy = np.linalg.norm(coord_A - coord_B)
        distances.append(dist_numpy)
    # This is the RMSD formula from wikipedia
    rmsd = math.sqrt(1 / len(distances) * sum([i * i for i in distances]))
    return round(rmsd, 3)


def spyRMSD_calc(*args):
    '''Calculates the symmetry-corrected RMSD metric between two molecules'''
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
    spyRMSD = rmsd.symmrmsd(
        coords_ref,
        coords_test,
        anum_ref,
        anum_test,
        adj_ref,
        adj_test)
    return round(spyRMSD, 3)


def espsim_calc(*args):
    '''Calculates the electrostatic shape similarity metric between two molecules'''
    return GetEspSim(args[0], args[1])


def SPLIF_calc(*args):
    '''Calculates the Protein-Ligand Interaction fingerprint similarity metric between two molecules'''
    pocket_file = args[2].replace('.pdb', '_pocket.pdb')
    protein = next(oddt.toolkit.readfile('pdb', pocket_file))
    protein.protein = True
    splif_mol = oddt.toolkits.rdk.Molecule(args[0])
    splif_jmol = oddt.toolkits.rdk.Molecule(args[1])
    mol_fp = oddt.fingerprints.SimpleInteractionFingerprint(splif_mol, protein)
    jmol_fp = oddt.fingerprints.SimpleInteractionFingerprint(
        splif_jmol, protein)
    SPLIF_sim = oddt.fingerprints.tanimoto(mol_fp, jmol_fp)
    return round(SPLIF_sim, 3)


def USRCAT_calc(*args):
    '''Calculates the Shape similarity metric between two molecules'''
    shape_mol = oddt.toolkits.rdk.Molecule(args[0])
    shape_jmol = oddt.toolkits.rdk.Molecule(args[1])
    mol_fp = oddt.shape.usr_cat(shape_mol)
    jmol_fp = oddt.shape.usr_cat(shape_jmol)
    usr_sim = oddt.shape.usr_similarity(mol_fp, jmol_fp)
    return round(usr_sim, 3)

# NOT WORKING!


def symmRMSD_calc(*args):
    rms = Chem.rdMolAlign.CalcRMS(args[0], args[1])
    return round(rms, 3)
