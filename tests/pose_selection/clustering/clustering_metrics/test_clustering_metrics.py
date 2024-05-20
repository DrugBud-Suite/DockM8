from pathlib import Path
import sys

from rdkit import Chem
from rdkit.Chem import AllChem


# Search for 'DockM8' in parent directories
tests_path = next((p / "tests"
                   for p in Path(__file__).resolve().parents
                   if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.pose_selection.clustering.clustering_metrics.clustering_metrics import espsim_calc
from scripts.pose_selection.clustering.clustering_metrics.clustering_metrics import simpleRMSD_calc
from scripts.pose_selection.clustering.clustering_metrics.clustering_metrics import spyRMSD_calc
from scripts.pose_selection.clustering.clustering_metrics.clustering_metrics import USRCAT_calc


def test_simpleRMSD_calc():
    mol = Chem.MolFromSmiles('O=C(NC1=CC=CC=C1Cl)C2=CC=CC=C2OC')

    # Generate conformers
    num_conformers = 2
    AllChem.EmbedMultipleConfs(mol, num_conformers)

    # Access the conformers
    conformers = mol.GetConformers()

    # Separate the conformers into two molecules
    molecules = []
    for conformer in conformers:
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conformer)
        molecules.append(new_mol)

    rmsd_value = simpleRMSD_calc(molecules[0], molecules[1])
    assert isinstance(rmsd_value, float)
    assert 0 <= rmsd_value <= 100
    assert rmsd_value is not None
    
def test_spyRMSD_calc():
    mol = Chem.MolFromSmiles('O=C(NC1=CC=CC=C1Cl)C2=CC=CC=C2OC')

    # Generate conformers
    num_conformers = 2
    AllChem.EmbedMultipleConfs(mol, num_conformers)

    # Access the conformers
    conformers = mol.GetConformers()

    # Separate the conformers into two molecules
    molecules = []
    for conformer in conformers:
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conformer)
        molecules.append(new_mol)

    rmsd_value = spyRMSD_calc(molecules[0], molecules[1])
    assert isinstance(rmsd_value, float)
    assert 0 <= rmsd_value <= 100
    assert rmsd_value is not None
    
def test_espsim_calc():
    mol = Chem.MolFromSmiles('O=C(NC1=CC=CC=C1Cl)C2=CC=CC=C2OC')

    # Generate conformers
    num_conformers = 2
    AllChem.EmbedMultipleConfs(mol, num_conformers)

    # Access the conformers
    conformers = mol.GetConformers()

    # Separate the conformers into two molecules
    molecules = []
    for conformer in conformers:
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conformer)
        molecules.append(new_mol)

    espsim_value = espsim_calc(molecules[0], molecules[1])
    assert isinstance(espsim_value, float)
    assert 0 <= espsim_value <= 100
    assert espsim_value is not None
    
def test_USRCAT_calc():
    mol = Chem.MolFromSmiles('O=C(NC1=CC=CC=C1Cl)C2=CC=CC=C2OC')

    # Generate conformers
    num_conformers = 2
    AllChem.EmbedMultipleConfs(mol, num_conformers)

    # Access the conformers
    conformers = mol.GetConformers()

    # Separate the conformers into two molecules
    molecules = []
    for conformer in conformers:
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conformer)
        molecules.append(new_mol)

    USRCAT_value = USRCAT_calc(molecules[0], molecules[1])
    assert isinstance(USRCAT_value, float)
    assert 0 <= USRCAT_value <= 100
    assert USRCAT_value is not None