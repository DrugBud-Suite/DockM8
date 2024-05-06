import os
import sys
from pathlib import Path

from rdkit import Chem

# Search for 'DockM8' in parent directories
dockm8_path = next((p / 'DockM8' for p in Path(__file__).resolve().parents if (p / 'DockM8').is_dir()), None)
sys.path.append(str(dockm8_path))

from scripts.pocket_finding.utils import (
    get_ligand_coordinates,
    process_protein_and_ligand,
)
from scripts.utilities.utilities import load_molecule, printlog


def find_pocket_default(ligand_file: Path, protein_file: Path, radius: int):
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
    printlog(
        f"Extracting pocket from {protein_file.stem} using {ligand_file.stem} as reference ligand"
    )
    # Load the reference ligand molecule
    ligand_mol = load_molecule(str(ligand_file))
    if not os.path.exists(str(protein_file).replace(".pdb", "_pocket.pdb")):
        # Process the protein and ligand to extract the pocket
        pocket_mol, temp_file = process_protein_and_ligand(
            str(protein_file), ligand_mol, radius
        )
        pocket_path = str(protein_file).replace(".pdb", "_pocket.pdb")
        # Convert the pocket molecule to PDB file format and save it
        Chem.MolToPDBFile(pocket_mol, pocket_path)
        os.remove(temp_file)
        printlog(
            f"Finished extracting pocket from {protein_file.stem} using {ligand_file.stem} as reference ligand"
        )
    else:
        pass
    # Calculate the center coordinates of the pocket
    ligu = get_ligand_coordinates(ligand_mol)
    center_x = ligu["x_coord"].mean().round(2)
    center_y = ligu["y_coord"].mean().round(2)
    center_z = ligu["z_coord"].mean().round(2)
    # Create a dictionary with the pocket coordinates and size
    pocket_coordinates = {
        "center": [center_x, center_y, center_z],
        "size": [float(radius) * 2, float(radius) * 2, float(radius) * 2],
    }
    return pocket_coordinates