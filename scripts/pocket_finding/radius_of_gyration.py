import os
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors3D

cwd = Path.cwd()
dockm8_path = next((path for path in cwd.parents if path.name == "DockM8"), None)
sys.path.append(str(dockm8_path))

from scripts.pocket_finding.utils import (
    get_ligand_coordinates,
    process_protein_and_ligand,
)
from scripts.utilities import load_molecule, printlog


def find_pocket_RoG(ligand_file: Path, protein_file: Path):
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
    printlog(
        f"Extracting pocket from {protein_file.stem} using {ligand_file.stem} as reference ligand"
    )
    # Load the reference ligand molecule and calculate its radius of gyration
    ligand_mol = load_molecule(str(ligand_file))
    radius_of_gyration = Descriptors3D.RadiusOfGyration(ligand_mol)
    if not os.path.exists(str(protein_file).replace(".pdb", "_pocket.pdb")):
        printlog(f"Radius of Gyration of reference ligand is: {radius_of_gyration}")
        # Process the protein and ligand to extract the pocket
        pocket_mol, temp_file = process_protein_and_ligand(
            str(protein_file),
            ligand_mol,
            round(0.5 * 2.857 * float(radius_of_gyration), 2),
        )
        pocket_path = str(protein_file).replace(".pdb", "_pocket.pdb")
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
        "size": [
            round(2.857 * float(radius_of_gyration), 2),
            round(2.857 * float(radius_of_gyration), 2),
            round(2.857 * float(radius_of_gyration), 2),
        ],
    }
    return pocket_coordinates