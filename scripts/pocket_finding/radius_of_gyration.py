import sys
from pathlib import Path

from rdkit.Chem import Descriptors3D

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts"
                     for p in Path(__file__).resolve().parents
                     if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.pocket_finding.utils import get_ligand_coordinates
from scripts.utilities.utilities import load_molecule, printlog


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
            round(2.857 * float(radius_of_gyration), 2),],}
    return pocket_coordinates
