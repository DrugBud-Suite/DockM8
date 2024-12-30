import sys
from pathlib import Path

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb

pd.options.mode.chained_assignment = None
import warnings

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

from pathlib import Path


def extract_pocket(pocket_definition, protein_file: Path):
    """
    Extracts a pocket from a protein file using the provided pocket definition.

    Args:
            pocket_definition (dict): A dictionary containing the pocket definition, including the center and size.
            protein_file (Path): The path to the protein file.

    Returns:
            Path: The path to the output file containing the extracted pocket in PDB format.
                      Returns None if the extraction fails or the pocket is empty.
    """
    center_x, center_y, center_z = pocket_definition["center"]
    size_x, size_y, size_z = pocket_definition["size"]
    radius = size_x // 2

    printlog(f"Extracting pocket from {protein_file.stem} using provided pocket definition: {pocket_definition}")

    output_file = protein_file.with_name(protein_file.stem + "_pocket.pdb")

    if not output_file.exists():
        success = process_protein(protein_file, pocket_definition["center"], radius, output_file)
        if not success:
            printlog(f"Failed to extract pocket from {protein_file.stem}. The pocket might be empty.")
            return None
        printlog(
            f"Finished extracting pocket from {protein_file.stem} using provided pocket definition: {pocket_definition}"
        )
    else:
        printlog(
            f"Pocket already extracted from {protein_file.stem} using provided pocket definition: {pocket_definition}"
        )

    return output_file


def process_protein(protein_file, center_coordinates, cutoff, output_file):
    """
    Process the protein file to extract a pocket based on the specified center coordinates and cutoff distance.

    Args:
            protein_file (str): The path to the protein file.
            center_coordinates (tuple): The coordinates of the center of the pocket.
            cutoff (float): The cutoff distance for selecting residues within the pocket.
            output_file (str): The path to save the extracted pocket.

    Returns:
            bool: True if the pocket extraction is successful, False otherwise.
    """
    ppdb = PandasPdb()
    ppdb.read_pdb(str(protein_file))
    protein_dataframe = ppdb.df["ATOM"]
    protein_cut = select_cutoff_residues(protein_dataframe, center_coordinates, cutoff)

    if protein_cut.empty:
        printlog(f"Warning: No residues found within {cutoff} Å of the specified center. The pocket might be empty.")
        return False

    ppdb.df["ATOM"] = protein_cut
    ppdb.to_pdb(path=output_file, records=["ATOM"])
    return True


def select_cutoff_residues(protein_dataframe, center_coordinates, cutoff):
    """
    Selects residues within a specified cutoff distance from given coordinates in a protein dataframe.

    Args:
        protein_dataframe (pandas.DataFrame): The protein dataframe containing information about the protein residues.
        center_coordinates (tuple): The x, y, z coordinates of the pocket center.
        cutoff (float): The cutoff distance for selecting residues.

    Returns:
        pandas.DataFrame: The updated protein dataframe with selected residues.
    """
    center_x, center_y, center_z = center_coordinates
    # Calculate the distance from each residue to the center coordinates
    protein_dataframe["distance"] = protein_dataframe.apply(
        lambda row: calculate_distance(
            [row["x_coord"], row["y_coord"], row["z_coord"]], [center_x, center_y, center_z]
        ),
        axis=1,
    )

    # Select residues within the cutoff distance
    residues_within_cutoff = protein_dataframe[protein_dataframe["distance"] < cutoff]
    return residues_within_cutoff


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
