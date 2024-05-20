from pathlib import Path
import sys
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb

pd.options.mode.chained_assignment = None
import warnings

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


def extract_pocket(pocket_definition, protein_file: Path):
	"""
    Extracts the pocket from a protein file using provided pocket definition.

    Args:
        pocket_definition (dict): A dictionary containing the center coordinates and size of the pocket.
            The dictionary has the following structure:
            {
                "center": [center_x, center_y, center_z],
                "size": [size_x, size_y, size_z],
            }
        protein_file (Path): The path to the protein file in pdb format.

    Returns:
        dict: A dictionary containing the coordinates and size of the extracted pocket.
            The dictionary has the following structure:
            {
                "center": [center_x, center_y, center_z],
                "size": [size_x, size_y, size_z]
            }
    """
	center_x, center_y, center_z = pocket_definition["center"]
	size_x, size_y, size_z = pocket_definition["size"]
	radius = size_x             # Assuming the size is twice the radius

	print(f"Extracting pocket from {protein_file.stem} using provided pocket definition: {pocket_definition}")

	output_file = protein_file.with_name(protein_file.stem + "_pocket.pdb")

	if not output_file.exists():
		process_protein(protein_file, pocket_definition["center"], radius, output_file)
		print(
			f"Finished extracting pocket from {protein_file.stem} using provided pocket definition: {pocket_definition}"
		)
	else:
		printlog(
			f"Pocket already extracted from {protein_file.stem} using provided pocket definition: {pocket_definition}")

	return output_file


def process_protein(protein_file, center_coordinates, cutoff, output_file):
	"""
    Process the protein to select cutoff residues around the given coordinates and generate a pocket file.

    Args:
        protein_file (str): Path to the protein file in PDB format.
        center_coordinates (tuple): The x, y, z coordinates of the pocket center.
        cutoff (float): Cutoff distance for selecting residues near the center coordinates.
        output_file (Path): Path to the output pocket file.
    """
	center_x, center_y, center_z = center_coordinates
	# Read the protein file using PandasPdb
	ppdb = PandasPdb()
	ppdb.read_pdb(str(protein_file))
	protein_dataframe = ppdb.df["ATOM"]
	# Select residues within the cutoff distance from the center coordinates
	protein_cut = select_cutoff_residues(protein_dataframe, center_coordinates, cutoff)
	# Update the protein dataframe with the selected residues
	ppdb.df["ATOM"] = protein_cut
	# Save the updated protein dataframe as a pocket file
	ppdb.to_pdb(path=output_file, records=["ATOM"])


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
	protein_dataframe["distance"] = protein_dataframe.apply(lambda row: calculate_distance([
		row["x_coord"], row["y_coord"], row["z_coord"]], [center_x, center_y, center_z]),
															axis=1)

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
