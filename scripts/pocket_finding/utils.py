import pandas as pd


def add_coordinates(dataframe):
	"""
    Add coordinates column to the given dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The dataframe with the coordinates column added.
    """
	dataframe["coordinates"] = dataframe.apply(lambda row: [row["x_coord"], row["y_coord"], row["z_coord"]], axis=1)
	return dataframe


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
