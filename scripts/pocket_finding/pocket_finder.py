from pathlib import Path
from typing import Dict, List

import pandas as pd
from rdkit.Chem import Descriptors3D

from scripts.pocket_finding.dogsitescorer import (
    calculate_pocket_coordinates_from_pocket_pdb_file,
    get_dogsitescorer_metadata,
    get_selected_pocket_location,
    save_binding_site_to_file,
    sort_binding_sites,
    submit_dogsitescorer_job_with_pdbid,
    upload_pdb_file,
)
from scripts.pocket_finding.utils import extract_pocket
from scripts.utilities.logging import printlog
from scripts.utilities.utilities import load_molecule

VALID_METHODS = ["Reference", "RoG", "Dogsitescorer", "Manual"]
EXPECTED_COORDINATES_FORMAT = "center:x,y,z*size:x,y,z"


def validate_pocket_string(pocket_str: str) -> None:
    """
    Validate the format of the pocket coordinates string.

    Args:
        pocket_str (str): The pocket string to validate.

    Raises:
        ValueError: If the pocket string format is invalid.
    """
    if not pocket_str or "*" not in pocket_str:
        raise ValueError(f"Invalid pocket format. Expected format: {EXPECTED_COORDINATES_FORMAT}")

    parts = pocket_str.split("*")
    if len(parts) != 2:
        raise ValueError(f"Pocket string must contain exactly one '*'. Expected format: {EXPECTED_COORDINATES_FORMAT}")

    for part in parts:
        if ":" not in part:
            raise ValueError(f"Each part must contain ':' separator. Expected format: {EXPECTED_COORDINATES_FORMAT}")

        key, coords = part.split(":")
        if key not in ["center", "size"]:
            raise ValueError(f"Invalid key '{key}'. Must be either 'center' or 'size'")

        try:
            values = coords.split(",")
            if len(values) != 3:
                raise ValueError(f"Each coordinate must have exactly 3 values. Got {len(values)} values for {key}")
            [float(v) for v in values]
        except ValueError as e:
            raise ValueError(f"Invalid coordinate values for {key}: {str(e)}")


def parse_pocket_coordinates(manual_pocket: str) -> Dict[str, List[float]]:
    """
    Parse the pocket coordinates from the given pocket argument.

    Args:
        manual_pocket (str): The pocket argument to parse.

    Returns:
        Dict[str, List[float]]: A dictionary containing the parsed pocket coordinates.

    Raises:
        ValueError: If there is an error parsing the pocket coordinates.
    """
    validate_pocket_string(manual_pocket)

    pocket_str = manual_pocket.split("*")
    pocket_coordinates = {}

    for item in pocket_str:
        key, value = item.split(":")
        coordinates = [float(x) for x in value.split(",")]
        pocket_coordinates[key] = coordinates

    return pocket_coordinates


def find_pocket_default(ligand_file: Path, protein_file: Path, radius: int) -> Dict[str, List[float]]:
    """
    Extracts the pocket from a protein file using a reference ligand.

    Args:
        ligand_file (Path): The path to the reference ligand file in mol format.
        protein_file (Path): The path to the protein file in pdb format.
        radius (int): The radius of the pocket to be extracted.

    Returns:
        Dict[str, List[float]]: A dictionary containing the coordinates and size of the extracted pocket.
    """
    printlog(f"Extracting pocket from {protein_file.stem} using {ligand_file.stem} as reference ligand")
    ligand_mol = load_molecule(str(ligand_file))

    # Get ligand coordinates
    ligand_conformer = ligand_mol.GetConformers()[0]
    coordinates = ligand_conformer.GetPositions()
    ligand_coords = pd.DataFrame(coordinates, columns=["x_coord", "y_coord", "z_coord"])

    # Calculate center
    center_x = ligand_coords["x_coord"].mean().round(2)
    center_y = ligand_coords["y_coord"].mean().round(2)
    center_z = ligand_coords["z_coord"].mean().round(2)

    pocket_coordinates = {"center": [center_x, center_y, center_z], "size": [float(radius) * 2] * 3}
    return pocket_coordinates


def find_pocket_rog(ligand_file: Path, protein_file: Path, radius: int) -> Dict[str, List[float]]:
    """
    Extracts the pocket from a protein using a reference ligand and calculates the radius of gyration.

    Args:
        ligand_file (Path): The path to the reference ligand file in mol format.
        protein_file (Path): The path to the protein file in pdb format.
        radius (int): Not used in this method, kept for consistency with other methods.

    Returns:
        Dict[str, List[float]]: A dictionary containing the pocket coordinates and size.
    """
    printlog(f"Extracting pocket from {protein_file.stem} using {ligand_file.stem} as reference ligand")
    ligand_mol = load_molecule(str(ligand_file))

    # Calculate radius of gyration
    radius_of_gyration = Descriptors3D.RadiusOfGyration(ligand_mol)

    # Get ligand coordinates
    ligand_conformer = ligand_mol.GetConformers()[0]
    coordinates = ligand_conformer.GetPositions()
    ligand_coords = pd.DataFrame(coordinates, columns=["x_coord", "y_coord", "z_coord"])

    # Calculate center
    center_x = ligand_coords["x_coord"].mean().round(2)
    center_y = ligand_coords["y_coord"].mean().round(2)
    center_z = ligand_coords["z_coord"].mean().round(2)

    pocket_coordinates = {
        "center": [center_x, center_y, center_z],
        "size": [round(2.857 * float(radius_of_gyration), 2)] * 3,
    }
    return pocket_coordinates


def find_pocket_dogsitescorer(pdbpath: Path, method: str = "Volume") -> Dict[str, List[float]]:
    """
    Retrieves the binding site coordinates for a given PDB file using the DogSiteScorer method.

    Parameters:
    - pdbpath (Path): The path to the PDB file.
    - method (str): The method used to sort the binding sites. Default is 'Volume'.
                   Allowed values are 'Druggability_Score', 'Volume', 'Surface' or 'Depth'.

    Returns:
    - pocket_coordinates (list): The coordinates of the selected binding site pocket.
    """
    pdb_upload = upload_pdb_file(pdbpath)
    job_location = submit_dogsitescorer_job_with_pdbid(pdb_upload, "A", "")
    binding_site_df = get_dogsitescorer_metadata(job_location)
    best_binding_site = sort_binding_sites(binding_site_df, method)
    pocket_url = get_selected_pocket_location(job_location, best_binding_site)
    save_binding_site_to_file(pdbpath, pocket_url)
    pocket_coordinates = calculate_pocket_coordinates_from_pocket_pdb_file(str(pdbpath).replace(".pdb", "_pocket.pdb"))
    return pocket_coordinates


def find_pocket(
    mode: str,
    receptor: Path,
    ligand: Path = None,
    radius: int = 10,
    manual_pocket: str = None,
    dogsitescorer_method: str = "Volume",
) -> Dict[str, List[float]]:
    """
    Find and extract a docking pocket based on the specified mode.

    Args:
        mode (str): The mode for finding the docking pocket.
        receptor (Path): The path to the receptor file.
        ligand (Path, optional): The path to the ligand file.
        radius (int, optional): The radius for finding the docking pocket. Defaults to 10.
        manual_pocket (str, optional): The manually provided pocket coordinates.
        dogsitescorer_method (str, optional): The method to be used by DogSiteScorer. Defaults to 'Volume'.

    Returns:
        Dict[str, List[float]]: The definition of the docking pocket.

    Raises:
        ValueError: If an invalid pocket-finding method is specified.
        RuntimeError: If pocket extraction fails.
    """
    if mode not in VALID_METHODS:
        raise ValueError(f"Invalid pocket-finding method: {mode}. Valid methods are {', '.join(VALID_METHODS)}.")

    method_map = {
        "Reference": find_pocket_default,
        "RoG": find_pocket_rog,
        "Dogsitescorer": find_pocket_dogsitescorer,
        "Manual": parse_pocket_coordinates,
    }

    pocket_finder = method_map[mode]

    try:
        if mode == "Manual":
            pocket_definition = pocket_finder(manual_pocket)
        elif mode == "Dogsitescorer":
            pocket_definition = pocket_finder(receptor, dogsitescorer_method)
        elif mode in ["Reference", "RoG"]:
            if not ligand:
                raise ValueError(f"{mode} mode requires a ligand file")
            pocket_definition = pocket_finder(ligand, receptor, radius)

        pocket_path = extract_pocket(pocket_definition, receptor)
        if pocket_path is None:
            raise RuntimeError("Failed to extract pocket. The pocket might be empty.")
        return pocket_definition
    except Exception as e:
        raise RuntimeError(f"Error in pocket finding: {str(e)}") from e
