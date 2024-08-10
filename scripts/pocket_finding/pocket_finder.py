import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rdkit.Chem import Descriptors3D

from scripts.pocket_finding.dogsitescorer import (calculate_pocket_coordinates_from_pocket_pdb_file,
													get_dogsitescorer_metadata,
													get_selected_pocket_location,
													save_binding_site_to_file,
													sort_binding_sites,
													submit_dogsitescorer_job_with_pdbid,
													upload_pdb_file,
													)
from scripts.pocket_finding.utils import extract_pocket
from scripts.setup.software_manager import ensure_software_installed
from scripts.utilities.logging import printlog

# Assuming these utility functions are defined elsewhere
from scripts.utilities.utilities import load_molecule

POCKET_DETECTION_OPTIONS = ["Reference", "RoG", "Dogsitescorer", "p2rank", "Manual"]


class PocketFinderError(Exception):

	"""Base exception class for PocketFinder errors."""
	pass


class InvalidMethodError(PocketFinderError):

	"""Exception raised when an invalid pocket-finding method is specified."""
	pass


class PocketExtractionError(PocketFinderError):

	"""Exception raised when pocket extraction fails."""
	pass


class PocketFinder:

	"""
	A class to find and extract docking pockets using various methods.
	"""

	VALID_METHODS = ["Reference", "RoG", "Dogsitescorer", "p2rank", "Manual"]

	def __init__(self, software_path: Path = None):
		"""
		Initialize the PocketFinder.

		Args:
			software_path (Path, optional): Path to the software directory.
		"""
		self.software_path = software_path

	def find_pocket(self,
					mode: str,
					receptor: Path,
					ligand: Path = None,
					radius: int = 10,
					manual_pocket: str = None,
					dogsitescorer_method: str = 'Volume') -> Dict[str, List[float]]:
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
			InvalidMethodError: If an invalid pocket-finding method is specified.
			PocketExtractionError: If pocket extraction fails.
		"""
		if mode not in self.VALID_METHODS:
			raise InvalidMethodError(
				f"Invalid pocket-finding method: {mode}. Valid methods are {', '.join(self.VALID_METHODS)}.")

		method_map = {
			"Reference": self._find_pocket_default,
			"RoG": self._find_pocket_rog,
			"Dogsitescorer": self._find_pocket_dogsitescorer,
			"p2rank": self._find_pocket_p2rank,
			"Manual": self._parse_pocket_coordinates}

		pocket_finder = method_map[mode]

		try:
			if mode == "Manual":
				pocket_definition = pocket_finder(manual_pocket)
			elif mode == "Dogsitescorer":
				pocket_definition = pocket_finder(receptor, dogsitescorer_method)
			elif mode in ["Reference", "RoG"]:
				pocket_definition = pocket_finder(ligand, receptor, radius)
			elif mode == "p2rank":
				pocket_definition = pocket_finder(receptor, radius)

			pocket_path = extract_pocket(pocket_definition, receptor)
			if pocket_path is None:
				raise PocketExtractionError("Failed to extract pocket. The pocket might be empty.")
			return pocket_definition
		except Exception as e:
			raise PocketFinderError(f"Error in pocket finding: {str(e)}") from e

	def _find_pocket_default(self, ligand_file: Path, protein_file: Path, radius: int) -> Dict[str, List[float]]:
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

		pocket_coordinates = {"center": [center_x, center_y, center_z], "size": [float(radius) * 2] * 3, }
		return pocket_coordinates

	def _find_pocket_rog(self, ligand_file: Path, protein_file: Path, radius: int) -> Dict[str, List[float]]:
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
			"center": [center_x, center_y, center_z], "size": [round(2.857 * float(radius_of_gyration), 2)] * 3, }
		return pocket_coordinates

	def _find_pocket_dogsitescorer(self, pdbpath: Path, method: str = "Volume") -> Dict[str, List[float]]:
		"""
		Retrieves the binding site coordinates for a given PDB file using the DogSiteScorer method.

		Parameters:
		- pdbpath (Path): The path to the PDB file.
		- method (str): The method used to sort the binding sites. Default is 'Volume'. Allowed values are 'Druggability_Score', 'Volume', 'Surface' or 'Depth'.

		Returns:
		- pocket_coordinates (list): The coordinates of the selected binding site pocket.
		"""
		# Upload the PDB file
		pdb_upload = upload_pdb_file(pdbpath)
		# Submit the DoGSiteScorer job with the PDB ID
		job_location = submit_dogsitescorer_job_with_pdbid(pdb_upload, "A", "")
		# Get the metadata of the DoGSiteScorer job
		binding_site_df = get_dogsitescorer_metadata(job_location)
		# Sort the binding sites based on the given method
		best_binding_site = sort_binding_sites(binding_site_df, method)
		# Get the URL of the selected binding site
		pocket_url = get_selected_pocket_location(job_location, best_binding_site)
		# Save the binding site to a file
		save_binding_site_to_file(pdbpath, pocket_url)
		# Calculate the pocket coordinates from the saved PDB file
		pocket_coordinates = calculate_pocket_coordinates_from_pocket_pdb_file(
			str(pdbpath).replace(".pdb", "_pocket.pdb"))
		return pocket_coordinates

	@ensure_software_installed("P2RANK")
	def _find_pocket_p2rank(self, receptor: Path, radius: int) -> Dict[str, List[float]]:
		"""
		Finds the pocket coordinates using p2rank software.

		Args:
			receptor (Path): The path to the receptor file.
			radius (int): The radius of the pocket.

		Returns:
			Dict[str, List[float]]: A dictionary containing the pocket coordinates with keys 'center' and 'size'.

		Raises:
			FileNotFoundError: If the p2rank executable is not found.
			subprocess.CalledProcessError: If the p2rank command fails to execute.
			pd.errors.EmptyDataError: If the predictions file is empty or cannot be read.
			IndexError: If the predictions file doesn't contain the expected data.
			ValueError: If there's an issue with the pocket coordinates.
		"""
		try:
			p2rank_path = self.software_path / "p2rank" / "prank"

			output_dir = receptor.parent / "p2rank_output"
			os.makedirs(output_dir, exist_ok=True)

			p2rank_command = f'{p2rank_path} predict -f {receptor} -o {output_dir}'
			try:
				subprocess.run(p2rank_command, shell=True, check=True, capture_output=True, text=True)
			except subprocess.CalledProcessError as e:
				raise subprocess.CalledProcessError(
					e.returncode, e.cmd, "p2rank command failed. \n" + f"Stdout: {e.stdout} \n" + f"Stderr: {e.stderr}")

			predictions_file = output_dir / f"{receptor.name}_predictions.csv"
			if not predictions_file.exists():
				raise FileNotFoundError(f"p2rank Predictions file not found at {predictions_file}")

			try:
				df = pd.read_csv(predictions_file)
			except pd.errors.EmptyDataError:
				raise pd.errors.EmptyDataError(
					f"p2rank Predictions file is empty or cannot be read: {predictions_file}")

			df.columns = df.columns.str.replace(" ", "")

			pocket_coordinates = {
				"center": [df["center_x"][0], df["center_y"][0], df["center_z"][0]], "size": [float(radius) * 2] * 3, }

			if not all(isinstance(coord, (int, float)) for coord in pocket_coordinates["center"]):
				raise ValueError("Invalid pocket coordinates generated by P2Rank")

		finally:
			shutil.rmtree(output_dir, ignore_errors=True)

		return pocket_coordinates

	def _parse_pocket_coordinates(self, manual_pocket: str) -> Dict[str, List[float]]:
		"""
		Parses the pocket coordinates from the given pocket argument.

		Args:
			manual_pocket (str): The pocket argument to parse.

		Returns:
			Dict[str, List[float]]: A dictionary containing the parsed pocket coordinates.

		Raises:
			ValueError: If there is an error parsing the pocket coordinates.
		"""
		try:
			pocket_str = manual_pocket.split("*")
			pocket_coordinates = {
				key: list(map(float, value.split(","))) for item in pocket_str for key, value in [item.split(":")]}
			return pocket_coordinates
		except Exception as e:
			raise ValueError(f"Error parsing pocket coordinates: {e}. "
								"Make sure the pocket coordinates are in the format 'center:1,2,3*size:1,2,3'")

	def _get_ligand_coordinates(self, ligand_molecule) -> pd.DataFrame:
		"""
		Get the coordinates of a ligand molecule.

		Args:
			ligand_molecule: The ligand molecule.

		Returns:
			pd.DataFrame: A DataFrame containing the x, y, and z coordinates of the ligand molecule.
		"""
		ligand_conformer = ligand_molecule.GetConformers()[0]
		coordinates = ligand_conformer.GetPositions()
		dataframe = pd.DataFrame(coordinates, columns=["x_coord", "y_coord", "z_coord"])
		return self._add_coordinates(dataframe)

	def _add_coordinates(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		"""
		Add coordinates column to the given dataframe.

		Args:
			dataframe (pd.DataFrame): The input dataframe.

		Returns:
			pd.DataFrame: The dataframe with the coordinates column added.
		"""
		dataframe["coordinates"] = dataframe.apply(lambda row: [row["x_coord"], row["y_coord"], row["z_coord"]], axis=1)
		return dataframe


# Usage example:
# pocket_finder = PocketFinder(software_path=Path("/path/to/software"))
# pocket = pocket_finder.find_pocket(mode="Reference", receptor=Path("receptor.pdb"), ligand=Path("ligand.mol"), radius=10)
