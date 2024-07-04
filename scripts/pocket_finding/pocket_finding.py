import sys
from pathlib import Path

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.pocket_finding.default import find_pocket_default
from scripts.pocket_finding.dogsitescorer import find_pocket_dogsitescorer
from scripts.pocket_finding.manual import parse_pocket_coordinates
from scripts.pocket_finding.p2rank import find_pocket_p2rank
from scripts.pocket_finding.radius_of_gyration import find_pocket_RoG
from scripts.utilities.pocket_extraction import extract_pocket

POCKET_DETECTION_OPTIONS = ["Reference", "RoG", "Dogsitescorer", "p2rank", "Manual"]

def pocket_finder(mode: str,
					software: Path = None,
					receptor: Path = None,
					ligand: Path = None,
					radius: int = 10,
					manual_pocket: str = None,
					dogsitescorer_method: str = 'Volume',
					):
	"""
	Find and extract a docking pocket based on the specified mode.

	Parameters:
	- mode (str): The mode for finding the docking pocket. Possible values are:
		- "Reference": Find the docking pocket using a reference ligand and receptor.
		- "RoG": Find the docking pocket based on the radius of gyration of the ligand.
		- "Dogsitescorer": Find the docking pocket using the DogSiteScorer method.
		- "p2rank": Find the docking pocket using the p2rank method.
		- "Manual": Find the docking pocket based on manually provided pocket coordinates.

	- software (Path, optional): The path to the software folder. Required if mode is "p2rank".

	- receptor (Path, optional): The path to the receptor file. Required for all modes except "Manual".

	- ligand (Path, optional): The path to the ligand file. Required for modes "Reference" and "RoG".

	- radius (int, optional): The radius (in Angstroms) for finding the docking pocket. Default is 10.

	- manual_pocket (str, optional): The manually provided pocket coordinates. Required if mode is "Manual".

	- dogsitescorer_method (str, optional): The method to be used by DogSiteScorer. Default is 'Volume'.

	Returns:
	- pocket_definition: The definition of the docking pocket.

	"""
	if mode == "Reference":
		pocket_definition = find_pocket_default(ligand, receptor, radius)
		pocket_path = extract_pocket(pocket_definition, receptor)
		if pocket_path is None:
			raise ValueError("Failed to extract pocket. The pocket might be empty.")
		return pocket_definition
	elif mode == "RoG":
		pocket_definition = find_pocket_RoG(ligand, receptor)
		pocket_path = extract_pocket(pocket_definition, receptor)
		if pocket_path is None:
			raise ValueError("Failed to extract pocket. The pocket might be empty.")
		return pocket_definition
	elif mode == "Dogsitescorer":
		pocket_definition = find_pocket_dogsitescorer(receptor, dogsitescorer_method)
		pocket_path = extract_pocket(pocket_definition, receptor)
		if pocket_path is None:
			raise ValueError("Failed to extract pocket. The pocket might be empty.")
		return pocket_definition
	elif mode == "p2rank":
		pocket_definition = find_pocket_p2rank(software, receptor, radius)
		pocket_path = extract_pocket(pocket_definition, receptor)
		if pocket_path is None:
			raise ValueError("Failed to extract pocket. The pocket might be empty.")
		return pocket_definition
	elif mode == "Manual":
		pocket_definition = parse_pocket_coordinates(manual_pocket)
		pocket_path = extract_pocket(pocket_definition, receptor)
		if pocket_path is None:
			raise ValueError("Failed to extract pocket. The pocket might be empty.")
		return pocket_definition
	return pocket_definition
