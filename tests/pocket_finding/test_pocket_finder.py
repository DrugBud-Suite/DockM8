from pathlib import Path
import sys
import warnings
import pytest

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.pocket_finding.pocket_finder import PocketFinder

# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture
def pocket_finder():
	"""Fixture to create a PocketFinder instance."""
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	software_path = dockm8_path / "software"
	return PocketFinder(software_path=software_path)


@pytest.fixture
def common_test_data():
	"""Set up common test data."""
	ligand = dockm8_path / "tests/test_files/pocket_finder/1fvv_l.sdf"
	receptor = dockm8_path / "tests/test_files/pocket_finder/1fvv_p.pdb"
	radius = 10
	return ligand, receptor, radius


@pytest.fixture
def cleanup(request):
	"""Cleanup fixture to remove generated files after each test."""
	output_dir = dockm8_path / "tests/test_files/pocket_finder/"

	def remove_created_files():
		for file in output_dir.iterdir():
			if file.name in ["1fvv_p_pocket.pdb"]:
				file.unlink()

	request.addfinalizer(remove_created_files)


def test_reference_mode(pocket_finder, common_test_data, cleanup):
	"""Test pocket finding in reference mode."""
	ligand, receptor, radius = common_test_data
	pocket_definition = pocket_finder.find_pocket("Reference", receptor=receptor, ligand=ligand, radius=radius)
	expected_output = {"center": [-9.67, 207.73, 113.41], "size": [20.0, 20.0, 20.0]}
	assert pocket_definition == pytest.approx(expected_output, rel=1e-2)


def test_rog_mode(pocket_finder, common_test_data, cleanup):
	"""Test pocket finding in radius of gyration (RoG) mode."""
	ligand, receptor, _ = common_test_data
	pocket_definition = pocket_finder.find_pocket("RoG", receptor=receptor, ligand=ligand)
	expected_output = {"center": [-9.67, 207.73, 113.41], "size": [14.73, 14.73, 14.73]}
	assert pocket_definition == pytest.approx(expected_output, rel=1e-2)


def test_dogsitescorer_mode(pocket_finder, common_test_data, cleanup):
	"""Test pocket finding using Dogsitescorer."""
	_, receptor, _ = common_test_data
	pocket_definition = pocket_finder.find_pocket("Dogsitescorer", receptor=receptor)
	expected_output = {"center": [206.57, 113.81, 17.46], "size": [17.46, 17.46, 17.46]}
	assert pocket_definition == pytest.approx(expected_output, rel=1e-2)


def test_manual_mode(pocket_finder, common_test_data, cleanup):
	"""Test pocket finding with manual coordinates."""
	_, receptor, _ = common_test_data
	manual_pocket = "center:-30,220,110.1*size:20,20,20"
	pocket_definition = pocket_finder.find_pocket("Manual", receptor=receptor, manual_pocket=manual_pocket)
	expected_output = {"center": [-30, 220, 110.1], "size": [20, 20, 20]}
	assert pocket_definition == expected_output


def test_p2rank_mode(pocket_finder, common_test_data, cleanup):
	"""Test pocket finding using p2rank."""
	_, receptor, radius = common_test_data
	pocket_definition = pocket_finder.find_pocket("p2rank", receptor=receptor, radius=radius)
	expected_output = {"center": [-15.4301, 196.0235, 98.3675], "size": [20.0, 20.0, 20.0]}
	assert pocket_definition == pytest.approx(expected_output, rel=1e-2)


def test_invalid_mode(pocket_finder, common_test_data):
	"""Test that an invalid mode raises an InvalidMethodError."""
	_, receptor, _ = common_test_data
	from scripts.pocket_finding.pocket_finder import InvalidMethodError
	with pytest.raises(InvalidMethodError):         # Assuming InvalidMethodError is a ValueError subclass
		pocket_finder.find_pocket("InvalidMode", receptor=receptor)


def test_missing_ligand(pocket_finder, common_test_data):
	"""Test that Reference mode without a ligand raises an error."""
	_, receptor, radius = common_test_data
	from scripts.pocket_finding.pocket_finder import PocketFinderError
	with pytest.raises(PocketFinderError):          # or whatever error type is appropriate
		pocket_finder.find_pocket("Reference", receptor=receptor, radius=radius)


def test_invalid_manual_pocket(pocket_finder, common_test_data):
	"""Test that an invalid manual pocket string raises an error."""
	_, receptor, _ = common_test_data
	invalid_manual_pocket = "invalid:pocket:string"
	with pytest.raises(ValueError):
		pocket_finder.find_pocket("Manual", receptor=receptor, manual_pocket=invalid_manual_pocket)


def test_nonexistent_receptor(pocket_finder):
	"""Test that a nonexistent receptor file raises an error."""
	nonexistent_receptor = Path("/path/to/nonexistent/receptor.pdb")
	from scripts.pocket_finding.pocket_finder import PocketFinderError
	with pytest.raises(PocketFinderError):
		pocket_finder.find_pocket("Reference", receptor=nonexistent_receptor)


def test_different_dogsitescorer_methods(pocket_finder, common_test_data):
	"""Test that different DogSiteScorer methods produce different results."""
	_, receptor, _ = common_test_data
	volume_pocket = pocket_finder.find_pocket("Dogsitescorer", receptor=receptor, dogsitescorer_method="Volume")
	druggability_pocket = pocket_finder.find_pocket("Dogsitescorer",
													receptor=receptor,
													dogsitescorer_method="Druggability_Score")
	assert volume_pocket != druggability_pocket


def test_consistent_results(pocket_finder, common_test_data):
	"""Test that repeated calls with the same parameters produce consistent results."""
	ligand, receptor, radius = common_test_data
	result1 = pocket_finder.find_pocket("Reference", receptor=receptor, ligand=ligand, radius=radius)
	result2 = pocket_finder.find_pocket("Reference", receptor=receptor, ligand=ligand, radius=radius)
	assert result1 == result2
