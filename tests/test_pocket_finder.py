from pathlib import Path
import sys
import warnings
import pytest

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.pocket_finding.pocket_finder import find_pocket

# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture
def common_test_data():
    """Set up common test data."""
    dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
    ligand = dockm8_path / "test_data/1fvv_l.sdf"
    receptor = dockm8_path / "test_data/1fvv_p.pdb"
    radius = 10
    return ligand, receptor, radius


@pytest.fixture
def cleanup(request):
    """Cleanup fixture to remove generated files after each test."""
    output_dir = dockm8_path / "test_data/"

    def remove_created_files():
        for file in output_dir.iterdir():
            if file.name in ["1fvv_p_pocket.pdb"]:
                file.unlink()

    request.addfinalizer(remove_created_files)


def test_reference_mode(common_test_data, cleanup):
    """Test pocket finding in reference mode."""
    ligand, receptor, radius = common_test_data
    pocket_definition = find_pocket(mode="Reference", receptor=receptor, ligand=ligand, radius=radius)
    expected_output = {"center": [-9.67, 207.73, 113.41], "size": [20.0, 20.0, 20.0]}
    assert pocket_definition == pytest.approx(expected_output, rel=1e-2)


def test_rog_mode(common_test_data, cleanup):
    """Test pocket finding in radius of gyration (RoG) mode."""
    ligand, receptor, _ = common_test_data
    pocket_definition = find_pocket(mode="RoG", receptor=receptor, ligand=ligand)
    expected_output = {"center": [-9.67, 207.73, 113.41], "size": [14.73, 14.73, 14.73]}
    assert pocket_definition == pytest.approx(expected_output, rel=1e-2)


def test_dogsitescorer_mode(common_test_data, cleanup):
    """Test pocket finding using Dogsitescorer."""
    _, receptor, _ = common_test_data
    pocket_definition = find_pocket(mode="Dogsitescorer", receptor=receptor)
    expected_output = {"center": [206.57, 113.81, 17.46], "size": [17.46, 17.46, 17.46]}
    assert pocket_definition == pytest.approx(expected_output, rel=1e-2)


def test_manual_mode(common_test_data, cleanup):
    """Test pocket finding with manual coordinates."""
    _, receptor, _ = common_test_data
    manual_pocket = "center:-30,220,110.1*size:20,20,20"
    pocket_definition = find_pocket(mode="Manual", receptor=receptor, manual_pocket=manual_pocket)
    expected_output = {"center": [-30, 220, 110.1], "size": [20, 20, 20]}
    assert pocket_definition == expected_output


def test_invalid_mode(common_test_data):
    """Test that an invalid mode raises a ValueError."""
    _, receptor, _ = common_test_data
    with pytest.raises(ValueError, match="Invalid pocket-finding method"):
        find_pocket(mode="InvalidMode", receptor=receptor)


def test_missing_ligand(common_test_data):
    """Test that Reference mode without a ligand raises a ValueError."""
    _, receptor, radius = common_test_data
    with pytest.raises(RuntimeError, match="Reference mode requires a ligand file"):
        find_pocket(mode="Reference", receptor=receptor, radius=radius)


def test_invalid_manual_pocket(common_test_data):
    """Test that an invalid manual pocket string raises a ValueError."""
    _, receptor, _ = common_test_data
    invalid_manual_pocket = "invalid:pocket:string"
    with pytest.raises(RuntimeError):
        find_pocket(mode="Manual", receptor=receptor, manual_pocket=invalid_manual_pocket)


def test_nonexistent_receptor():
    """Test that a nonexistent receptor file raises a RuntimeError."""
    nonexistent_receptor = Path("/path/to/nonexistent/receptor.pdb")
    with pytest.raises(RuntimeError):
        find_pocket(mode="Reference", receptor=nonexistent_receptor)


def test_different_dogsitescorer_methods(common_test_data):
    """Test that different DogSiteScorer methods produce different results."""
    _, receptor, _ = common_test_data
    volume_pocket = find_pocket(mode="Dogsitescorer", receptor=receptor, dogsitescorer_method="Volume")
    druggability_pocket = find_pocket(
        mode="Dogsitescorer", receptor=receptor, dogsitescorer_method="Druggability_Score"
    )
    assert volume_pocket != druggability_pocket


def test_consistent_results(common_test_data):
    """Test that repeated calls with the same parameters produce consistent results."""
    ligand, receptor, radius = common_test_data
    result1 = find_pocket(mode="Reference", receptor=receptor, ligand=ligand, radius=radius)
    result2 = find_pocket(mode="Reference", receptor=receptor, ligand=ligand, radius=radius)
    assert result1 == result2
