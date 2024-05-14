from pathlib import Path
import sys
import warnings

import pytest

# Search for 'DockM8' in parent directories
dockm8_path = next((p / 'DockM8' for p in Path(__file__).resolve().parents if (p / 'DockM8').is_dir()), None)
sys.path.append(str(dockm8_path))

print("DockM8 is in : "+str(dockm8_path))

from scripts.pocket_finding.pocket_finding import pocket_finder

# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

@pytest.fixture
def common_test_data():
    """Set up common test data."""
    ligand = dockm8_path / 'tests/test_files/1fvv_l.sdf'
    receptor =dockm8_path / 'tests/test_files/1fvv_p.pdb'
    software = dockm8_path / 'software'
    radius = 10
    return ligand, receptor, software, radius

def test_reference_mode(common_test_data):
    """Test pocket finding in reference mode."""
    ligand, receptor, software, radius = common_test_data
    pocket_definition = pocket_finder('Reference', receptor=receptor, ligand=ligand, radius=radius)
    expected_output = {'center': [-9.67, 207.73, 113.41], 'size': [20.0, 20.0, 20.0]}
    # Example assertion - customize based on expected structure
    assert pocket_definition == expected_output

def test_rog_mode(common_test_data):
    """Test pocket finding in radius of gyration (RoG) mode."""
    ligand, receptor, software, radius = common_test_data
    pocket_definition = pocket_finder('RoG', receptor=receptor, ligand=ligand)
    expected_output = {'center': [-9.67, 207.73, 113.41], 'size': [14.73, 14.73, 14.73]}
    assert pocket_definition == expected_output

def test_dogsitescorer_mode(common_test_data):
    """Test pocket finding using Dogsitescorer."""
    ligand, receptor, software, radius = common_test_data
    pocket_definition = pocket_finder('Dogsitescorer', receptor=receptor) 
    expected_output = {'center': [206.57, 113.81, 17.46], 'size': [17.46, 17.46, 17.46]}
    assert pocket_definition == expected_output

def test_manual_mode():
    """Test pocket finding with manual coordinates."""
    manual_pocket = 'center:-11,25.3,34.2*size:10,10,10'
    pocket_definition = pocket_finder("Manual", manual_pocket=manual_pocket)
    expected_output = {'center': [-11, 25.3, 34.2], 'size': [10, 10, 10]}
    assert pocket_definition == expected_output

def test_p2rank_mode(common_test_data):
    """Test pocket finding using p2rank."""
    ligand, receptor, software, radius = common_test_data
    pocket_definition = pocket_finder('p2rank', software=software, receptor=receptor, radius=radius)
    expected_output = {'center': (-15.4301, 196.0235, 98.3675), 'size': [20.0, 20.0, 20.0]}
    assert pocket_definition == expected_output