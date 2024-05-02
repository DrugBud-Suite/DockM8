import pytest
from pathlib import Path
import sys
import os
from scripts.pocket_finding.main import pocket_finder
import warnings

# Ensure the scripts directory is accessible
sys.path.append(os.path.abspath('..'))


# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

@pytest.fixture
def common_test_data():
    """Set up common test data."""
    base_path = Path(__file__).parent.resolve()
    ligand = base_path / 'test_files/1fvv_l.sdf'
    receptor = base_path / 'test_files/1fvv_p.pdb'
    radius = 10
    return ligand, receptor, radius

def test_reference_mode(common_test_data):
    """Test pocket finding in reference mode."""
    ligand, receptor, radius = common_test_data
    pocket_definition = pocket_finder('Reference', receptor=receptor, ligand=ligand, radius=radius)
    print(pocket_definition)
    expected_output = {'center': [-9.67, 207.73, 113.41], 'size': [20.0, 20.0, 20.0]}
    # Example assertion - customize based on expected structure
    assert pocket_definition == expected_output

def test_rog_mode(common_test_data):
    """Test pocket finding in radius of gyration (RoG) mode."""
    ligand, receptor, _ = common_test_data
    pocket_definition = pocket_finder('RoG', receptor=receptor, ligand=ligand)
    print(pocket_definition)
    expected_output = {'center': [-9.67, 207.73, 113.41], 'size': [14.73, 14.73, 14.73]}
    assert pocket_definition == expected_output

def test_dogsitescorer_mode():
    """Test pocket finding using Dogsitescorer."""
    w_dir = Path('dockm8_testing/1fvv_p_protoss')
    receptor = Path('test_files/1fvv_p.pdb')
    pocket_definition = pocket_finder('Dogsitescorer', receptor=receptor, w_dir=w_dir, method='volume') 
    expected_output = {'center': [206.57, 113.81, 17.46], 'size': [17.46, 17.46, 17.46]}
    assert pocket_definition == expected_output

def test_manual_mode():
    """Test pocket finding with manual coordinates."""
    mode = 'center:-11,25.3,34.2*size:10,10,10'
    pocket_definition = pocket_finder(mode)
    expected_output = {'center': [-11, 25.3, 34.2], 'size': [10, 10, 10]}
    assert pocket_definition == expected_output
