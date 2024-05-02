import unittest
from pathlib import Path
import sys
import os

# Ensure the scripts directory is accessible
sys.path.append(os.path.abspath('..'))

from scripts.pocket_finding.main import pocket_finder
import warnings

# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestPocketFinder(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        base_path = Path(__file__).parent.resolve()
        self.ligand = base_path / 'test_files/1fvv_l.sdf'
        self.receptor = base_path / 'test_files/1fvv_p.pdb'
        self.radius = 10

    def test_reference_mode(self):
        """Test pocket finding in reference mode."""
        pocket_definition = pocket_finder('Reference', receptor=self.receptor, ligand=self.ligand, radius=self.radius)
        print(pocket_definition)
        expected_output = {'center': [-9.67, 207.73, 113.41], 'size': [20.0, 20.0, 20.0]}
        # Example assertion - customize based on expected structure
        self.assertEqual(pocket_definition, expected_output)

    def test_rog_mode(self):
        """Test pocket finding in radius of gyration (RoG) mode."""
        pocket_definition = pocket_finder('RoG', receptor=self.receptor, ligand=self.ligand)
        print(pocket_definition)
        expected_output = {'center': [-9.67, 207.73, 113.41], 'size': [14.73, 14.73, 14.73]}
        self.assertEqual(pocket_definition, expected_output)

    def test_dogsitescorer_mode(self):
        """Test pocket finding using Dogsitescorer."""
        w_dir = Path('dockm8_testing/1fvv_p_protoss')
        pocket_definition = pocket_finder('Dogsitescorer', receptor=self.receptor, w_dir=w_dir, method='volume') 
        expected_output = {'center': [206.57, 113.81, 17.46], 'size': [17.46, 17.46, 17.46]}
        self.assertEqual(pocket_definition, expected_output)

    def test_manual_mode(self):
        """Test pocket finding with manual coordinates."""
        mode = 'center:-11,25.3,34.2*size:10,10,10'
        pocket_definition = pocket_finder(mode)
        expected_output = {'center': [-11, 25.3, 34.2], 'size': [10, 10, 10]}
        self.assertEqual(pocket_definition, expected_output)
        
if __name__ == '__main__':
    unittest.main(verbosity=2)
