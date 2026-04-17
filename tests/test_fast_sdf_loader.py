import pytest
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import PandasTools
from typing import Any
import sys

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.fast_sdf_loader import fast_load_sdf

def write_test_sdf(filename: str, mol_data: list[tuple[str, dict[str, Any]]]) -> Path:
    """Write molecular data to an SDF file with proper naming."""
    output_path = Path(filename)
    writer = Chem.SDWriter(str(output_path))
    
    for smiles, props in mol_data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Set the molecule name using the ID property
            mol_name = props.get("ID", "")
            mol.SetProp("_Name", mol_name)
            
            # Set all other properties
            for key, value in props.items():
                mol.SetProp(key, str(value))
            writer.write(mol)
    
    writer.close()
    return output_path

class TestOptimizedSDFLoader:
    @pytest.fixture
    def test_data_dir(self):
        """Get the path to the test data directory."""
        return Path(__file__).parent.parent / "test_data" / "fast_sdf_loader"
    
    @pytest.fixture
    def simple_sdf_path(self, test_data_dir):
        """Path to the simple molecules SDF file."""
        return test_data_dir / "simple_molecules.sdf"
    
    @pytest.fixture
    def complex_sdf_path(self, test_data_dir):
        """Path to the complex molecules SDF file."""
        return test_data_dir / "complex_molecules.sdf"

    def test_basic_loading(self, simple_sdf_path):
        """Test basic functionality and comparison with RDKit."""
        # Load with optimized loader
        df_fast = fast_load_sdf(
            sdf_path=simple_sdf_path,
            molColName="Molecule",
            idName="ID"
        )

        # Load with RDKit
        df_rdkit = PandasTools.LoadSDF(
            str(simple_sdf_path),
            molColName='Molecule',
            idName='ID'
        )

        # Compare results
        assert len(df_fast) == len(df_rdkit)

        # RDKit's LoadSDF adds a SMILES column, so handle that
        fast_cols = set(df_fast.columns)
        rdkit_cols = {col for col in df_rdkit.columns if col != 'SMILES'}
        assert fast_cols == rdkit_cols

        # Compare IDs and molecule names
        assert all(df_fast['ID'] == df_rdkit['ID'])

        print(df_fast)
        print(df_rdkit)

    def test_parallel_processing(self, simple_sdf_path):
        """Test parallel processing with different CPU counts."""
        df_single = fast_load_sdf(simple_sdf_path, "Molecule", "ID", n_cpus=1)
        df_multi = fast_load_sdf(simple_sdf_path, "Molecule", "ID", n_cpus=2)
        
        # Results should be identical regardless of CPU count
        cols_to_compare = [col for col in df_single.columns if col != "Molecule"]
        pd.testing.assert_frame_equal(
            df_single[cols_to_compare],
            df_multi[cols_to_compare],
            check_dtype=False,
            atol=1e-6
        )

    def test_required_properties(self, complex_sdf_path):
        """Test selective property loading."""
        required_props = {'ID', 'LogP'}
        
        df = fast_load_sdf(
            complex_sdf_path,
            "Molecule",
            "ID",
            required_props=required_props
        )
        
        # Check that only required properties are loaded
        expected_cols = {'Molecule', 'ID', 'LogP'}
        assert set(df.columns) == expected_cols
