import pytest
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors, rdMolDescriptors
from typing import Any
import tempfile
import os
import sys

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.fast_sdf_loader import fast_load_sdf
from scripts.utilities.utilities import parallel_SDF_loader

def write_test_sdf(filename: str, mol_data: list[tuple[str, dict[str, Any]]]) -> Path:
    """Write molecular data to an SDF file with proper naming"""
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
        """Get the path to the test data directory"""
        return Path(__file__).parent.parent / "test_data" / "fast_sdf_loader"
    
    @pytest.fixture
    def simple_sdf_path(self, test_data_dir):
        """Path to the simple molecules SDF file"""
        return test_data_dir / "simple_molecules.sdf"
    
    @pytest.fixture
    def complex_sdf_path(self, test_data_dir):
        """Path to the complex molecules SDF file"""
        return test_data_dir / "complex_molecules.sdf"

    def test_basic_loading(self, simple_sdf_path):
        """Test basic functionality and comparison with RDKit"""
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
        rdkit_cols = set(col for col in df_rdkit.columns if col != 'SMILES')
        assert fast_cols == rdkit_cols

        # Compare IDs and molecule names
        assert all(df_fast['ID'] == df_rdkit['ID'])

        print(df_fast)
        print(df_rdkit)

    def test_parallel_processing(self, simple_sdf_path):
        """Test parallel processing with different CPU counts"""
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
        """Test selective property loading"""
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

    @pytest.mark.performance
    def test_performance_comparison(self, test_data_dir):
        """Compare performance with RDKit's LoadSDF using large file"""
        import time
        
        # Load and multiply SMILES data for large file test
        smiles_path = test_data_dir / "SMILES.smi"
        smiles_data = self._load_smiles_from_file(smiles_path)
        large_data = smiles_data * 5000
        
        # Create large test file
        with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
            #large_sdf_path = write_test_sdf(tmp.name, large_data)
            large_sdf_path = Path("/home/tony/FINAL_RESULTS/v1_1/lit-pcba/mapk1/results/gnina/gnina_poses.sdf")
            
            try:
                # Time optimized loader
                print("Starting Fast SDF Loader")
                start = time.time()
                df_fast = fast_load_sdf(large_sdf_path, "Molecule", "ID")
                fast_time = time.time() - start
                
                # Time RDKit loader
                print("Starting RDKit SDF Loader")
                start = time.time()
                df_rdkit = PandasTools.LoadSDF(
                    str(large_sdf_path),
                    molColName='Molecule',
                    idName='ID'
                )
                rdkit_time = time.time() - start

                # Time RDKit loader
                print("Starting RDKit SDF Loader")
                start = time.time()
                df_pl = parallel_SDF_loader(
                    large_sdf_path,
                    molColName='Molecule',
                    idName='ID',
                )
                pl_time = time.time() - start
                
                print(f"Optimized loader: {fast_time:.2f}s")
                print(f"RDKit loader: {rdkit_time:.2f}s")
                print(f"Parallel loader: {pl_time:.2f}s")
                print(f"Speedup RDKit: {rdkit_time/fast_time:.2f}x")
                print(f"Speedup Parallel: {pl_time/fast_time:.2f}x")
                
                # Verify results
                #assert len(df_fast) == len(large_data)
                assert len(df_fast) == len(df_rdkit)
            
            finally:
                os.unlink(tmp.name)

    def _load_smiles_from_file(self, file_path: Path) -> list[tuple[str, dict[str, Any]]]:
        """Load SMILES and IDs from a .smi file"""
        import random
        import string
        
        def generate_random_string(length: int = 5) -> str:
            """Generate a random string of fixed length"""
            return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
        
        smiles_data = []
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    smiles, mol_id = line.strip().split()
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        props = {
                            "ID": mol_id,
                            "MW": str(Descriptors.MolWt(mol)),
                            "RotBonds": str(Descriptors.NumRotatableBonds(mol)),
                            "HBA": str(rdMolDescriptors.CalcNumHBA(mol)),
                            "HBD": str(rdMolDescriptors.CalcNumHBD(mol)),
                            "RandomID": generate_random_string()
                        }
                        smiles_data.append((smiles, props))
        return smiles_data
