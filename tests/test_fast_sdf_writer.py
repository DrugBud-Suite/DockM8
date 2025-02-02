import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os
import sys
import gzip
from rdkit.Chem import PandasTools

# Add project root to path (adjust as needed for your project structure)
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
project_path = tests_path.parent
sys.path.append(str(project_path))

from scripts.utilities.fast_sdf_loader import fast_load_sdf
from scripts.utilities.fast_sdf_writer import fast_write_sdf

class TestOptimizedSDFWriter:
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

    def test_basic_writing(self, simple_sdf_path):
        """Test basic SDF writing functionality"""
        # Read back using fast loader
        df_read = PandasTools.LoadSDF(
            simple_sdf_path,
            molColName='Molecule',
            idName='ID'
        )

        try:
            # Write using fast writer
            properties_to_write = ['LogP', 'MW']
            fast_write_sdf(
                df=df_read,
                output_path=Path(str(simple_sdf_path).replace(".sdf", "_written.sdf")),
                molColName='Molecule',
                idName='ID',
                properties=properties_to_write
            )

            df_read_2 = PandasTools.LoadSDF(
                Path(str(simple_sdf_path).replace(".sdf", "_written.sdf")),
                molColName='Molecule',
                idName='ID'
            )

            # Compare results
            assert len(df_read) == len(df_read_2)
            assert all(df_read['ID'] == df_read_2['ID'])

        finally:
            os.unlink(Path(str(simple_sdf_path).replace(".sdf", "_written.sdf")))

    def test_compression(self, simple_sdf_path):
        """Test writing compressed SDF files"""
        # Read the input data first
        df_read = fast_load_sdf(
            simple_sdf_path,
            molColName='Molecule',
            idName='ID'
        )

        with tempfile.NamedTemporaryFile(suffix='.sdf.gz', delete=False) as tmp:
            output_path = Path(tmp.name)
            try:
                properties_to_write = ['LogP', 'MW']
                fast_write_sdf(
                    df=df_read,
                    output_path=output_path,
                    molColName='Molecule',
                    idName='ID',
                    properties=properties_to_write,
                    compress=True
                )

                # Verify file exists and is not empty
                assert output_path.exists()
                assert output_path.stat().st_size > 0

                # Create a temporary uncompressed file
                uncompressed_path = Path(str(output_path).replace('.gz', ''))
                try:
                    # Decompress the file
                    with gzip.open(output_path, 'rt') as f_in:
                        with open(uncompressed_path, 'w') as f_out:
                            f_out.write(f_in.read())
                    
                    # Read the uncompressed file
                    df_read_compressed = fast_load_sdf(
                        uncompressed_path,
                        molColName='Molecule',
                        idName='ID'
                    )
                    
                    assert len(df_read_compressed) == len(df_read)
                    assert all(df_read_compressed['ID'] == df_read['ID'])
                    
                    # Verify properties were preserved
                    for prop in properties_to_write:
                        pd.testing.assert_series_equal(
                            df_read_compressed[prop].astype(float),
                            df_read[prop].astype(float),
                            check_names=False,
                            rtol=1e-5
                        )
                finally:
                    if uncompressed_path.exists():
                        os.unlink(uncompressed_path)

            finally:
                os.unlink(tmp.name)

    def test_property_handling(self, simple_sdf_path):
        """Test handling of different property types"""
        # Read the input data first
        df_read = fast_load_sdf(
            simple_sdf_path,
            molColName='Molecule',
            idName='ID'
        )

        # Add some additional property types
        df_read['IntProp'] = list(range(1, len(df_read) + 1))
        df_read['BoolProp'] = [i % 2 == 0 for i in range(len(df_read))]
        df_read['StringProp'] = [f'test{i}' for i in range(len(df_read))]

        with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
            output_path = Path(tmp.name)
            try:
                properties_to_write = ['LogP', 'MW', 'IntProp', 'BoolProp', 'StringProp']
                fast_write_sdf(
                    df=df_read,
                    output_path=output_path,
                    molColName='Molecule',
                    idName='ID',
                    properties=properties_to_write
                )

                df_read_props = fast_load_sdf(
                    output_path,
                    molColName='Molecule',
                    idName='ID'
                )

                assert len(df_read_props) == len(df_read)
                for prop in properties_to_write:
                    if prop in ['LogP', 'MW']:
                        pd.testing.assert_series_equal(
                            df_read_props[prop].astype(float),
                            df_read[prop].astype(float),
                            check_names=False,
                            rtol=1e-5
                        )
                    else:
                        assert all(df_read_props[prop].astype(str) == df_read[prop].astype(str))

            finally:
                os.unlink(tmp.name)

    def test_parallel_processing(self, simple_sdf_path):
        """Test parallel processing with different CPU counts"""
        # Read the input data first
        df_read = fast_load_sdf(
            simple_sdf_path,
            molColName='Molecule',
            idName='ID'
        )

        with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp2:
            path1, path2 = Path(tmp1.name), Path(tmp2.name)
            try:
                properties_to_write = ['LogP', 'MW']
                
                # Write with single CPU
                fast_write_sdf(
                    df=df_read,
                    output_path=path1,
                    molColName='Molecule',
                    idName='ID',
                    properties=properties_to_write,
                    n_cpus=1
                )

                # Write with multiple CPUs
                fast_write_sdf(
                    df=df_read,
                    output_path=path2,
                    molColName='Molecule',
                    idName='ID',
                    properties=properties_to_write,
                    n_cpus=2
                )

                # Read both files and compare contents
                df1 = fast_load_sdf(path1, molColName='Molecule', idName='ID')
                df2 = fast_load_sdf(path2, molColName='Molecule', idName='ID')

                assert len(df1) == len(df2)
                assert all(df1['ID'] == df2['ID'])
                for prop in properties_to_write:
                    pd.testing.assert_series_equal(
                        df1[prop].astype(float),
                        df2[prop].astype(float),
                        check_names=False,
                        rtol=1e-5
                    )

            finally:
                os.unlink(tmp1.name)
                os.unlink(tmp2.name)

    @pytest.mark.performance
    def test_performance_comparison(self, test_data_dir):
        """Compare performance with RDKit's SDWriter using large file"""
        import time
        from rdkit import Chem
        
        # Load an existing large SDF file for testing
        large_sdf_path = Path("/home/tony/FINAL_RESULTS/v1_1/lit-pcba/mapk1/results/gnina/gnina_poses.sdf")
        
        # First load the data into a DataFrame
        print("Loading test data...")
        df = PandasTools.LoadSDF(large_sdf_path, molColName="Molecule", idName="Pose ID")
        
        properties_to_write = ['ID', 'GNINA_Affinity']  # Add any properties present in your data
        
        # Create temporary files for output
        with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp2:
            
            fast_output = Path(tmp1.name)
            rdkit_output = Path(tmp2.name)
            
            try:
                # Time optimized writer
                print("Starting Fast SDF Writer")
                start = time.time()
                fast_write_sdf(
                    df=df,
                    output_path=fast_output,
                    molColName='Molecule',
                    idName='Pose ID',
                    properties=properties_to_write
                )
                fast_time = time.time() - start
                
                # Time RDKit writer
                print("Starting RDKit SDWriter")
                start = time.time()
                writer = Chem.SDWriter(str(rdkit_output))
                for _, row in df.iterrows():
                    mol = row['Molecule']
                    if mol is not None:
                        mol.SetProp('_Name', str(row['Pose ID']))
                        for prop in properties_to_write:
                            if prop != 'Pose ID' and prop in row:
                                mol.SetProp(prop, str(row[prop]))
                        writer.write(mol)
                writer.close()
                rdkit_time = time.time() - start
                
                # Print performance comparison
                print(f"Optimized writer: {fast_time:.2f}s")
                print(f"RDKit writer: {rdkit_time:.2f}s")
                print(f"Speedup: {rdkit_time/fast_time:.2f}x")
                
                # Verify results by reading back both files
                df_fast = fast_load_sdf(fast_output, "Molecule", "Pose ID")
                df_rdkit = fast_load_sdf(rdkit_output, "Molecule", "Pose ID")
                df_fast.sort_values("Pose ID", inplace=True)
                df_rdkit.sort_values("Pose ID", inplace=True)

                print(df_fast.head(20))
                print(df_rdkit.head(20))

                # Compare results
                assert len(df_fast) == len(df_rdkit)
                
            finally:
                # Clean up temporary files
                os.unlink(tmp1.name)
                os.unlink(tmp2.name)
