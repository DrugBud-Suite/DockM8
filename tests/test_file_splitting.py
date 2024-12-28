import pytest
from pathlib import Path
import sys
import math

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.file_splitting import split_sdf_str

SDF_TEST_FILE = f"{dockm8_path}/test_data/library.sdf"


def count_sdf_files(directory):
    """Helper function to count SDF files in directory."""
    return len(list(Path(directory).glob("*.sdf")))


def count_compounds(sdf_file):
    """Helper function to count compounds in an SDF file."""
    with open(sdf_file, "r") as f:
        return f.read().count("$$$$\n")


def calculate_expected_files(total_compounds, n_cpus):
    """Calculate expected number of files based on splitting logic."""
    if total_compounds > 100000:
        compounds_per_file = max(1, math.ceil(total_compounds // n_cpus // 8))
    else:
        compounds_per_file = max(1, math.ceil(total_compounds // n_cpus // 2))
    return math.ceil(total_compounds / compounds_per_file)


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture to provide temporary directory."""
    return tmp_path


class TestSDFSplitter:
    def setup_method(self):
        """Setup method to verify test file exists."""
        assert Path(SDF_TEST_FILE).exists(), f"Test file {SDF_TEST_FILE} not found"
        self.total_compounds = count_compounds(SDF_TEST_FILE)
        print(f"Total compounds in test file: {self.total_compounds}")

    def test_cpu_mode_splitting(self, temp_dir):
        # Test CPU-based splitting with different CPU counts
        for n_cpus in [2, 4, 8]:
            result_dir = split_sdf_str(temp_dir, SDF_TEST_FILE, split_mode="cpu", n_cpus=n_cpus)

            assert result_dir.exists()
            split_files = list(result_dir.glob("*.sdf"))
            assert len(split_files) > 0

            # Verify total compounds are preserved
            total_split_compounds = sum(count_compounds(f) for f in split_files)
            assert total_split_compounds == self.total_compounds

            # Verify file count is reasonable
            expected_files = calculate_expected_files(self.total_compounds, n_cpus)
            assert len(split_files) == expected_files

            # Verify rough distribution of compounds
            compounds_per_file = [count_compounds(f) for f in split_files]
            max_compounds = max(compounds_per_file)
            min_compounds = min(compounds_per_file)
            # Allow for some variation in compound distribution
            assert max_compounds - min_compounds <= math.ceil(self.total_compounds / expected_files)

            # Clean up for next iteration
            for file in split_files:
                file.unlink()
            result_dir.rmdir()

    def test_single_mode_splitting(self, temp_dir):
        result_dir = split_sdf_str(temp_dir, SDF_TEST_FILE, split_mode="single")

        assert result_dir.exists()
        split_files = list(result_dir.glob("*.sdf"))

        # Verify number of files matches number of compounds
        assert len(split_files) == self.total_compounds

        # Verify each file contains exactly one compound
        for sdf_file in split_files:
            assert count_compounds(sdf_file) == 1

        # Verify total compounds are preserved
        total_split_compounds = sum(count_compounds(f) for f in split_files)
        assert total_split_compounds == self.total_compounds

    def test_file_content_preservation(self, temp_dir):
        """Test that the content of compounds is preserved during splitting."""
        # Get content of first compound from original file
        with open(SDF_TEST_FILE, "r") as f:
            original_content = f.read()
            first_compound = original_content.split("$$$$\n")[0] + "$$$$\n"

        # Split in single mode and check first compound
        result_dir = split_sdf_str(temp_dir, SDF_TEST_FILE, split_mode="single")
        first_split_file = sorted(result_dir.glob("*.sdf"))[0]

        with open(first_split_file, "r") as f:
            split_content = f.read()

        assert split_content == first_compound

    def test_invalid_split_mode(self, temp_dir):
        with pytest.raises(ValueError, match="split_mode must be either 'cpu' or 'single'"):
            split_sdf_str(temp_dir, SDF_TEST_FILE, split_mode="invalid")

    def test_missing_n_cpus(self, temp_dir):
        with pytest.raises(ValueError, match="n_cpus must be provided when split_mode='cpu'"):
            split_sdf_str(temp_dir, SDF_TEST_FILE, split_mode="cpu")

    def test_existing_output_directory(self, temp_dir):
        # Create the output directory first
        output_dir = temp_dir / f"split_{Path(SDF_TEST_FILE).name.replace('.sdf', '')}"
        output_dir.mkdir(parents=True)

        # Should not raise an error
        result_dir = split_sdf_str(temp_dir, SDF_TEST_FILE, split_mode="single")
        assert result_dir.exists()
        assert count_sdf_files(result_dir) == self.total_compounds
