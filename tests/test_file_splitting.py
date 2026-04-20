import pytest
import shutil
import math
from pathlib import Path
import sys

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.file_splitting import split_sdf, split_pdbqt_str

SDF_TEST_FILE = Path(dockm8_path) / "test_data" / "library.sdf"
PDBQT_TEST_FILE = (
    Path(dockm8_path)
    / "software"
    / "SCORCH-1.0.0"
    / "examples"
    / "predocked_1a0q"
    / "ligands"
    / "1a0q_docked_ligand.pdbqt"
)


def count_compounds_in_file(sdf_file):
    with open(sdf_file, "r") as f:
        return f.read().count("$$$$\n")


def calculate_expected_files(total_compounds, n_splits):
    compounds_per_split = math.ceil(total_compounds / n_splits)
    return math.ceil(total_compounds / compounds_per_split)


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


class TestSplitSDF:
    def setup_method(self):
        assert SDF_TEST_FILE.exists(), f"Test file {SDF_TEST_FILE} not found"
        self.total_compounds = count_compounds_in_file(SDF_TEST_FILE)

    def test_cpu_mode_splitting(self, temp_dir):
        for splits in [2, 4, 8]:
            run_dir = temp_dir / f"cpu_{splits}"
            run_dir.mkdir()
            result_dir = split_sdf(SDF_TEST_FILE, run_dir, mode="cpu", splits=splits)

            assert result_dir.exists()
            split_files = list(result_dir.glob("*.sdf"))
            assert len(split_files) > 0

            total_split = sum(count_compounds_in_file(f) for f in split_files)
            assert total_split == self.total_compounds

            expected = calculate_expected_files(self.total_compounds, splits)
            assert len(split_files) == expected

    def test_cpu_mode_splits_exceeding_compounds(self, temp_dir):
        result_dir = split_sdf(SDF_TEST_FILE, temp_dir, mode="cpu", splits=20)
        split_files = list(result_dir.glob("*.sdf"))
        assert len(split_files) == self.total_compounds
        for f in split_files:
            assert count_compounds_in_file(f) == 1

    def test_single_mode_splitting(self, temp_dir):
        result_dir = split_sdf(SDF_TEST_FILE, temp_dir, mode="single")

        assert result_dir.exists()
        split_files = list(result_dir.glob("*.sdf"))
        assert len(split_files) == self.total_compounds

        for sdf_file in split_files:
            assert count_compounds_in_file(sdf_file) == 1

        total_split = sum(count_compounds_in_file(f) for f in split_files)
        assert total_split == self.total_compounds

    def test_count_mode_splitting(self, temp_dir):
        for n in [1, 3, 5]:
            run_dir = temp_dir / f"count_{n}"
            run_dir.mkdir()
            result_dir = split_sdf(SDF_TEST_FILE, run_dir, mode="count", splits=n)

            split_files = list(result_dir.glob("*.sdf"))
            expected = calculate_expected_files(self.total_compounds, n)
            assert len(split_files) == expected

            total_split = sum(count_compounds_in_file(f) for f in split_files)
            assert total_split == self.total_compounds

    def test_file_content_preservation(self, temp_dir):
        with open(SDF_TEST_FILE, "r") as f:
            original_content = f.read()
            first_compound = original_content.split("$$$$\n")[0] + "$$$$\n"

        result_dir = split_sdf(SDF_TEST_FILE, temp_dir, mode="single")
        first_split_file = result_dir / "split_1.sdf"

        with open(first_split_file, "r") as f:
            split_content = f.read()

        assert split_content == first_compound

    def test_missing_splits_raises_error(self, temp_dir):
        with pytest.raises(ValueError, match="Number of splits must be provided"):
            split_sdf(SDF_TEST_FILE, temp_dir, mode="cpu")

    def test_nonexistent_file_raises_error(self, temp_dir):
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            split_sdf(temp_dir / "nonexistent.sdf", temp_dir, mode="single")

    def test_existing_output_directory_is_cleaned(self, temp_dir):
        output_dir = temp_dir / f"split_{SDF_TEST_FILE.stem}"
        output_dir.mkdir(parents=True)
        stale_file = output_dir / "stale.sdf"
        stale_file.write_text("stale data")

        result_dir = split_sdf(SDF_TEST_FILE, temp_dir, mode="single")
        assert result_dir.exists()
        assert not stale_file.exists()
        assert len(list(result_dir.glob("*.sdf"))) == self.total_compounds


class TestSplitPDBQT:
    def setup_method(self):
        assert PDBQT_TEST_FILE.exists(), f"Test file {PDBQT_TEST_FILE} not found"

    def test_pdbqt_splitting(self, temp_dir):
        test_file = temp_dir / PDBQT_TEST_FILE.name
        shutil.copy2(PDBQT_TEST_FILE, test_file)

        split_pdbqt_str(test_file)

        assert not test_file.exists()

        output_files = list(temp_dir.glob("*.pdbqt"))
        assert len(output_files) == 20

        for f in output_files:
            content = f.read_text()
            assert content.count("MODEL") == 1
            assert content.count("ENDMDL") == 1

    def test_pdbqt_output_naming(self, temp_dir):
        test_file = temp_dir / PDBQT_TEST_FILE.name
        shutil.copy2(PDBQT_TEST_FILE, test_file)
        stem = test_file.stem

        split_pdbqt_str(test_file)

        for i in range(1, 21):
            expected_file = temp_dir / f"{stem}_{i}.pdbqt"
            assert expected_file.exists(), f"Missing expected file: {expected_file.name}"

    def test_pdbqt_content_preservation(self, temp_dir):
        test_file = temp_dir / PDBQT_TEST_FILE.name
        shutil.copy2(PDBQT_TEST_FILE, test_file)

        with open(test_file, "r") as f:
            original_lines = f.readlines()
        original_model_count = sum(1 for l in original_lines if l.startswith("MODEL"))

        split_pdbqt_str(test_file)

        output_files = list(temp_dir.glob("*.pdbqt"))
        total_models = 0
        for f in output_files:
            content = f.read_text()
            if "MODEL" in content:
                total_models += 1

        assert total_models == original_model_count
