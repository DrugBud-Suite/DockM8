import os
import sys
from pathlib import Path
import pytest
from rdkit.Chem import PandasTools
from rdkit import Chem

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.library_preparation.conformer_generation.confgen_GypsumDL import generate_conformers_GypsumDL
from scripts.library_preparation.conformer_generation.confgen_rdkit import generate_conformers_RDKit
from scripts.library_preparation.protonation.protgen_GypsumDL import protonate_GypsumDL
from scripts.library_preparation.standardization.standardize import standardize_library
from scripts.library_preparation.library_preparation import prepare_library


@pytest.fixture
def common_test_data():
    """Set up common test data."""
    dockm8_path = next((p for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
    library = dockm8_path / "test_data/library.sdf"
    software = dockm8_path / "software"
    return library, software


def test_generate_conformers_GypsumDL(common_test_data):
    """Test generate_conformers_GypsumDL function."""
    library, software = common_test_data
    n_cpus = 4  # Set a fixed number of CPUs for testing

    # Load input library
    input_df = PandasTools.LoadSDF(str(library), molColName="Molecule", idName="ID")

    # Call the function
    output_df = generate_conformers_GypsumDL(input_df, software, n_cpus)

    # Check if the output DataFrame is not empty
    assert not output_df.empty

    # Check if the number of molecules in the input and output DataFrames are the same
    assert len(input_df) == len(output_df)

    # Check if the output DataFrame has the expected columns
    assert set(output_df.columns) == {"Molecule", "ID"}

    # Check if all molecules in the output DataFrame have 3D coordinates
    assert all(mol.GetNumConformers() > 0 for mol in output_df["Molecule"])

    # Check if the IDs in the input and output DataFrames match
    assert set(input_df["ID"]) == set(output_df["ID"])


def test_generate_conformers_RDKit(common_test_data):
    """Test generate_conformers_RDKit function."""
    library, software = common_test_data
    n_cpus = 4  # Set a fixed number of CPUs for testing
    forcefield = "MMFF"  # Choose either 'MMFF' or 'UFF'

    # Load input library
    input_df = PandasTools.LoadSDF(str(library), molColName="Molecule", idName="ID")

    # Call the function
    output_df = generate_conformers_RDKit(input_df, n_cpus, forcefield)

    # Check if the output DataFrame is not empty
    assert not output_df.empty

    # Check if the number of molecules in the output is less than or equal to the input
    # (some molecules might fail conformer generation)
    assert len(output_df) <= len(input_df)

    # Check if all molecules in the output DataFrame have 3D conformers
    assert all(mol.GetConformer().Is3D() for mol in output_df["Molecule"])

    # Check if the IDs in the output DataFrame are a subset of the input DataFrame
    assert set(output_df["ID"]).issubset(set(input_df["ID"]))


def test_protonate_GypsumDL(common_test_data):
    """Test protonate_GypsumDL function."""
    library, software = common_test_data
    n_cpus = 4  # Set a fixed number of CPUs for testing

    # Load input library
    input_df = PandasTools.LoadSDF(str(library), molColName="Molecule", idName="ID")

    # Call the function
    output_df = protonate_GypsumDL(input_df, software, n_cpus)

    # Check if the output DataFrame is not empty
    assert not output_df.empty

    # Check if the number of molecules in the output matches the input
    assert len(output_df) == len(input_df)

    # Check if the output DataFrame has the expected columns
    assert set(output_df.columns) == {"Molecule", "ID"}

    # Check if all molecules in the output DataFrame are valid
    assert all(mol is not None for mol in output_df["Molecule"])

    # Check if the IDs in the input and output DataFrames match
    assert set(input_df["ID"]) == set(output_df["ID"])

    # Check if any molecules have changed (protonation should modify some molecules)
    assert any(
        input_mol.GetNumAtoms() != output_mol.GetNumAtoms()
        for input_mol, output_mol in zip(input_df["Molecule"], output_df["Molecule"])
    )


def test_standardize_library(common_test_data):
    """Test standardize_library function."""
    library, software = common_test_data  # Properly unpack the tuple
    n_cpus = 4  # Set a fixed number of CPUs for testing

    # Load input library
    input_df = PandasTools.LoadSDF(str(library), molColName="Molecule", idName="ID")

    # Call the function
    standardized_df = standardize_library(input_df, n_cpus=n_cpus)

    # Check if the output DataFrame is not empty
    assert not standardized_df.empty

    # Check if the number of molecules in the output is less than or equal to the input
    # (some molecules might fail standardization)
    assert len(standardized_df) <= len(input_df)

    # Check if all molecules in the output DataFrame are valid
    assert all(mol is not None for mol in standardized_df["Molecule"])

    # Check if there are no duplicate IDs
    assert len(standardized_df["ID"].unique()) == len(standardized_df)

    # Check if SMILES strings are updated after standardization
    if "SMILES" in standardized_df.columns:
        assert all(
            Chem.MolToSmiles(mol) == smiles
            for mol, smiles in zip(standardized_df["Molecule"], standardized_df["SMILES"])
        )


def test_prepare_library_protonation(common_test_data):
    """Test library preparation with standardization and protonation."""
    library, software = common_test_data
    n_cpus = 4  # Set a fixed number of CPUs for testing

    # Load input library
    input_df = PandasTools.LoadSDF(str(library), molColName="Molecule", idName="ID")

    # Call the function
    final_df = prepare_library(library, "GypsumDL", "GypsumDL", software, n_cpus)

    # Check if the output DataFrame is not empty
    assert not final_df.empty

    # Check if the number of molecules in the output is less than or equal to the input
    # (some molecules might fail during preparation)
    assert len(final_df) <= len(input_df)

    # Check if the output DataFrame has the expected columns
    assert set(final_df.columns) == {"Molecule", "ID"}

    # Check if all molecules in the output DataFrame are valid
    assert all(mol is not None for mol in final_df["Molecule"])

    # Check if all molecules have 3D coordinates
    assert all(mol.GetNumConformers() > 0 for mol in final_df["Molecule"])


def test_prepare_library_no_protonation(common_test_data):
    """Test library preparation without protonation."""
    library, software = common_test_data
    n_cpus = 4  # Set a fixed number of CPUs for testing

    # Load input library
    input_df = PandasTools.LoadSDF(str(library), molColName="Molecule", idName="ID")

    # Call the function
    final_df = prepare_library(library, "None", "GypsumDL", software, n_cpus)

    # Check if the output DataFrame is not empty
    assert not final_df.empty

    # Check if the number of molecules in the output is less than or equal to the input
    assert len(final_df) <= len(input_df)

    # Check if the output DataFrame has the expected columns
    assert set(final_df.columns) == {"Molecule", "ID"}

    # Check if all molecules in the output DataFrame are valid
    assert all(mol is not None for mol in final_df["Molecule"])

    # Check if all molecules have 3D coordinates
    assert all(mol.GetNumConformers() > 0 for mol in final_df["Molecule"])


def test_prepare_library_invalid_conformers(common_test_data):
    """Test library preparation with invalid conformers method."""
    library, software = common_test_data
    n_cpus = 4  # Set a fixed number of CPUs for testing

    # Check if the function raises a ValueError for an invalid conformers method
    with pytest.raises(ValueError):
        prepare_library(library, "GypsumDL", "InvalidMethod", software, n_cpus)
