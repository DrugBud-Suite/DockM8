import sys
import json
from pathlib import Path
import shutil

import pytest
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.docking.gnina_docking import GninaDocking

@pytest.fixture
def test_data():
    dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
    protein_file = dockm8_path / "test_data/docking/prepared_protein.pdb"
    library_file = dockm8_path / "test_data/docking/prepared_library.sdf"
    ligand_file = dockm8_path / "test_data/docking/prepared_ligand.sdf"
    software = dockm8_path / "software"
    n_cpus = 10  # Keep it low for testing
    output_dir = dockm8_path / "test_data/docking/output"
    output_dir.mkdir(exist_ok=True)
    pocket_definition = {"center": [-9.67, 207.73, 113.41], "size": [20.0, 20.0, 20.0]}
    return protein_file, library_file, ligand_file, software, n_cpus, output_dir, pocket_definition

def test_gnina_dock_batch(test_data):
    protein_file, library_file, ligand_file, software, n_cpus, output_dir, pocket_definition = test_data

    gnina = GninaDocking(software)
    assert gnina._temp_dir is not None
    assert gnina._temp_dir.exists()
    assert (gnina._temp_dir / "run_info.json").exists()

    # Test dock_batch method with a single ligand file
    result_file = gnina.dock_batch(
        batch_file=ligand_file,
        protein_file=protein_file,
        pocket_definition=pocket_definition,
        exhaustiveness=1,
        n_poses=10,
    )

    assert result_file is not None
    assert result_file.is_file()

    # Verify the processed output directly
    output_df = PandasTools.LoadSDF(str(result_file), molColName="Molecule", idName="Pose ID")

    # Check the processed output structure
    assert "Pose ID" in output_df.columns
    assert "ID" in output_df.columns
    assert "Molecule" in output_df.columns
    assert "CNN-Score" in output_df.columns
    assert len(output_df) > 0

    # Check if the number of poses matches the input ligands
    suppl = Chem.SDMolSupplier(str(ligand_file))
    num_input_ligands = len([mol for mol in suppl if mol is not None])
    assert len(output_df) <= num_input_ligands * 10

    # Verify pose naming convention
    assert all(pid.startswith(id_ + "_GNINA_") for pid, id_ in zip(output_df["Pose ID"], output_df["ID"], strict=False))

    if gnina._temp_dir.exists():
        shutil.rmtree(gnina._temp_dir, ignore_errors=True)

def test_gnina_dock_full(test_data):
    protein_file, library_file, ligand_file, software, n_cpus, output_dir, pocket_definition = test_data

    gnina = GninaDocking(software)
    assert gnina._temp_dir is not None
    assert gnina._temp_dir.exists()
    assert (gnina._temp_dir / "run_info.json").exists()

    output_sdf = output_dir / "gnina_docking_results.sdf"
    final_output = gnina.dock(
        library=library_file,
        protein_file=protein_file,
        pocket_definition=pocket_definition,
        exhaustiveness=1,
        n_poses=10,
        n_cpus=n_cpus,
        output_sdf=output_sdf,
    )

    # Verify the output path was returned
    assert isinstance(final_output, Path)
    assert final_output.is_file()

    # Load and verify the final combined output
    final_df = PandasTools.LoadSDF(str(final_output), molColName="Molecule", idName="Pose ID")

    assert "Pose ID" in final_df.columns
    assert "CNN-Score" in final_df.columns
    assert "ID" in final_df.columns
    assert "Molecule" in final_df.columns
    assert len(final_df) > 0

    # Check if the number of poses matches the input ligands in the library
    library_suppl = Chem.SDMolSupplier(str(library_file))
    num_library_ligands = len([mol for mol in library_suppl if mol is not None])
    assert len(final_df) <= num_library_ligands * 10

    # Verify each compound has correct number of poses and proper ranking
    pose_counts = final_df.groupby("ID").size()
    assert all(count <= 10 for count in pose_counts)  # No more than 10 poses per compound

    # Verify scoring and ranking
    final_df["CNN-Score"] = pd.to_numeric(final_df["CNN-Score"], errors="coerce")
    for id_ in final_df["ID"].unique():
        compound_poses = final_df[final_df["ID"] == id_]
        # Scores should be sorted in descending order
        assert all(compound_poses["CNN-Score"].diff().fillna(0) <= 0)

    # Verify temporary directory cleanup
    assert not gnina._temp_dir.exists()

def test_gnina_dock_failure(test_data, tmp_path):
    protein_file, library_file, ligand_file, software, n_cpus, output_dir, pocket_definition = test_data
    
    # Test with valid initialization but invalid input files
    gnina = GninaDocking(software)
    assert gnina._temp_dir is not None
    assert gnina._temp_dir.exists()
    initial_temp_dir = gnina._temp_dir
    run_id = gnina._run_id
    
    # Test with non-existent protein file
    result_file = gnina.dock_batch(
        batch_file=ligand_file,
        protein_file=tmp_path / "nonexistent.pdb",
        pocket_definition=pocket_definition,
        exhaustiveness=1,
        n_poses=10,
    )
    assert result_file is None
    
    # Test with invalid pocket definition
    invalid_pocket = {"center": [1, 2], "size": [1, 2, 3]}  # Missing Z coordinate in center
    result_file = gnina.dock_batch(
        batch_file=ligand_file,
        protein_file=protein_file,
        pocket_definition=invalid_pocket,
        exhaustiveness=1,
        n_poses=10,
    )
    assert result_file is None

    # Create an empty SDF file
    empty_sdf = tmp_path / "empty.sdf"
    empty_sdf.touch()

    # Test dock_batch with an empty SDF file
    result_file = gnina.dock_batch(
        batch_file=empty_sdf,
        protein_file=protein_file,
        pocket_definition=pocket_definition,
        exhaustiveness=1,
        n_poses=10,
    )
    assert result_file is None
    
    # Test full docking with invalid setup
    output_sdf = output_dir / "gnina_docking_results_fail.sdf"
    final_output = gnina.dock(
        library=tmp_path / "nonexistent.sdf",  # Non-existent input file
        protein_file=protein_file,
        pocket_definition=pocket_definition,
        exhaustiveness=1,
        n_poses=10,
        n_cpus=n_cpus,
        output_sdf=output_sdf,
    )
    assert final_output is None

    # Verify that the failed run was properly saved
    recovery_dir = Path.home() / "dockm8_recovery" / f"failed_{gnina.name}_{run_id}"
    assert recovery_dir.exists(), "Recovery directory was not created"
    
    # Check run_info.json exists and contains correct failure information
    run_info_path = recovery_dir / "run_info.json"
    assert run_info_path.exists(), "run_info.json not found in recovery directory"
    
    with open(run_info_path) as f:
        run_info = json.load(f)
        assert run_info["status"] == "failed", "Run status not marked as failed"
        assert "error" in run_info, "Error message not found in run_info"
        assert "ERROR in docking" in run_info["error"], "Incorrect error message"

    

    # Verify that the original temporary directory no longer exists
    assert not initial_temp_dir.exists(), "Original temporary directory still exists"
    
    # Clean up the recovery directory after the test
    shutil.rmtree(recovery_dir, ignore_errors=True)


def test_gnina_resume_dock(test_data):
    protein_file, library_file, ligand_file, software, n_cpus, output_dir, pocket_definition = test_data
    
    # Simulate a failed run by creating a temporary directory and run_info
    gnina_resume = GninaDocking(software)
    initial_temp_dir = gnina_resume._temp_dir
    assert initial_temp_dir.exists()
    
    # Create processed directory with actual SDF content
    processed_dir = initial_temp_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    shutil.copy(ligand_file, processed_dir / "split_0_processed.sdf")
    
    # Create splits directory with actual SDF content
    splits_dir = initial_temp_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    for i in range(2):  # Create two split files
        shutil.copy(ligand_file, splits_dir / f"split_{i}.sdf")
    
    # Save run parameters
    params_to_save = {
        "library_file": str(library_file),
        "protein_file": str(protein_file),
        "pocket_definition": pocket_definition,
        "exhaustiveness": 1,
        "n_poses": 10,
        "output_sdf": str(output_dir / "gnina_docking_results_resumed.sdf"),
    }
    with open(initial_temp_dir / "run_parameters.json", "w") as f:
        json.dump(params_to_save, f, indent=2)
    
    # Update run info
    with open(initial_temp_dir / "run_info.json") as f:
        run_info = json.load(f)
    run_info.update({
        "status": "failed",
        "software_path": str(software),
        "run_id": gnina_resume._run_id
    })
    with open(initial_temp_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    
    # Resume the docking run
    resumed_gnina = GninaDocking.resume_from_recovery(initial_temp_dir)
    assert resumed_gnina is not None
    assert resumed_gnina._run_id == gnina_resume._run_id
    
    final_output_resumed = resumed_gnina.resume_dock(n_cpus=n_cpus)
    
    assert final_output_resumed is not None
    assert final_output_resumed.is_file()
    
    # Verify the output
    final_df_resumed = PandasTools.LoadSDF(
        str(final_output_resumed),
        molColName="Molecule",
        idName="Pose ID"
    )
    assert len(final_df_resumed) > 0
    
    # Verify cleanup
    assert not resumed_gnina._temp_dir.exists()
    assert not initial_temp_dir.exists()

def test_gnina_resume_dock_no_run_info(test_data, tmp_path):
    # Attempt to resume from a directory without run_info.json
    recovery_dir = tmp_path / "invalid_recovery"
    recovery_dir.mkdir()
    resumed_gnina = GninaDocking.resume_from_recovery(recovery_dir)
    assert resumed_gnina is None

def test_gnina_resume_dock_invalid_run_info(test_data, tmp_path):
    # Attempt to resume from a directory with invalid run_info.json
    recovery_dir = tmp_path / "invalid_recovery_json"
    recovery_dir.mkdir()
    with open(recovery_dir / "run_info.json", "w") as f:
        f.write("this is not json")
    resumed_gnina = GninaDocking.resume_from_recovery(recovery_dir)
    assert resumed_gnina is None
