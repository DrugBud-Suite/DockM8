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

    assert gnina.stats["processed_batches"] == 1
    assert gnina.stats["failed_batches"] == 0
    assert gnina.stats["empty_results"] == 0

    # Verify temporary directory cleanup
    assert not gnina._temp_dir.exists()

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

    # Verify statistics were tracked
    assert hasattr(gnina, "stats")
    assert isinstance(gnina.stats, dict)
    assert "processed_batches" in gnina.stats
    assert "failed_batches" in gnina.stats
    assert "empty_results" in gnina.stats
    assert gnina.stats["processed_batches"] > 0
    assert gnina.stats["failed_batches"] == 0
    assert gnina.stats["empty_results"] == 0

    # Verify temporary directory cleanup
    assert not gnina._temp_dir.exists()

def test_gnina_dock_failure(test_data, tmp_path):
    protein_file, library_file, ligand_file, software, n_cpus, output_dir, pocket_definition = test_data

    # Create a GninaDocking instance with an invalid executable path
    invalid_software_path = tmp_path / "invalid_gnina_path"
    gnina_failure = GninaDocking(invalid_software_path)
    assert gnina_failure._temp_dir is not None
    assert gnina_failure._temp_dir.exists()
    run_info_path = gnina_failure._temp_dir / "run_info.json"
    assert run_info_path.exists()

    # Test dock_batch with an invalid executable - should return None
    result_file = gnina_failure.dock_batch(
        batch_file=ligand_file,
        protein_file=protein_file,
        pocket_definition=pocket_definition,
        exhaustiveness=1,
        n_poses=10,
    )
    assert result_file is None
    assert gnina_failure.stats["failed_batches"] == 1

    # Test the full docking process with an invalid executable
    output_sdf = output_dir / "gnina_docking_results_fail.sdf"
    final_output = gnina_failure.dock(
        library=library_file,
        protein_file=protein_file,
        pocket_definition=pocket_definition,
        exhaustiveness=1,
        n_poses=10,
        n_cpus=n_cpus,
        output_sdf=output_sdf,
    )
    assert final_output is None
    assert gnina_failure.stats["failed_batches"] > 1  # Should increment
    assert run_info_path.exists() # Run info should still be there
    with open(run_info_path) as f:
        run_info = json.load(f)
        assert run_info["status"] == "failed"
        assert "ERROR in docking" in run_info.get("error", "")

    # Verify temporary directory is moved to recovery
    recovery_dir_pattern = Path.home() / "dockm8_recovery" / f"failed_run_{gnina_failure._run_id}"
    assert recovery_dir_pattern.exists()
    assert (recovery_dir_pattern / "run_info.json").exists()
    shutil.rmtree(recovery_dir_pattern, ignore_errors=True) # Clean up recovery dir

def test_gnina_dock_empty_batch(test_data, tmp_path):
    protein_file, library_file, ligand_file, software, n_cpus, output_dir, pocket_definition = test_data

    gnina = GninaDocking(software)
    assert gnina._temp_dir is not None
    assert gnina._temp_dir.exists()

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
    assert gnina.stats["empty_results"] == 1
    assert gnina.stats["failed_batches"] == 0  # Should not be a failure

    # Verify temporary directory cleanup
    assert not gnina._temp_dir.exists()

def test_gnina_resume_dock(test_data):
    protein_file, library_file, ligand_file, software, n_cpus, output_dir, pocket_definition = test_data

    # Simulate a failed run by creating a temporary directory and run_info
    gnina_resume = GninaDocking(software)
    initial_temp_dir = gnina_resume._temp_dir
    assert initial_temp_dir.exists()

    # Create dummy processed files to simulate partial completion
    processed_dir = initial_temp_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    (processed_dir / "split_0_processed.sdf").touch()

    # Create dummy split files
    splits_dir = initial_temp_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    (splits_dir / "split_0.sdf").touch()
    (splits_dir / "split_1.sdf").touch()

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

    # Simulate a failure status
    with open(initial_temp_dir / "run_info.json") as f:
        run_info = json.load(f)
    run_info["status"] = "failed"
    with open(initial_temp_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    # Attempt to resume the docking run
    resumed_gnina = GninaDocking.resume_from_recovery(initial_temp_dir)
    assert resumed_gnina is not None
    assert resumed_gnina._run_id == gnina_resume._run_id

    output_sdf_resumed = output_dir / "gnina_docking_results_resumed.sdf"
    final_output_resumed = resumed_gnina.resume_dock(n_cpus=n_cpus)

    assert final_output_resumed is not None
    assert final_output_resumed.is_file()

    # Load and verify the final combined output
    final_df_resumed = PandasTools.LoadSDF(str(final_output_resumed), molColName="Molecule", idName="Pose ID")
    assert len(final_df_resumed) > 0

    # Verify the run info is updated to completed
    with open(resumed_gnina._temp_dir / "run_info.json") as f:
        updated_run_info = json.load(f)
    assert updated_run_info["status"] == "completed"

    # Verify temporary directory cleanup
    assert not resumed_gnina._temp_dir.exists()
    assert not initial_temp_dir.exists() # Ensure original temp dir is cleaned up

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
